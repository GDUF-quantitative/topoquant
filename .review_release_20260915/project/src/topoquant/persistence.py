"""持久化提交原语（依赖树最底层）。

把「库被占用」的可重试判定与带退避重试的提交从 ``pipeline`` 下移至此，
消除 ``storage`` → ``pipeline`` 的反向依赖，使 ``storage`` 层也能用带重试的提交。

本模块只依赖 ``sqlite3`` / ``logging`` / ``time``，刻意不 import 任何本项目其它模块，
以避免与 ``storage`` / ``pipeline`` 形成循环 import。``pipeline`` 通过同名薄封装
（``_commit_with_retry`` / ``_is_sqlite_busy``）转发，全部既有调用点与测试 monkeypatch 不变。
"""

from __future__ import annotations

import logging
import sqlite3
import time

LOGGER = logging.getLogger(__name__)

# 可重试的 SQLite 占用错误码：官方枚举，跨版本/语言环境稳定。
_SQLITE_BUSY = 5
_SQLITE_LOCKED = 6


def is_sqlite_busy(exc: BaseException) -> bool:
    """判断异常是否为「库被占用」类的可重试错误（而非 schema/约束等确定性错误）。

    优先用扩展错误码 ``exc.sqlite_errorcode``（Python 3.11+ 稳定支持）：它是
    SQLite 官方定义的枚举，不随版本/语言环境变化——SQLITE_BUSY=5、SQLITE_LOCKED=6。
    英文字符串匹配仅作兜底（老版本或异常缺失该属性时）。其余 OperationalError（如
    ``no such table``、``too many SQL variables``）属于代码或数据缺陷，
    重试毫无意义，必须立刻抛出，绝不能被退避循环掩盖。
    """
    if not isinstance(exc, sqlite3.OperationalError):
        return False
    code = getattr(exc, "sqlite_errorcode", None)
    if code is not None:
        return code in (_SQLITE_BUSY, _SQLITE_LOCKED)
    text = str(exc).lower()
    return "locked" in text or "busy" in text


def commit_with_retry(
    db: sqlite3.Connection,
    *,
    what: str,
    attempts: int,
    base_delay: float,
) -> None:
    """提交事务；遇到「库被占用」时有限次退避重试，且每次等待都写日志。

    与直接 ``db.commit()`` 的差别只在异常路径：

    - 正常情况：行为完全一致（一次 commit 成功即返回），无额外开销；
    - ``database is locked`` / ``database is busy``：打 WARNING 说明正在等谁、
      第几次重试、还剩几次，退避后重试；``attempts`` 次仍失败才抛出原异常；
    - 其它 ``OperationalError``（如 ``too many SQL variables``）与其它异常：
      **立即原样抛出**，不做任何重试——那是缺陷，不是争抢。

    之所以必须重试而不是直接抛：``connect()`` 已设 ``busy_timeout=30000``，
    能抛到这里说明对方持锁已超过 30 秒。此时放弃会丢掉整个阶段的进度，
    而多等几轮通常就能拿到锁（对方多半是另一个阶段的短事务或 checkpoint）。
    """
    last_exc: BaseException | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            db.commit()
            if attempt > 1:
                LOGGER.warning("提交%s成功（第 %d 次尝试）", what, attempt)
            return
        except Exception as exc:
            if not is_sqlite_busy(exc):
                raise
            last_exc = exc
            if attempt >= max(1, attempts):
                break
            delay = min(30.0, base_delay * (2 ** (attempt - 1)))
            LOGGER.warning(
                "提交%s时数据库被占用（%s）：已在 busy_timeout 上等待超时，"
                "%.1fs 后重试（第 %d/%d 次）。请确认没有第二个实例或数据库浏览器正在写同一个库",
                what,
                exc,
                delay,
                attempt,
                attempts,
            )
            time.sleep(delay)
    # 防御性断言（S101 豁免）：重试耗尽后 last_exc 必非 None（首次 busy 即赋值）。
    assert last_exc is not None  # noqa: S101
    LOGGER.error(
        "提交%s连续 %d 次均因数据库被占用而失败，放弃本次提交：%s",
        what,
        max(1, attempts),
        last_exc,
    )
    raise last_exc


class CommitBatch:
    """按项数节流提交，退出时保证最后一批必落盘（R2-S3）。

    替代散落的 ``stock_index % 25`` / ``index % 100`` / ``_written % 200`` /
    ``index % 50`` 魔法数提交节奏：调用方统一用 ``with CommitBatch(...) as batch:``
    包裹循环、循环体内 ``batch.tick()``，提交节奏（``every``）由 R5 的
    ``StagePolicy.commit_every_*`` 统一提供。

    - 每 ``tick()`` 计数 +1，达 ``every`` 阈值才提交一次（带退避重试）；
    - 正常退出（``__exit__`` 无异常）补提交尚未达阈值的尾批；
    - 异常路径（``__exit__`` 收到异常）**不提交、不吞异常**，与原先
      「漏提交静默丢结果」的修复口径一致。
    """

    def __init__(
        self,
        db: sqlite3.Connection,
        *,
        every: int,
        what: str,
        attempts: int = 5,
        base_delay: float = 1.0,
    ) -> None:
        self._db = db
        self._every = max(1, int(every))
        self._what = what
        self._attempts = attempts
        self._base_delay = base_delay
        self._count = 0

    def __enter__(self) -> CommitBatch:
        return self

    def tick(self) -> None:
        """计数 +1；达阈值则提交一次（带退避重试）。"""
        self._count += 1
        if self._count % self._every == 0:
            commit_with_retry(
                self._db, what=self._what, attempts=self._attempts, base_delay=self._base_delay
            )

    def flush(self) -> None:
        """补提交尚未达阈值的尾批（正常收尾时由 ``__exit__`` 自动调用）。"""
        if self._count % self._every != 0:
            commit_with_retry(
                self._db, what=self._what, attempts=self._attempts, base_delay=self._base_delay
            )

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        # 异常路径：不提交、不吞异常（返回 None → 原样上抛）。
        if exc_type is None:
            self.flush()
