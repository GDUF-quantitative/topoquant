"""SQLite 提交重试与阶段完成标记原语（Phase 3 从 god-module 抽取）。

机械搬迁（byte-for-byte）：原 ``pipeline.py`` 中的 SQLITE_BUSY 判定转发
（_is_sqlite_busy）、提交重试（_commit_with_retry）与阶段完成标记
（_write_incomplete_marker / _clear_incomplete_marker）平移至此；``pipeline.py``
以 ``from .db_utils import (...)`` 重导出，外部 ``pipeline._commit_with_retry``
等引用零破坏。常量 COMMIT_RETRY_ATTEMPTS / COMMIT_RETRY_BASE_DELAY 一并迁入。
函数体逐字节一致，无循环依赖（仅依赖 persistence/policy 与标准库）。
"""

from __future__ import annotations

import sqlite3
from contextlib import suppress
from pathlib import Path

from .persistence import (
    commit_with_retry as _commit_with_retry_impl,
)
from .persistence import (
    is_sqlite_busy as _is_sqlite_busy_impl,
)
from .policy import POLICY

# ── 落库阻塞防护参数（本次修复）────────────────────────────────────
# 匹配/预测阶段的父进程持有一条长生命周期写连接，每 N 项 commit 一次。
# 若此刻另一个进程（第二个 run.py 实例、DB 浏览器、编辑器的 SQLite 插件等）
# 正持有读/写锁，commit 会先在 busy_timeout(30s) 上静默阻塞、随后抛
# sqlite3.OperationalError: database is locked。
# 该异常原先会从 consume 回调一路抛穿整个阶段，把已算好的结果连带作废，
# 且外观就是「进度条停在半路、无报错、进程不退出」。
# 这里给 commit 加「有限次退避重试 + 每次重试都打 WARNING」，做到两点：
#   1) 短暂的跨进程争抢能自愈，不再因一次 SQLITE_BUSY 断掉整轮；
#   2) 阻塞过程始终可见，不会再出现「静默等 30 秒」。
COMMIT_RETRY_ATTEMPTS = POLICY.commit_retry_attempts  # 兼容别名（R5）
COMMIT_RETRY_BASE_DELAY = POLICY.commit_retry_base_delay  # 兼容别名（R5）


def _is_sqlite_busy(exc: BaseException) -> bool:
    """保留原名与原签名（R2-S2）：实际判定已下移 ``persistence.is_sqlite_busy``。

    现有调用点与测试 monkeypatch 均不需改动；此处仅作薄封装转发。
    """
    return _is_sqlite_busy_impl(exc)


def _write_incomplete_marker(work_dir: Path, stage: str, exc: BaseException) -> None:
    """完成标记落库失败时，在 work_dir 写一个告警标记文件。

    目的：让「库不可写 → 本轮结果无法被续算复用（下次被迫全量重算）」这一严重状态
    显式可见、可被监控/运维捕获。只写文件系统、不依赖 DB——库不可写时文件系统通常
    仍可写；若文件系统同样不可写，调用方已用 try/except 兜底忽略，不会二次抛出。
    """
    from datetime import datetime

    marker = work_dir / f".{stage}_incomplete"
    marker.write_text(
        f"stage={stage}\ntime={datetime.now().isoformat()}\nerror={exc!r}\n",
        encoding="utf-8",
    )


def _clear_incomplete_marker(work_dir: Path, stage: str) -> None:
    """阶段起始 best-effort 清除上一轮可能残留的「完成标记写入失败」告警标记。

    若该轮其实已成功落盘，此处删除的只是过期标记；若确实未落盘（签名未写成），
    本轮仍会重写（并在失败时再次落 marker），不影响正确性。
    """
    with suppress(OSError):
        (work_dir / f".{stage}_incomplete").unlink()


def _commit_with_retry(
    db: sqlite3.Connection,
    *,
    what: str,
    attempts: int = COMMIT_RETRY_ATTEMPTS,
    base_delay: float = COMMIT_RETRY_BASE_DELAY,
) -> None:
    """保留原名与原签名（R2-S2）：实际提交逻辑已下移 ``persistence.commit_with_retry``。

    转发时补上常量默认值（来自 R5 的 ``POLICY.commit_retry_attempts`` /
    ``POLICY.commit_retry_base_delay``）。现有 8 处调用点与测试 monkeypatch 均不需改动。
    """
    _commit_with_retry_impl(db, what=what, attempts=attempts, base_delay=base_delay)
