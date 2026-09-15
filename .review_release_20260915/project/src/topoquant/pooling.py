"""阶段级并行执行原语（Phase 3 从 god-module 抽取）。

机械搬迁（byte-for-byte）：原 ``pipeline.py`` 中的进程池创建/终止、per-task 超时
执行器（_run_pool_stage）、单/多 worker 统一隔离执行器（_run_consumer）与阶段级
看门狗（_StageWatchdog）平移至此；``pipeline.py`` 以 ``from .pooling import (...)``
重导出，外部 ``pipeline._run_consumer`` 等引用零破坏。常量 STAGE_TASK_TIMEOUT /
STAGE_WATCHDOG / STAGE_WATCHDOG_INTERVAL / CONSUME_SLOW_WARN_SECONDS 一并迁入。
函数体逐字节一致，无循环依赖（仅依赖 policy.POLICY 与标准库）。
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import threading
import time
from collections import deque
from collections.abc import Callable, Sequence

from .policy import POLICY

LOGGER = logging.getLogger(__name__)

STAGE_TASK_TIMEOUT = POLICY.task_timeout  # 兼容别名：单任务上限（秒）
STAGE_WATCHDOG = STAGE_TASK_TIMEOUT * 3  # 看门狗窗口：整阶段多久没有任何进展就告警
STAGE_WATCHDOG_INTERVAL = 30.0  # 看门狗巡检间隔（秒）
# 单次 consume（父进程落库）耗时超过该秒数即告警：把「卡在 commit / 锁等待」
# 从不可见变成一行可定位的日志，而不必等看门狗的 30 分钟窗口。
CONSUME_SLOW_WARN_SECONDS = float(os.environ.get("TOPO_CONSUME_SLOW_WARN", "5"))


def _make_pool(
    worker_count: int,
    initializer: Callable[..., None],
    initargs: tuple,
) -> mp.pool.Pool:
    """创建计算进程池。

    这里用 ``multiprocessing.Pool`` 而不是 ``ProcessPoolExecutor``：前者提供公开的
    ``terminate()`` 可以真正杀掉挂死的 worker，后者只能 ``shutdown(wait=True)`` 干等
    （挂死场景下等于二次卡死），也没有干净的公开 API 回收僵死进程。
    启动方式沿用默认上下文（Windows=spawn / Linux=fork），与改动前一致。
    """
    return mp.Pool(processes=worker_count, initializer=initializer, initargs=initargs)


def _terminate_pool(pool: mp.pool.Pool) -> None:
    """强杀进程池，绝不阻塞等待可能已挂死的 worker。

    之所以敢无条件 terminate：worker 全是纯计算，所有落库/落盘副作用都在父进程里完成，
    父进程已经取到的结果不会因为杀 worker 而丢失。
    """
    try:
        pool.terminate()
    except Exception as exc:  # pragma: no cover - 仅兜底，正常路径不触发
        LOGGER.warning("终止进程池失败：%s", exc)
    try:
        pool.join()
    except Exception as exc:  # pragma: no cover - 仅兜底，正常路径不触发
        LOGGER.warning("回收进程池失败：%s", exc)


def _run_pool_stage(  # noqa: PLR0913,PLR0915
    *,
    stage_label: str,
    worker_count: int,
    initializer: Callable[..., None],
    initargs: tuple,
    task_func: Callable[..., object],
    items: Sequence[object],
    labels: Sequence[str],
    consume: Callable[[int, object], None],
    on_failure: Callable[[int, str, str, bool], None],
    task_timeout: float = STAGE_TASK_TIMEOUT,
) -> None:
    """带 per-task 超时 + 看门狗 + 进程池自愈的并行执行器（P0 核心）。

    语义与原先的 ``executor.map(..., chunksize=1)`` 保持一致：
    **严格按 items 顺序**逐个把结果交给 ``consume(position, result)``（position 为 1 起的序号），
    因此调用方原有的循环体、``counts`` 聚合、``db.commit()`` 节奏、
    ``_progress`` 调用点都能原样复用。

    与原实现的唯一差别在异常路径：
      · ``get(timeout=...)`` 超时 → 记 ``on_failure(..., timed_out=True)``，
        立即 ``terminate()`` 杀掉卡死 worker 并重建池，**尚未消费的任务重新提交**，
        已消费的不重做；
      · 其它异常（含 worker 崩溃后残留的池级错误）→ 同样记 ``on_failure`` 并重建池继续。
    绝不静默吞掉：两条路径都会写 ``LOGGER.warning``。

    在途窗口固定为 ``worker_count * 2``：既能喂饱所有 worker（不出现批次栅栏导致的空转），
    又能把「已完成但还没被消费」的结果数量控制在常数级，内存占用严格优于原先
    ``executor.map`` 一次性提交全部任务的做法。
    """
    total = len(items)
    if total == 0:
        return
    watchdog = _StageWatchdog(stage_label, total).start()
    pool = _make_pool(worker_count, initializer, initargs)
    window = max(1, worker_count) * 2
    in_flight: deque[tuple[int, object]] = deque()
    next_submit = 0  # 下一个待提交任务的下标
    consumed = 0  # 已处理（成功或失败）的任务数，等于下一个待消费任务的下标
    try:
        while consumed < total:
            while next_submit < total and len(in_flight) < window:
                in_flight.append((next_submit, pool.apply_async(task_func, (items[next_submit],))))
                next_submit += 1
            item_index, handle = in_flight.popleft()
            position = item_index + 1
            label = labels[item_index]
            try:
                result = handle.get(timeout=task_timeout)
            except mp.TimeoutError:
                LOGGER.warning(
                    "%s阶段任务超时（>=%.0fs）：%s（第 %d/%d 项），记为错误，"
                    "强杀并重建进程池后继续剩余任务",
                    stage_label,
                    task_timeout,
                    label,
                    position,
                    total,
                )
                consumed = position
                on_failure(position, label, f"任务超时（>={task_timeout:.0f}s 未返回）", True)
                # 杀掉卡死 worker：池内其余在途任务随之作废，回退提交游标重新排队。
                _terminate_pool(pool)
                pool = _make_pool(worker_count, initializer, initargs)
                in_flight.clear()
                next_submit = consumed
                continue
            except Exception as exc:
                LOGGER.warning(
                    "%s阶段任务异常：%s（第 %d/%d 项）：%s，记为错误并重建进程池后继续",
                    stage_label,
                    label,
                    position,
                    total,
                    exc,
                )
                consumed = position
                on_failure(position, label, str(exc), False)
                _terminate_pool(pool)
                pool = _make_pool(worker_count, initializer, initargs)
                in_flight.clear()
                next_submit = consumed
                continue
            watchdog.beat(position, label)
            # ── consume 异常隔离（本次修复的核心）──────────────────────────
            # consume 在**父进程**里落库（SQLite 写 + 每 N 项一次 commit），是本函数中
            # 唯一会因跨进程锁争抢而长时间阻塞、甚至抛 sqlite3.OperationalError
            # (database is locked) 的环节。
            #
            # 原先它是裸调用：一旦抛异常，异常会穿过 finally 的 _terminate_pool 一路
            # 抛出整个阶段——已算好的在途结果全部作废，调用方（match_clouds）也来不及把
            # matching_signature 落成完成态，于是签名永久停留在 "in_progress:"，下次运行
            # 必须全量重算。实测现场正是这样：match_runs 停在 2160（恰为 commit 周期
            # 20 的整数倍，即卡点就在 commit 上），签名残留 in_progress: 前缀。
            #
            # 现在把它与上面的超时/异常分支对齐：记 on_failure 后继续下一项。
            # 刻意**不重建进程池**——池本身是健康的，故障出在父进程落库，
            # 重建只会白付一次全部 worker 的初始化成本（本项目约 1s/worker）。
            consume_started = time.monotonic()
            try:
                consume(position, result)
            except Exception as exc:
                LOGGER.exception(
                    "%s阶段结果落盘失败：%s（第 %d/%d 项）：%s，记为错误并继续后续任务",
                    stage_label,
                    label,
                    position,
                    total,
                    exc,
                )
                consumed = position
                on_failure(position, label, f"结果落盘失败：{exc}", False)
                continue
            consume_elapsed = time.monotonic() - consume_started
            if consume_elapsed >= CONSUME_SLOW_WARN_SECONDS:
                # 让「卡在 commit / 锁等待」立刻可见，不必等看门狗那 30 分钟的窗口。
                LOGGER.warning(
                    "%s阶段结果落盘耗时 %.1fs：%s（第 %d/%d 项），"
                    "通常意味着数据库正被其它进程占用；若反复出现请确认没有第二个实例在跑同一个库",
                    stage_label,
                    consume_elapsed,
                    label,
                    position,
                    total,
                )
            consumed = position
    finally:
        # 收尾一律用 terminate 而非 shutdown(wait=True)/close()+join()：
        # 结果早已全部取回父进程，此刻等待 worker「优雅退出」没有任何收益，
        # 却会在残留挂死 worker 时再卡死一次（正是本次要修的坑）。
        _terminate_pool(pool)
        watchdog.stop()


def _run_consumer(  # noqa: PLR0913
    *,
    stage_label: str,
    worker_count: int,
    initializer: Callable[..., None],
    initargs: tuple,
    worker_fn: Callable[..., object],
    items: Sequence[object],
    labels: Sequence[str],
    consume: Callable[[int, object], None],
    fail_handler: Callable[[int, str, str, bool], None],
    task_timeout: float = STAGE_TASK_TIMEOUT,
) -> None:
    """单/多 worker 共用隔离执行器：统一 worker 异常的 try/except 隔离。

    失败走 fail_handler 而非抛出——消除 match 阶段「单 worker（``map``）」与
    「多 worker（``_run_pool_stage``）」两条分支的隔离逻辑漂移：原先单 worker
    路径裸调 ``worker_fn``，任一目标异常会直接穿透整阶段；现与多 worker 一致，
    异常/超时记 ``fail_handler`` 后继续后续目标，断点续跑语义保持幂等
    （失败目标无 match_runs 行，重跑 match 自动重试）。

    多 worker 仍复用 :func:`_run_pool_stage` 的「per-task 超时 + 强杀子进程 + 重建池」；
    单 worker 在父进程串行 ``map``（结果顺序与 items 一致），且对 ``worker_fn`` 与
    ``consume`` 的调用都包在 try/except 里（与多 worker 路径的 consume 隔离对齐）。
    """
    if worker_count <= 1:
        initializer(*initargs)
        for index, item in enumerate(items, 1):
            label = labels[index - 1]
            try:
                result = worker_fn(item)
            except Exception as exc:
                LOGGER.warning(
                    "%s阶段单 worker 任务异常：%s（第 %d/%d 项）：%s，记为错误并继续后续目标",
                    stage_label,
                    label,
                    index,
                    len(items),
                    exc,
                )
                fail_handler(index, label, str(exc), False)
                continue
            # consume 隔离与多 worker 路径对齐：落盘异常不再穿透整阶段。
            try:
                consume(index, result)
            except Exception as exc:
                LOGGER.exception(
                    "%s阶段结果落盘失败：%s（第 %d/%d 项）：%s，记为错误并继续后续目标",
                    stage_label,
                    label,
                    index,
                    len(items),
                    exc,
                )
                fail_handler(index, label, f"结果落盘失败：{exc}", False)
                continue
    else:
        _run_pool_stage(
            stage_label=stage_label,
            worker_count=worker_count,
            initializer=initializer,
            initargs=initargs,
            task_func=worker_fn,
            items=items,
            labels=labels,
            consume=consume,
            on_failure=fail_handler,
            task_timeout=task_timeout,
        )


class _StageWatchdog:
    """阶段级看门狗：后台守护线程，只在长时间毫无进展时打印告警。

    刻意**不杀任何进程**——强杀由 :func:`_run_pool_stage` 的超时分支负责，
    看门狗只负责「让挂死可见」：万一子进程真挂了，日志里能直接看到卡在哪个阶段、
    第几个任务、最近一个成功的项是谁，而不是对着静止的进度条干等。
    这样也不会把「正常但极慢」的任务误判成故障。
    """

    def __init__(
        self,
        stage_label: str,
        total: int,
        window: float = STAGE_WATCHDOG,
        interval: float = STAGE_WATCHDOG_INTERVAL,
    ) -> None:
        self._stage_label = stage_label
        self._total = total
        self._window = window
        self._interval = max(1.0, interval)
        self._lock = threading.Lock()
        self._last_progress = time.monotonic()
        self._index = 0
        self._item = ""
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> _StageWatchdog:
        if self._window > 0:
            self._thread = threading.Thread(
                target=self._run, name=f"watchdog-{self._stage_label}", daemon=True
            )
            self._thread.start()
        return self

    def beat(self, index: int, item: str) -> None:
        """记录一次「确实拿到结果」的进展。"""
        with self._lock:
            self._last_progress = time.monotonic()
            self._index = index
            self._item = item

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None:
            thread.join(timeout=1.0)
        self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            with self._lock:
                idle = time.monotonic() - self._last_progress
                index = self._index
                item = self._item
            if idle >= self._window:
                LOGGER.warning(
                    "%s阶段已 %.0fs 无进度（已完成 %d/%d，最近成功项=%s），疑似子进程挂死；"
                    "单任务超时 %.0fs 到点后会强杀并继续",
                    self._stage_label,
                    idle,
                    index,
                    self._total,
                    item or "-",
                    STAGE_TASK_TIMEOUT,
                )
