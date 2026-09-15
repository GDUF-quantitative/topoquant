"""纯文本单行刷新进度 + 多进程日志合并（去重）。

修复需求（2026-08-24）：
1. 进度（审计表头 N/5303、各阶段 current/total）使用回车覆盖式单行刷新，
   不重复追加行；纯文本、无 ANSI，兼容 Windows 控制台。
2. 初始化 / 阶段提示仅在主进程打印一次；worker 进程禁止直接重复输出
   （worker 日志经队列上送主进程，由主进程统一去重打印）。
3. 合并 8 个 worker 的并行日志流：worker 经 ``QueueHandler`` 把记录发到主进程的
   ``WorkerLogHub``，主进程去重后打印，且与进度单行互不撕裂。
4. 纯文本、无 ANSI 乱码，Windows 控制台兼容：进度走 stdout，worker 日志走 stderr。

设计要点：
- ``ProgressWriter`` 直接写 ``sys.stdout``（绕过 rich）。真终端用 ``\r`` 回到行首 + 空格覆盖
  旧内容实现单行刷新；非 TTY（被 GUI 终端捕获 / 重定向）改用真实换行 + 节流，
  不输出任何 ANSI 转义序列，Windows 原生 conhost / WT 均安全。
- ``WorkerLogHub`` 在独立后台线程里从 ``multiprocessing.Queue`` 收记录，对相同文本在
  滑动时间窗内去重（避免 8 个 worker 同一错误刷 8 遍），打印到 ``sys.stderr``
  （与进度 stdout 分离，永不撕碎进度单行）。
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import sys
import threading
import time
import unicodedata
from collections import deque

# ── 主进程判定 ────────────────────────────────────────────
# worker（spawn 子进程）name != "MainProcess"。横幅 / 阶段提示应只在主进程打印。
def is_main_process() -> bool:
    return mp.current_process().name == "MainProcess"


def _display_width(s: str) -> int:
    """终端显示宽度：CJK 等宽(W)/全角(F)字符记 2 列，其余记 1 列。

    ``ProgressWriter`` 用 ``"\\r" + 空格覆盖`` 做单行刷新，空格数必须按*显示宽度*
    而非 ``len()``（字符数）计算——中文等双宽字符若按字符数计数会被少算一半，
    导致上一行未被完全覆盖，残留尾部碎片（如 ``...s``）。改用本函数后覆盖完整。
    """
    w = 0
    for ch in s:
        w += 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
    return w


# ── 单行刷新进度 ──────────────────────────────────────────
class ProgressWriter:
    """回车覆盖式单行进度，纯文本、无 ANSI。

    多次 ``update`` 回到同一行覆盖，不新增行；适合审计表头 N/5303、各阶段进度。
    与 worker 日志（stderr）分属不同流，互不干扰。

    用法::

        pw = ProgressWriter()
        pw.update("审计表头", 100, 5303, "valid=98")
        pw.update("审计表头", 200, 5303, "valid=196")   # 覆盖上一行
        pw.finish()                                      # 收尾换行
    """

    def __init__(self, stream=None, enabled: bool = True,
                 pct_step: int = 5, min_interval: float = 2.0):
        self._stream = stream or sys.stdout
        self._enabled = enabled
        self._lock = threading.Lock()
        self._last_len = 0
        # TTY 判定：真终端（cmd / WT / conhost）用 \r 覆盖式单行刷新；
        # 非 TTY（被 launcher GUI 终端捕获 / 重定向 / CI）下 \r 不被识别为覆盖，
        # 会被渲染成「同一行反复打印 + \r 清行残片（如 es=0）」，故切换为真实换行 + 节流。
        self._is_tty = hasattr(self._stream, "isatty") and self._stream.isatty()
        # 非 TTY 节流状态：每跨 pct_step 个百分点、或间隔 min_interval 秒、
        # 或任务收尾时才落一行，避免 5303 行刷屏与卡顿时每秒重复打印。
        self._pct_step = max(1, int(pct_step))
        self._min_interval = float(min_interval)
        self._last_emit_pct = -1
        self._last_emit_time = 0.0
        self._last_text = ""
        self._ever_emitted = False

    def update(self, label: str, current: int, total: int, summary: str = "") -> None:
        if not self._enabled:
            return
        pct = current * 100 // total if total else 0
        if total:
            text = f"{label} {current}/{total} ({pct}%) {summary}".rstrip()
        else:
            text = f"{label} {summary}".rstrip()
        with self._lock:
            if self._is_tty:
                # 真终端：回车覆盖式单行刷新（原有行为，保持不变）。
                if self._last_len:
                    self._stream.write("\r" + " " * self._last_len + "\r")
                self._stream.write("\r" + text)
                self._stream.flush()
                self._last_len = _display_width(text)
                return

            # 非 TTY：真实换行 + 节流；内容不变（如卡在 0%）则不重复落行。
            now = time.monotonic()
            is_final = bool(total) and current >= total
            pct_crossed = abs(pct - self._last_emit_pct) >= self._pct_step
            time_ok = (now - self._last_emit_time) >= self._min_interval
            changed = text != self._last_text
            if is_final or (not self._ever_emitted) or (pct_crossed and changed) or (time_ok and changed):
                self._stream.write(text + "\n")
                self._stream.flush()
                self._last_emit_pct = pct
                self._last_emit_time = now
                self._last_text = text
                self._ever_emitted = True
                self._last_len = 0

    def finish(self) -> None:
        with self._lock:
            if self._is_tty:
                if self._last_len:
                    self._stream.write("\r" + " " * self._last_len + "\r")
                    self._last_len = 0
                self._stream.write("\n")
            else:
                # 非 TTY：直接换行收尾，不写 \r 清行序列（避免残片）。
                self._stream.write("\n")
            self._stream.flush()


# ── 多进程日志合并（去重）────────────────────────────────
class WorkerLogHub:
    """主进程侧：从 worker 队列收日志、去重、打印到 stderr。

    worker 经 ``logging.handlers.QueueHandler`` 发送 ``LogRecord``；本 hub 在后台线程
    ``drain``，对相同 ``message`` 在 ``dedup_window`` 秒内去重（8 个 worker 同一错误只
    打印一次），并经 ``sys.stderr`` 输出。队列放入 ``None`` 哨兵可立即结束。
    """

    def __init__(self, queue, stream=None, dedup_window: float = 5.0, max_buffer: int = 200):
        self._queue = queue
        self._stream = stream or sys.stderr
        self._dedup_window = dedup_window
        self._max_buffer = max_buffer
        self._seen: deque[tuple[str, float]] = deque()
        self._seen_lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._drain, name="worker-log-hub", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop.set()
        t = self._thread
        if t is not None:
            t.join(timeout=timeout)

    def _dedup(self, message: str) -> bool:
        now = time.monotonic()
        with self._seen_lock:
            while self._seen and now - self._seen[0][1] > self._dedup_window:
                self._seen.popleft()
            for m, _ in self._seen:
                if m == message:
                    return False
            self._seen.append((message, now))
            while len(self._seen) > self._max_buffer:
                self._seen.popleft()
            return True

    @staticmethod
    def _render(rec) -> tuple[int, str]:
        # rec 可能是 LogRecord（QueueHandler 发送）或 (levelno, message) 元组（兼容）。
        if hasattr(rec, "getMessage"):
            return rec.levelno, rec.getMessage()
        if isinstance(rec, tuple) and len(rec) == 2:
            return int(rec[0]), str(rec[1])
        return logging.INFO, str(rec)

    def _drain(self) -> None:
        while not self._stop.is_set():
            try:
                rec = self._queue.get(timeout=0.2)
            except Exception:
                continue
            if rec is None:
                break
            levelno, message = self._render(rec)
            if not self._dedup(message):
                continue
            try:
                name = logging.getLevelName(levelno)
                self._stream.write(f"[worker:{name}] {message}\n")
                self._stream.flush()
            except Exception:
                # 打印失败绝不影响流水线主流程
                pass
