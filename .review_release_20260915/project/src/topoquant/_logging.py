"""日志启动统一入口（Phase 2b P1-2 收敛，零 handler 行为语义变更）。

将原先散落在 ``cli.py`` / ``pipeline.py`` / ``run.py`` 的三处日志启动实现收敛到此处单入口：

- ``configure_logging("rich", level, console)``   ← 原 ``cli._rich_basic_config``
- ``configure_logging("worker-silent")``           ← 原 ``pipeline._silence_worker_logging``
- ``configure_logging("instance", work_dir)``       ← 原 ``run._configure_instance_logging``
- ``configure_logging("ensure-root")``             ← 原 run.py 的 basicConfig 守卫

**不变量**：本模块只搬迁函数位置、收敛入口，所有 handler 的级别 / 传播 / 渲染语义与原实现
逐字节一致；不触 INV-1（数值）/ INV-2（签名键）/ INV-3（持久化）/ INV-4（CLI）。
"""

from __future__ import annotations

import logging
import logging.handlers
import multiprocessing as mp
import os
from pathlib import Path

from rich.console import Console
from rich.logging import RichHandler

# 与旧 run.py ``_INSTANCE_LOG_FORMAT`` 完全一致的共享格式串（时间/级别/进程号）。
LOG_FORMAT = "%(asctime)s %(levelname)-7s [pid=%(process)d] %(name)s: %(message)s"

# 默认控制台（人看）。各调用点通常显式传入 CONSOLE / CONSOLE_ERR，此处仅作缺省。
CONSOLE = Console()
CONSOLE_ERR = Console(stderr=True)


def configure_rich(level: int, console: Console | None = None) -> None:
    """原 ``cli._rich_basic_config``：把根 logger 接到 rich Console（默认 stdout，给人看）。

    历史实现用 ``logging.basicConfig`` 的默认 StreamHandler 写 **stderr**，与
    ``Progress(console=CONSOLE)`` 的 Live 重绘互不知情，会顶碎进度条。改用
    ``RichHandler(console=CONSOLE)`` 后日志由 rich 统一在 Live 区上方「上滚打印」。

    ``console`` 可传 ``CONSOLE_ERR``（stderr）：JSON 模式下诊断日志应走 stderr，否则会与
    ``print(json.dumps(result))`` 一同落在 stdout，破坏 ``--json`` 只输出机器可读 JSON 的契约。

    若已存在绑定到**不同** console 的 RichHandler（例如同一进程里前一次调用是 stdout、本次要
    stderr），先移除再按本次请求的 console 重建；``handlers`` 守卫保证同 console 不重复添加。
    用 addHandler + setLevel 而非 basicConfig：后者在「根 logger 已有 handler」时会拒绝新增
    handler，导致 RichHandler 挂不上、根级别停在设计值之上，吞掉 INFO 级流水线日志。
    """
    console = console or CONSOLE
    root = logging.getLogger()
    for handler in list(root.handlers):
        if isinstance(handler, RichHandler) and handler.console is not console:
            root.removeHandler(handler)
    if any(isinstance(handler, RichHandler) for handler in root.handlers):
        return
    root.addHandler(
        RichHandler(console=console, markup=False, show_path=False, rich_tracebacks=True)
    )
    root.setLevel(level)


def _redirect_fds_to_devnull() -> None:
    """重定向子进程真实 fd 1/2 到 devnull，封死 C++ 级 fprintf / 裸 print 噪声。

    仅改 fd 1/2 即可同时封死 GUDHI C++ 后端（CGAL）的 ``fprintf(stderr)`` 与 Python 级
    ``sys.stderr``；结果仍经 multiprocessing 管道回传（用独立 fd），不受影响。
    """
    try:
        _null_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(_null_fd, 1)  # stdout
        os.dup2(_null_fd, 2)  # stderr
        os.close(_null_fd)
    except OSError:
        pass


def silence_worker_logging(log_queue=None) -> None:
    """原 ``pipeline._silence_worker_logging``：spawn 子进程内静默一切输出，
    避免与父进程进度单行抢屏。

    两层防护（均仅在「子进程 root logger 无 handler」时执行；父进程已配好 RichHandler 时
    直接跳过，不会误伤）：

    1. root logger 挂 handler：
       - 提供 ``log_queue`` 时挂 ``QueueHandler``：子进程 Python 日志经队列上送主进程
         ``WorkerLogHub`` 合并去重后打印（见 :mod:`_progress_log`），不再整段丢弃——
         逐支股票的跳过/失败等诊断得以保留且只出现一次；
       - 未提供 ``log_queue`` 时挂 ``NullHandler``：维持旧行为，子进程日志整段丢弃。
    2. **重定向真实文件描述符 1/2 到 devnull**：``mp.Pool``（spawn）默认不重定向子进程流，
       GUDHI 的 C++ 后端（CGAL）及任意 ``print`` / ``warnings`` 会直接写继承来的终端 fd，
       绕过父进程进度单行导致屏幕撕裂。仅改 fd 1/2 即可封死（Python 级日志已走队列，C++ 级
       噪声走 devnull）；结果仍经 multiprocessing 管道回传（独立 fd），不受影响。

    **父进程保护**：原先仅靠「root logger 无 handler」间接推断自己是子进程，在两条真实路径上
    不成立（``_ensure_pivot_cache`` 在父进程里直接调 ``_init_match_worker_mmap``；``worker_count
    == 1`` 分支同样在父进程内直接调 initializer）。改为**显式判断进程身份**：只有真正的子进程
    才动 fd，父进程一律直接返回。
    """
    if mp.current_process().name == "MainProcess":
        # 父进程：绝不重定向 fd / 挂队列 handler，否则会吞掉自己的进度条与异常信息。
        return
    if logging.getLogger().handlers:
        return
    if log_queue is not None:
        # worker 日志上送主进程合并去重，而非丢弃。
        logging.getLogger().addHandler(logging.handlers.QueueHandler(log_queue))
    else:
        logging.getLogger().addHandler(logging.NullHandler())
    # 重定向子进程真实 fd，封死 GUDHI C++ 与子进程内直写终端的噪声
    _redirect_fds_to_devnull()


def configure_instance_logging(work_dir: Path) -> logging.Handler | None:
    """原 ``run._configure_instance_logging``：为当前 run.py 实例挂一个
    按 work_dir 隔离的日志文件处理器。

    单实例锁按 work_dir 加，故每个 work_dir 同时只有一个实例。把结构化日志（流水线 LOGGER、
    经 captureWarnings 收编的 warnings）写入 ``<work_dir>/.run_instance.log``，即可保证：
      - 隔离：不同实例落到各自文件，互不穿插；
      - 有序：单写入者顺序行写，天然有序。
    交互式进度/提示仍走 rich Console（由 ``configure_rich`` 配置），文件作为权威日志真源。
    """
    try:
        log_path = work_dir / ".run_instance.log"
        handler = logging.FileHandler(log_path, encoding="utf-8", delay=True)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt="%Y-%m-%dT%H:%M:%S"))
        logging.getLogger().addHandler(handler)
        # warnings 改走 logging，避免裸写 stderr 绕过 rich Console、撕碎进度条并在多实例间交错。
        logging.captureWarnings(True)
        return handler
    except OSError:
        return None


def ensure_root_handler() -> None:
    """原 run.py 守卫：主进程 root logger 无 handler 时补一个 WARNING 级 basicConfig。

    用于 ``python run.py`` 自带执行（无 pytest / 无 cli ``_rich_basic_config`` 配置）的场景，
    确保 ``_silence_worker_logging`` 的 MainProcess 守卫失效兜底前，主进程 fd 不会被重定向到
    devnull 而吞掉自检输出与异常。
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.WARNING)


def configure_logging(
    mode: str,
    *,
    level: int | None = None,
    console: Console | None = None,
    work_dir: Path | None = None,
    log_queue=None,
) -> logging.Handler | None:
    """日志启动单入口（P1-2 收敛目标）。

    :param mode: ``"rich"`` / ``"worker-silent"`` / ``"instance"`` / ``"ensure-root"``。
    :param level: ``"rich"`` 模式的根级别（缺省 ``logging.INFO``）。
    :param console: ``"rich"`` 模式的 rich Console（缺省 ``CONSOLE``）。
    :param work_dir: ``"instance"`` 模式的日志隔离目录。
    :param log_queue: ``"worker-silent"`` 模式可选；提供 ``multiprocessing.Queue`` 时 worker
        Python 日志经队列上送主进程 ``WorkerLogHub`` 合并去重（而非整段丢弃）。
    :returns: ``"instance"`` 模式返回所挂 FileHandler，其余返回 ``None``。
    """
    if mode == "rich":
        configure_rich(level if level is not None else logging.INFO, console)
        return None
    if mode == "worker-silent":
        silence_worker_logging(log_queue)
        return None
    if mode == "instance":
        if work_dir is None:
            raise ValueError("instance mode requires work_dir")
        return configure_instance_logging(work_dir)
    if mode == "ensure-root":
        ensure_root_handler()
        return None
    raise ValueError(f"unknown logging mode: {mode!r}")
