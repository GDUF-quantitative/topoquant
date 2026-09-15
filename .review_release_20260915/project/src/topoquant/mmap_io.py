"""图持久图 mmap 零拷贝 IO 原语（Phase 3 从 god-module 抽取）。

机械搬迁（byte-for-byte）：原 ``pipeline.py`` 中 diagram mmap 读/写/校验/清理相关
函数平移至此；``pipeline.py`` 以 ``from .mmap_io import (...)`` 重导出，外部
``pipeline._xxx`` 引用零破坏。函数体逐字节一致，无循环依赖（仅依赖 locks/policy/
config/storage 与标准库）。共享可变全局 ``_MMAP_HANDLES`` 仍驻留 pipeline.py，
本模块经惰性 ``from . import pipeline`` 访问，保证单一真相、无导入期循环。
"""

from __future__ import annotations

import ast
import gc
import json
import logging
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from contextlib import closing, contextmanager, suppress
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .data import DataError
from .locks import (
    _EXPORT_LOCK_BOOT_TOKEN,
    _export_lock_heartbeat,
    _lock_file_age,
    _lock_owner_is_valid,
    _pid_alive,
    _read_lock_owner_and_token,
)
from .policy import POLICY
from .signatures import (
    _compute_source_signature,
    _count_exportable_diagrams,
    _exportable_diagram_rows,
)
from .storage import connect

LOGGER = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int, Mapping[str, int]], None]


def _read_npy_header(path: Path) -> tuple[tuple, np.dtype]:
    """只读 npy 头部，返回 ``(shape, dtype)``，并显式关闭文件句柄。

    直接按 ``.npy`` 格式解析头部，避免依赖 numpy 私有 API（如 ``_read_array_header``，
    其在不同 numpy 版本间不稳定），保证跨版本可用。
    """
    with open(str(path), "rb") as handle:
        magic = handle.read(8)
        if magic[:6] != b"\x93NUMPY":
            raise ValueError("不是合法的 .npy 文件")
        major, minor = magic[6], magic[7]
        if major == 1:
            header_len = int.from_bytes(handle.read(2), "little")
        elif major == 2:
            header_len = int.from_bytes(handle.read(4), "little")
        else:
            raise ValueError(f"不支持的 npy 版本 {major}.{minor}")
        header_bytes = handle.read(header_len)
    header = ast.literal_eval(header_bytes.decode("latin-1"))
    shape = tuple(int(dim_size) for dim_size in header["shape"])
    dtype = np.dtype(header["descr"])
    return shape, dtype


def _close_mmap(arr: np.ndarray) -> None:
    """释放由 ``np.lib.format.open_memmap`` 产生的底层 mmap 句柄。

    ``open_memmap`` 返回的数组在 ``del`` 后由 GC 释放句柄，但释放时机不确定；
    在随后的 ``os.replace`` 原子发布之前显式关闭，可避免 Windows 上因句柄未及时释放而
    PermissionError。
    """
    base = getattr(arr, "base", None)
    if base is not None and hasattr(base, "close"):
        try:  # noqa: SIM105
            base.close()
        except Exception:  # pragma: no cover - 释放失败不影响正确性  # noqa: S110
            pass
    del arr


def _replace_atomic(src: str, dst: str) -> None:
    """跨平台原子替换；Windows 上若目标文件被未释放句柄占用，短暂重试以规避 ``PermissionError``。"""
    last_exc: OSError | None = None
    for _ in range(10):
        try:
            os.replace(src, dst)
            return
        except PermissionError as exc:
            last_exc = exc
            gc.collect()
            time.sleep(0.05)
    assert last_exc is not None  # noqa: S101
    raise last_exc


def _load_diagram_index(index_path: Path) -> dict[str, int]:
    """解析 ``diagram_index.json``，返回 ``{cloud_id: row}`` 字典。"""
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("索引不是字典")
    return {str(key): int(value) for key, value in raw.items()}


def _clear_mmap_handles() -> None:
    from . import pipeline

    """释放父进程持有的只读 mmap 句柄，避免覆盖文件时被 Windows 锁住。

    句柄缓存 ``_MMAP_HANDLES`` 在单次运行内不会跨 build/match 阶段冲突
    （build 先于 match 运行，彼时缓存为空），但同进程内重复运行流水线、
    或一次运行失败后再次导出时，旧句柄会阻塞 ``os.replace``。此处显式关闭并清空。
    """
    for path in list(pipeline._MMAP_HANDLES.keys()):
        handle = pipeline._MMAP_HANDLES.pop(path, None)
        if handle is None:
            continue
        base = getattr(handle, "base", None)
        if base is not None and hasattr(base, "close"):
            try:  # noqa: SIM105
                base.close()
            except Exception:  # pragma: no cover - 释放失败不影响正确性  # noqa: S110
                pass


def _export_dimensions(config: PipelineConfig) -> list[int]:
    """需要导出为 mmap 的同调维度 —— 只有 ``distance_dimensions`` 会被匹配阶段读取。

    以前无条件导出 ``0..max_homology_dimension`` 的全部维度，但匹配、剪枝、
    预测都只读 ``distance_dimensions`` 两维，其余维度的 .npy 纯属浪费磁盘与导出耗时（P1-1）。
    升序返回，保证文件生成与校验顺序确定。
    """
    return sorted(config.distance_dimensions)


def _remove_unused_dimension_files(work_dir: Path, export_dimensions: list[int]) -> None:
    """删除本次不再导出的 ``diagrams_h*.npy``，避免旧维度文件与当前配置混淆。

    典型场景：把 ``distance_dimensions`` 从 ``[0,1]`` 改成 ``[1,2]`` 后，
    旧的 ``diagrams_h0.npy`` 已无任何消费者，留着只会占空间并误导排查（P1-1）。
    删除失败（例如 Windows 上仍被别的进程映射）不影响正确性，仅记一条 debug。
    """
    keep = {f"diagrams_h{dimension}.npy" for dimension in export_dimensions}
    for path in work_dir.glob("diagrams_h*.npy"):
        if path.name in keep:
            continue
        try:
            path.unlink()
            LOGGER.info("已清理不再使用的持久图文件：%s", path.name)
        except OSError as exc:  # pragma: no cover - 清理失败不影响导出正确性
            LOGGER.debug("清理 %s 失败：%s", path.name, exc)


def _cleanup_stale_mmap_artifacts(work_dir: Path) -> None:
    """清理历史遗留的临时导出目录，避免长期运行后堆积重复文件。"""
    if not work_dir.is_dir():
        return
    for stale in work_dir.glob("_mmap_stage_*"):
        shutil.rmtree(stale, ignore_errors=True)


def _progress(
    callback: ProgressCallback | None,
    stage: str,
    current: int,
    total: int,
    stats: Mapping[str, int],
) -> None:
    if callback is not None:
        callback(stage, current, total, stats)


DIAGRAM_COUNTS_FILENAME = "diagram_counts.npy"
DIAGRAM_INDEX_FILENAME = "diagram_index.json"
DIAGRAM_SIGNATURE_FILENAME = "diagram_signature.txt"
# 导出锁等待期间的进度刷新间隔（秒）：同时作为等待轮询间隔，兼顾响应性与 CPU 占用。
EXPORT_LOCK_POLL_INTERVAL = POLICY.export_lock_poll  # 兼容别名（R5）
# mmap 导出锁的过期阈值：锁文件 mtime 超过该秒数即视为残留锁，可被抢占（P2）。
# 心跳（EXPORT_LOCK_HEARTBEAT_INTERVAL=60s）落地后，锁年龄 = 距最近心跳，与导出总时长无关。
# 取心跳间隔的 5 倍（300s）：容忍 4 次连续心跳丢失（GC 停顿/磁盘卡顿/进程被临时挂起），
# 同时把崩溃恢复等待从 30min 压到 5min。小于 5× 有误抢风险，大于 5× 则恢复过慢。
EXPORT_LOCK_STALE_SECONDS = POLICY.export_lock_stale  # 兼容别名（R5）
# 导出进度上报的最小间隔（秒）：导出主循环按批推进，节流上报避免刷屏拖慢导出本身。
EXPORT_PROGRESS_MIN_INTERVAL = 0.2


@contextmanager
def _export_lock(  # noqa: PLR0912,PLR0915
    work_dir: Path,
    timeout: float = 1800.0,
    *,
    progress: ProgressCallback | None = None,
    stage: str = "export_wait",
):
    """跨进程导出锁：保证同一时刻只有一个进程执行 mmap 导出。

    使用 ``work_dir/.mmap_export.lock`` 锁文件（``O_CREAT | O_EXCL`` 原子创建）。
    持锁进程把自身 ``pid:boot_token`` 写入文件（R7b：boot_token 为进程级随机串，
    用于根除 Windows PID 复用导致的误判「持有者仍活」）；若发现锁已被占用且持有者仍存活，
    且 token 一致，则轮询等待；
    获锁后会重新检查文件是否已由其它进程导出完毕（通常可直接复用，避免重复重导出）。
    若锁的持有者已不存在（崩溃遗留），则视为过期锁并抢占，保证不会死锁。

    P2：有效持有者的判定从「PID 存活」收紧为「PID 存活 **且** 锁文件 mtime 未超过
    ``EXPORT_LOCK_STALE_SECONDS``」。只看 PID 有个隐患——残留锁里的 PID 若被系统
    分配给了另一个无关进程，就会被误判成「持有者健在」，于是一声不吭地等满 30 分钟，
    外观和卡死完全一样。同时补上等待日志，让等待过程始终可见。

    另注：锁文件 mtime 现由持锁方后台心跳线程定期刷新（见
    ``EXPORT_LOCK_HEARTBEAT_INTERVAL``），因此「mtime 未超过阈值」等价于「持有者
    最近仍在健康导出」。长耗时导出（如数十万点云）不会因创建时刻过早而被误判过期抢占，
    避免了两个进程并发写同一 mmap 持久图导致的数据损坏。

    **等待进度条（本次新增）**：仅靠日志仍有盲区——``waiting_logged`` / ``half_logged``
    两个标志位决定了整段等待只打 2 条日志（进入等待时、等到一半时）。也就是说在最长
    30 分钟的等待里，屏幕上可能连续十几分钟一个字都不动，用户无法区分「在排队等锁」
    和「真的挂死了」。因此新增可选的 ``progress`` 回调：等待期间按
    ``EXPORT_LOCK_POLL_INTERVAL`` 逐秒上报「已等待秒数 / 上限秒数」，
    由 CLI 的 rich 进度条实时渲染（``stage`` 决定进度条标题，因此同一把锁在
    持久图导出与 pivot 缓存两处的等待可以分别显示）。

    刻意只在**真正进入等待**时才上报第一帧：无争抢的常规路径（一次
    ``O_CREAT|O_EXCL`` 即获锁）完全不建进度条，屏幕保持干净、零额外开销。
    """
    lock_path = work_dir / ".mmap_export.lock"
    work_dir.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    start = time.monotonic()
    deadline = start + timeout
    waiting_logged = False
    half_logged = False
    # 是否已经上报过等待进度：决定获锁后要不要补一帧满格把进度条收起来。
    wait_reported = False
    limit_seconds = max(1, int(timeout))
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{pid}:{_EXPORT_LOCK_BOOT_TOKEN}\n".encode())
            break
        except FileExistsError:
            owner, owner_token = _read_lock_owner_and_token(lock_path)
            age = _lock_file_age(lock_path)
            owner_alive = bool(owner) and _pid_alive(owner)
            # R7b：有效持有者 = pid 存活 且（旧格式无 token，或 token 与当前进程一致）；
            # token 不匹配（疑似 Windows PID 复用）即判死，不再误等满超时窗口。
            owner_valid = _lock_owner_is_valid(lock_path)
            now = time.monotonic()
            if owner_valid and age < EXPORT_LOCK_STALE_SECONDS and now < deadline:
                if not waiting_logged:
                    LOGGER.warning(
                        "等待 mmap 导出锁（owner_pid=%s，锁文件已存在 %.0fs），最多等待 %.0fs",
                        owner,
                        age,
                        timeout,
                    )
                    waiting_logged = True
                elif not half_logged and now - start >= timeout / 2:
                    LOGGER.warning(
                        "仍在等待 mmap 导出锁（owner_pid=%s），已等待 %.0fs/%.0fs",
                        owner,
                        now - start,
                        timeout,
                    )
                    half_logged = True
                # 等待过程实时可见：逐轮上报「已等待秒数 / 上限秒数」，
                # 由 CLI 的 rich 进度条渲染，替代原先长达数分钟的静默。
                waited = now - start
                _progress(
                    progress,
                    stage,
                    min(int(waited), limit_seconds),
                    limit_seconds,
                    {"owner_pid": owner, "waited_s": int(waited), "limit_s": limit_seconds},
                )
                wait_reported = True
                time.sleep(EXPORT_LOCK_POLL_INTERVAL)
                continue
            # 过期锁或持有者不可判定：抢占（下面按具体原因给出可定位的日志）
            if owner_alive and age >= EXPORT_LOCK_STALE_SECONDS:
                LOGGER.warning(
                    "mmap 导出锁已失联（owner_pid=%s 仍存活，但锁文件心跳停止 %.0fs 未刷新，"
                    "判定为进程失联并抢占）",
                    owner,
                    age,
                )
            elif owner_alive and owner_token is not None and owner_token != _EXPORT_LOCK_BOOT_TOKEN:
                LOGGER.warning(
                    "mmap 导出锁 owner_pid=%s 仍存活但 boot_token 不匹配（疑似 PID 复用），"
                    "判定为失效并抢占残留锁",
                    owner,
                )
            elif owner_alive:
                LOGGER.error(
                    "等待 mmap 导出锁超时（%.0fs，owner_pid=%s 仍存活），强制抢占", timeout, owner
                )
            elif owner:
                LOGGER.warning("mmap 导出锁持有者进程 %s 已退出，抢占残留锁", owner)
            else:
                LOGGER.warning("mmap 导出锁内容不可解析，抢占残留锁")
            with suppress(OSError):
                os.unlink(str(lock_path))
            continue
    if wait_reported:
        # 曾经排队等待过：补一帧满格，让 rich 判定该任务完成并停掉 spinner，
        # 否则等待进度条会带着转圈动画一直残留在屏幕上（见 cli._run_with_progress）。
        _progress(
            progress,
            stage,
            limit_seconds,
            limit_seconds,
            {
                "owner_pid": 0,
                "waited_s": int(time.monotonic() - start),
                "limit_s": limit_seconds,
            },
        )
    # 持锁方心跳：每隔 EXPORT_LOCK_HEARTBEAT_INTERVAL 刷新锁文件 mtime，
    # 使锁年龄反映最近一次心跳而非创建时刻。长耗时但健康的导出不会被误判过期
    # 抢占（原实现仅靠创建时刻 mtime，导出超过阈值即被后到进程抢占，导致两进程
    # 并发写同一 mmap 损坏）。仅当持有者真正失联（不再刷新 mtime）达阈值后，
    # 等待方才会抢占。
    stop_heartbeat = threading.Event()
    heartbeat = threading.Thread(
        target=_export_lock_heartbeat,
        args=(lock_path, stop_heartbeat),
        name="export-lock-heartbeat",
        daemon=True,
    )
    heartbeat.start()

    try:
        yield
    finally:
        stop_heartbeat.set()
        try:  # noqa: SIM105
            heartbeat.join(timeout=2.0)
        except Exception:  # pragma: no cover - 仅兜底  # noqa: S110
            pass
        with suppress(OSError):
            os.close(fd)
        with suppress(OSError):
            os.unlink(str(lock_path))


def _export_diagrams_to_mmap(  # noqa: PLR0912,PLR0915
    config: PipelineConfig,
    progress: ProgressCallback | None = None,
) -> dict[str, int]:
    """把 SQLite 中的持久图导出为 memory-mapped .npy 文件（原子发布）。

    产出文件（位于 ``config.work_dir``）：

    - ``diagram_index.json``  ``{cloud_id: row_index}`` 映射，按 cloud_id 排序保证确定性
    - ``diagram_counts.npy``  ``(N, D) int32``，每个点云在各维度的有效持久对数量。
      **列下标即同调维度**（``D = max_homology_dimension + 1``），匹配阶段以
      ``counts[row, dim]`` 直接按维度取值；未导出的维度列保留为 0。
    - ``diagrams_h{d}.npy``   ``(N, max_pairs_d, 2) float64``，不足部分以 NaN 填充。
      **仅为 ``config.distance_dimensions`` 中的维度生成**：其余维度从不被读取（P1-1）。
    - ``diagram_signature.txt``  源持久图集合的廉价签名，供后续 ``_ensure`` 判断是否需要重导出

    所有文件先写入唯一的临时目录，确认完整后再 ``os.replace`` 到最终位置。
    这样即便导出过程被中断、或与其它进程并发重导出，最终文件永远不会是 0 字节或半成品，
    避免后续 ``np.load(..., mmap_mode='r')`` 在 Windows 上抛出 ``[Errno 22] Invalid argument``。

    **导出进度（本次新增）**：可选的 ``progress`` 回调让导出过程实时可见。
    导出耗时几乎全部集中在第 5 步「按 cloud_id 批量流式读取持久图 BLOB 并写入 memmap」
    ——数万个点云、数十 MB 的 BLOB 读写，原先整段没有任何输出，只在最后打一条
    「导出完成」，中途与卡死无法区分。这里以**点云数**为进度基数（``current/total``
    语义直白），在每批写完后按 ``EXPORT_PROGRESS_MIN_INTERVAL`` 节流上报。
    第 2~3 步只聚合 ``pair_count`` 整数列、不读 BLOB，耗时可忽略，补日志即可。

    Args:
        config: 流水线配置。
        progress: 可选进度回调；``None``（默认）时完全不产生额外开销，
            与改动前行为一致，因此既有调用方与测试无需改动。

    Returns:
        ``{cloud_id: row_index}`` 映射。
    """
    # 释放父进程可能仍持有的旧 mmap 句柄（同进程内重复导出/重跑场景），
    # 否则 Windows 上 os.replace 会因文件被占用而 PermissionError。
    _clear_mmap_handles()
    # dimensions：counts 表的列集合，列下标 == 同调维度，必须覆盖 0..max 以维持按维度直接索引。
    # export_dimensions：真正落盘为 .npy 的维度，只有匹配阶段会读的那两维。
    dimensions = list(range(config.max_homology_dimension + 1))
    export_dimensions = _export_dimensions(config)
    config.work_dir.mkdir(parents=True, exist_ok=True)

    with closing(connect(config.database_path)) as db:
        # 1) 可导出点云列表（按 cloud_id 排序），即为 mmap 行序。
        rows = _exportable_diagram_rows(config)
        sorted_ids = [str(row["cloud_id"]) for row in rows]
        id_to_row = {cloud_id: row for row, cloud_id in enumerate(sorted_ids)}
        total = len(sorted_ids)

        # ── 导出进度上报器 ────────────────────────────────────────────
        # 以「点云数」为进度基数，语义直白（written/clouds）。按最小间隔节流：
        # 导出内层循环每个点云都会走一次，无节流的高频回调会让 rich 重绘成为
        # 新的瓶颈，反而拖慢导出本身。progress 为 None 时第一行即返回，零开销。
        last_report = 0.0

        def report_export(done: int, *, force: bool = False) -> None:
            """上报「已导出点云数 / 总点云数」。"""
            nonlocal last_report
            if progress is None:
                return
            now = time.monotonic()
            if not force and now - last_report < EXPORT_PROGRESS_MIN_INTERVAL:
                return
            last_report = now
            written = min(done, total)
            _progress(
                progress,
                "export",
                written,
                max(1, total),
                {"clouds": total, "written": written},
            )

        report_export(0, force=True)
        LOGGER.info("mmap 导出：开始扫描 %d 个点云的持久对计数……", total)

        # 2) 每维最大持久对数量：仅聚合 pair_count 整数列，不读取持久图 BLOB（P1-4）。
        #    非导出维度无 .npy，列宽为 0；counts 表中仍按 DB 实值填充（见下）。
        max_pairs = dict.fromkeys(dimensions, 0)
        if total:
            dim_ph = ",".join("?" for _ in export_dimensions)
            for dim, m in db.execute(
                f"SELECT dimension, MAX(pair_count) AS m FROM diagrams "
                f"WHERE dimension IN ({dim_ph}) GROUP BY dimension",
                (*export_dimensions,),
            ).fetchall():
                max_pairs[dim] = int(m)

        # 3) counts 表（列下标即维度）：同样只聚合 pair_count 整数列，避免把全部
        #    持久图 BLOB 读入内存（P1-4）。每维最大/最小列宽由此确定，仅导出维度落 .npy。
        dim_index = {dim: idx for idx, dim in enumerate(dimensions)}
        counts = np.zeros((total, len(dimensions)), dtype=np.int32)
        if total:
            dim_ph = ",".join("?" for _ in dimensions)
            # 分批查询：SQLite 默认最多 999 个 SQL 变量，cloud_id 数量（数千）远超此限，
            # 单个 IN (?) 列表会触发 OperationalError: too many SQL variables。
            # 按安全批大小切片，每批变量数 = 批大小 + len(dimensions)，严格 < 999。
            _chunk = max(1, 999 - len(dimensions) - 1)
            for batch_start in range(0, total, _chunk):
                batch = sorted_ids[batch_start : batch_start + _chunk]
                cid_ph = ",".join("?" for _ in batch)
                for cloud_id, dim, pair_count in db.execute(
                    f"SELECT cloud_id, dimension, pair_count FROM diagrams "
                    f"WHERE cloud_id IN ({cid_ph}) AND dimension IN ({dim_ph})",
                    (*batch, *dimensions),
                ).fetchall():
                    counts[id_to_row[str(cloud_id)], dim_index[dim]] = int(pair_count)
            LOGGER.info("mmap 导出：持久对计数扫描完成，开始写出持久图……")

    # 原子发布：先全部写到【唯一】临时目录，确认完整后再整批 replace 到最终位置。
    # 使用每次不同的临时目录可避免复用上一次导出可能残留的（带未释放句柄的）目录，
    # 从而在 Windows 上规避 os.replace 因源/目标文件被占用而报 PermissionError。
    work_dir = config.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    # 先清理历史遗留的临时导出目录，再创建本次的 stage，避免误删正在使用的目录。
    _cleanup_stale_mmap_artifacts(work_dir)
    stage = Path(tempfile.mkdtemp(prefix="_mmap_stage_", dir=str(work_dir)))
    try:
        counts_path = stage / DIAGRAM_COUNTS_FILENAME
        np.save(str(counts_path), counts)
        del counts

        # 各维度 memmap 块：先按维度创建（落盘，不占 RAM），随后按 cloud_id 流式填充。
        blocks: dict[int, np.ndarray] = {}
        for dimension in export_dimensions:
            width = max_pairs[dimension]
            path = stage / f"diagrams_h{dimension}.npy"
            if total == 0 or width == 0:
                np.save(str(path), np.empty((total, width, 2), dtype=np.float64))
                continue
            # open_memmap 直接落盘写入临时文件，避免在内存里再拼一份大数组。
            block = np.lib.format.open_memmap(
                str(path), mode="w+", dtype=np.float64, shape=(total, width, 2)
            )
            block[:] = np.nan
            blocks[dimension] = block

        # 5) 按 cloud_id 批量流式写出：每次仅把一个点云的持久图（仅导出维度）载入内存，
        #    写出后立即释放，内存峰值从「全部点云」降为「单个点云」（P1-4）。
        #    各维度 memmap 块本身在磁盘上，不额外占用 RAM。
        if total:
            with closing(connect(config.database_path)) as db:
                dim_ph = ",".join("?" for _ in export_dimensions)
                # 批量 IN 查询：按 sorted_ids 切片成批，每批一次性取回该批全部
                # (cloud_id, dimension, pair_count, pairs)，再在内存里按 cloud_id
                # （row 下标）归位，消除「每点云一次单查」的 DB 往返（E1 性能瓶颈）。
                # 批大小沿用既有 _chunk 公式，确保变量数 = 批大小 + len(dimensions) < 999。
                _chunk = max(1, 999 - len(dimensions) - 1)
                for batch_start in range(0, total, _chunk):
                    batch_ids = sorted_ids[batch_start : batch_start + _chunk]
                    cid_ph = ",".join("?" for _ in batch_ids)
                    for cloud_id, dimension, pair_count, blob in db.execute(
                        f"SELECT cloud_id, dimension, pair_count, pairs FROM diagrams "
                        f"WHERE cloud_id IN ({cid_ph}) AND dimension IN ({dim_ph})",
                        (*batch_ids, *export_dimensions),
                    ).fetchall():
                        row = id_to_row[str(cloud_id)]
                        block = blocks.get(dimension)
                        if block is None:
                            continue
                        try:
                            pairs = (
                                np.frombuffer(blob, dtype="<f8").copy().reshape(int(pair_count), 2)
                            )
                        except (ValueError, TypeError) as _exc:
                            # 持久图 blob 损坏（字节长度与 pair_count 不符）。显式 fail-fast：
                            # 跳过会令 block[row] 残留零值（空持久图），下游误判瓶颈距离=0、
                            # 产生静默错误匹配，比整批中止更危险。
                            raise DataError(
                                f"持久图 blob 损坏：cloud_id={cloud_id} dimension={dimension} "
                                f"pair_count={pair_count} blob_bytes={len(blob)} "
                                f"(期望 {int(pair_count) * 16})"
                            ) from _exc
                        block[row, : pairs.shape[0], :] = pairs
                    # 每批写完上报一次：batch_start + len(batch_ids) 即累计已写出的点云数。
                    report_export(batch_start + len(batch_ids))

        for block in blocks.values():
            block.flush()
            # 显式关闭底层 mmap 句柄，避免句柄延迟释放导致随后的 os.replace 在 Windows 上失败。
            _close_mmap(block)

        index_path = stage / DIAGRAM_INDEX_FILENAME
        index_path.write_text(json.dumps(id_to_row, ensure_ascii=False), encoding="utf-8")

        # 源数据签名：供后续 _ensure 智能判断“源是否变化”，未变化则跳过重导出。
        signature_path = stage / DIAGRAM_SIGNATURE_FILENAME
        signature_path.write_text(_compute_source_signature(config), encoding="utf-8")

        # 先发布持久图与计数，最后发布索引（索引即“就绪”标记）。
        for dimension in export_dimensions:
            _replace_atomic(
                str(stage / f"diagrams_h{dimension}.npy"),
                str(work_dir / f"diagrams_h{dimension}.npy"),
            )
        _replace_atomic(str(counts_path), str(work_dir / DIAGRAM_COUNTS_FILENAME))
        _replace_atomic(str(signature_path), str(work_dir / DIAGRAM_SIGNATURE_FILENAME))
        _replace_atomic(str(index_path), str(work_dir / DIAGRAM_INDEX_FILENAME))
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    _remove_unused_dimension_files(work_dir, export_dimensions)
    # 补一帧满格：让 CLI 判定该进度条完成并停掉 spinner（见 cli._run_with_progress）。
    report_export(total, force=True)
    LOGGER.info("mmap 持久图导出完成：%d 个点云，导出维度 %s", total, export_dimensions)
    return id_to_row


def _validate_diagram_mmap_files(
    config: PipelineConfig,
    index: dict[str, int],
    count_path: Path,
    dimension_paths: list[Path],
) -> None:
    """校验各 mmap 文件的完整性与 ``shape``/``dtype``，失败则抛 ``OSError/ValueError``。

    仅读取 npy 头部（不加载数据）即可判断 shape/dtype 是否正确，因此用显式 ``open`` +
    ``np.lib.format`` 解析头部，并在 ``with`` 中确保句柄一定释放。这比直接 ``np.load`` 更安全：
    对损坏/空文件 ``np.load`` 会在异常路径泄漏文件句柄，进而使随后的 ``os.replace`` 原子发布在
    Windows 上抛出 ``PermissionError``，导致自愈反而失败。
    """
    n = len(index)
    # counts 的列集合仍是 0..max（列下标即维度），但只有导出的维度才有对应 .npy 文件。
    column_count = config.max_homology_dimension + 1
    export_dimensions = _export_dimensions(config)

    count_shape, count_dtype = _read_npy_header(count_path)
    if count_shape != (n, column_count) or count_dtype != np.int32:
        raise ValueError(f"diagram_counts 形状/类型不符：{count_shape}/{count_dtype}")

    # 文件已通过头部校验且体积很小，这里整份读入以核对各维最大持久对数。
    try:
        counts = np.load(str(count_path))
    except (ValueError, OSError) as _exc:
        # 头部校验通过后，整份读入仍可能因文件在两次读取间被截断而失败。
        raise DataError(f"diagram_counts 读取失败（可能文件被截断）：{count_path}") from _exc
    for dimension, path in zip(export_dimensions, dimension_paths, strict=False):
        shape, dtype = _read_npy_header(path)
        if shape[0] != n or shape[2] != 2 or dtype != np.float64:
            raise ValueError(f"diagrams_h{dimension} 形状/类型不符：{shape}/{dtype}")
        if shape[1] < int(counts[:, dimension].max()):
            raise ValueError(f"diagrams_h{dimension} 宽度不足以容纳最大持久对数")


def _diagram_mmap_paths(config: PipelineConfig) -> tuple[Path, Path, list[Path]]:
    """返回 (索引文件, 计数文件, 各维度持久图文件) 的路径。"""
    return (
        config.work_dir / DIAGRAM_INDEX_FILENAME,
        config.work_dir / DIAGRAM_COUNTS_FILENAME,
        [
            config.work_dir / f"diagrams_h{dimension}.npy"
            for dimension in _export_dimensions(config)
        ],
    )


def _mmap_is_current(  # noqa: PLR0911
    config: PipelineConfig,
    index_path: Path,
    count_path: Path,
    dimension_paths: list[Path],
    expected: int,
) -> bool:
    """判断既有 mmap 文件是否仍有效：存在 + 结构完好 + 源数据签名一致。"""
    if not (
        index_path.is_file()
        and count_path.is_file()
        and all(path.is_file() for path in dimension_paths)
    ):
        return False
    try:
        index = _load_diagram_index(index_path)
    except (OSError, ValueError):
        return False
    try:
        _validate_diagram_mmap_files(config, index, count_path, dimension_paths)
    except (OSError, ValueError, EOFError):
        return False
    if len(index) != expected:
        return False
    signature_path = config.work_dir / DIAGRAM_SIGNATURE_FILENAME
    try:
        stored = signature_path.read_text(encoding="utf-8").strip()
    except OSError:
        return False
    if stored != _compute_source_signature(config):
        LOGGER.info("源持久图已变化，mmap 文件需要重新导出")
        return False
    return True


def _ensure_diagram_mmap(
    config: PipelineConfig,
    progress: ProgressCallback | None = None,
) -> dict[str, int]:
    """确保 mmap 持久图存在且与数据库一致，必要时自动重新导出。

    三层防冗余设计：

    1. **持锁后重新检查**——若另一并发进程已导出完毕，直接复用，不再重导出；
    2. **源数据签名比对**——源持久图集合未变化则跳过重导出（即使索引略旧）；
    3. **npy 头部强校验**——文件被截断/写坏时仍能及时发现并重导出，
       而不会在 worker 的 ``np.load(mmap_mode='r')`` 处抛出 ``[Errno 22]``。

    ``progress`` 会同时透传给两个可能长时间不返回的环节，使它们实时可见：
    等待跨进程导出锁（``export_wait``）与实际导出（``export``）。
    """
    index_path, count_path, dimension_paths = _diagram_mmap_paths(config)
    _cleanup_stale_mmap_artifacts(config.work_dir)
    _clear_mmap_handles()

    expected = _count_exportable_diagrams(config)
    if expected == 0:
        raise DataError("数据库中没有可用的持久图，请先运行 build 阶段")

    with _export_lock(config.work_dir, progress=progress, stage="export_wait"):
        if _mmap_is_current(config, index_path, count_path, dimension_paths, expected):
            LOGGER.info("mmap 持久图仍然有效，复用既有文件，跳过重导出")
            return _load_diagram_index(index_path)
        LOGGER.warning("mmap 持久图需要（重新）导出……")
        return _export_diagrams_to_mmap(config, progress)
