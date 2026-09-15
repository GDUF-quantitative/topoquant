from __future__ import annotations

import ast
import gc
import hashlib
import json
import logging
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import closing, contextmanager
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .data import (
    DataError,
    iter_cloud_windows,
    list_stock_files,
    load_future_directions,
    source_signature,
    standardized_points,
)
from .domain import CloudRecord, Diagram
from .storage import (
    check_or_set_identity,
    connect,
    get_metadata,
    load_cloud_records,
    load_diagrams,
    save_cloud,
    save_cloud_error,
    save_forecast_error,
    save_forecasts,
    save_matches,
    set_metadata,
)
from .topology import compute_persistence, finite_bottleneck_distance


LOGGER = logging.getLogger(__name__)

DIAGRAM_INDEX_FILENAME = "diagram_index.json"
DIAGRAM_COUNTS_FILENAME = "diagram_counts.npy"
DIAGRAM_SIGNATURE_FILENAME = "diagram_signature.txt"
MATCH_SUMMARY_SIGNATURE_FILENAME = "match_summary_signature.txt"
PIVOT_ROWS_FILENAME = "match_pivot_rows.npy"
PIVOT_SIGNATURE_FILENAME = "match_pivot_signature.txt"

_BUILD_CONFIG: PipelineConfig | None = None
_BUILD_COMPLETED: set[str] = set()

# ── mmap 零拷贝匹配：worker 进程内的轻量全局状态 ──────────────────
# 这里只存文件路径与整数索引，持久图数据始终留在 OS page cache 中，
# 由所有 worker 通过只读内存映射共享，内存占用不随并发数增长。
_MMAP_PATHS: dict[int, str] = {}
_MMAP_COUNT_PATH: str = ""
_MMAP_TARGET_IDX: dict[str, int] = {}
_MMAP_CANDIDATES: tuple[tuple[str, int], ...] = ()
_MMAP_CANDIDATE_ROWS: np.ndarray = np.array([], dtype=np.int64)
_MMAP_DIMS: tuple[int, int] = (0, 1)
_MMAP_THRESHOLD: float = 0.1
_MMAP_TOP_K: int = 5
_MMAP_HANDLES: dict[str, np.ndarray] = {}
# 各维有限持续度的降序谱，用于瓶颈距离下界剪枝。
_SPEC_RANKS: int = 8
_MMAP_SPEC: dict[int, np.ndarray] = {}
# 各维度本质类（death=+inf）的个数及按 birth 降序排列的 birth。
_MMAP_ESS_CNT: dict[int, np.ndarray] = {}
_MMAP_ESS_B: dict[int, np.ndarray] = {}
_MMAP_PIVOT_ROWS: np.ndarray = np.array([], dtype=np.int64)
_MMAP_PIVOT_DIST: dict[int, np.ndarray] = {}

ProgressCallback = Callable[[str, int, int, Mapping[str, int]], None]


def _progress(
    callback: ProgressCallback | None,
    stage: str,
    current: int,
    total: int,
    stats: Mapping[str, int],
) -> None:
    if callback is not None:
        callback(stage, current, total, stats)


def _init_build_worker(config: PipelineConfig, completed_cloud_ids: set[str]) -> None:
    global _BUILD_CONFIG, _BUILD_COMPLETED
    _BUILD_CONFIG = config
    _BUILD_COMPLETED = completed_cloud_ids


def _build_stock(source_path_text: str) -> dict[str, object]:
    if _BUILD_CONFIG is None:
        raise RuntimeError("持续同调工作进程尚未初始化")
    source_path = Path(source_path_text)
    counts = {"complete": 0, "skipped": 0, "error": 0, "stock_error": 0}
    completed: list[tuple[CloudRecord, Diagram]] = []
    errors: list[tuple[str, object, str, Path, str]] = []
    try:
        for window in iter_cloud_windows(_BUILD_CONFIG, [source_path]):
            if window.cloud_id in _BUILD_COMPLETED:
                counts["skipped"] += 1
                continue
            try:
                points = standardized_points(window, _BUILD_CONFIG.features)
                diagram = compute_persistence(
                    points,
                    _BUILD_CONFIG.max_edge_length,
                    _BUILD_CONFIG.max_homology_dimension,
                )
                completed.append((
                    CloudRecord(
                        window.cloud_id,
                        window.cloud_date,
                        window.stock_code,
                        window.source_path,
                        len(points),
                    ),
                    diagram,
                ))
                counts["complete"] += 1
            except Exception as exc:
                errors.append((
                    window.cloud_id,
                    window.cloud_date,
                    window.stock_code,
                    window.source_path,
                    str(exc),
                ))
                counts["error"] += 1
    except Exception as exc:
        counts["stock_error"] += 1
        return {"counts": counts, "completed": completed, "errors": errors, "fatal": str(exc)}
    return {"counts": counts, "completed": completed, "errors": errors, "fatal": None}


def build_topology(
    config: PipelineConfig,
    progress: ProgressCallback | None = None,
    reset: bool = False,
    force_rebuild: bool = False,
) -> dict[str, int]:
    if reset:
        _reset_experiment(config)
    files = list_stock_files(config.source_dir)
    config.work_dir.mkdir(parents=True, exist_ok=True)
    with closing(connect(config.database_path)) as db:
        check_or_set_identity(
            db,
            config.topology_signature(),
            source_signature(files),
            config.serializable(),
            force=reset or force_rebuild,
        )
        expected_dimensions = config.max_homology_dimension + 1
        rows = db.execute(
            "SELECT cloud_id FROM diagrams GROUP BY cloud_id HAVING COUNT(*) = ?",
            (expected_dimensions,),
        ).fetchall()
        completed_cloud_ids = {str(row["cloud_id"]) for row in rows}
        worker_count = config.resolved_topology_workers
        counts = {"complete": 0, "skipped": 0, "error": 0, "stock_error": 0}
        _progress(progress, "topology", 0, len(files), {**counts, "workers": worker_count})

        _init_build_worker(config, completed_cloud_ids)
        if worker_count == 1:
            results = map(_build_stock, (str(path) for path in files))
            executor = None
        else:
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_build_worker,
                initargs=(config, completed_cloud_ids),
            )
            results = executor.map(_build_stock, (str(path) for path in files), chunksize=1)

        try:
            for stock_index, result in enumerate(results, 1):
                result_counts = result["counts"]
                for key in counts:
                    counts[key] += int(result_counts[key])
                for record, diagram in result["completed"]:
                    save_cloud(db, record, diagram)
                for cloud_id, cloud_date, stock_code, source_path, error in result["errors"]:
                    save_cloud_error(db, cloud_id, cloud_date, stock_code, source_path, error)
                    LOGGER.warning("点云 %s 处理失败：%s", cloud_id, error)
                if result["fatal"]:
                    LOGGER.warning("行情文件 %s 处理失败：%s", files[stock_index - 1].name, result["fatal"])
                if stock_index % 25 == 0:
                    db.commit()
                    LOGGER.info("持续同调进度：%d/%d 支股票，%s", stock_index, len(files), counts)
                _progress(
                    progress,
                    "topology",
                    stock_index,
                    len(files),
                    {**counts, "workers": worker_count},
                )
        finally:
            if executor is not None:
                executor.shutdown()
        db.commit()
        counts["workers"] = worker_count

    # 在 SQLite 连接关闭之后再导出，避免与写连接争抢文件锁。
    LOGGER.info("导出 mmap 持久图，供匹配阶段零拷贝共享……")
    counts["mmap_diagrams"] = len(_export_diagrams_to_mmap(config))
    return counts


def _diagram_mmap_paths(config: PipelineConfig) -> tuple[Path, Path, list[Path]]:
    """返回 (索引文件, 计数文件, 各维度持久图文件) 的路径。"""
    dimensions = range(config.max_homology_dimension + 1)
    return (
        config.work_dir / DIAGRAM_INDEX_FILENAME,
        config.work_dir / DIAGRAM_COUNTS_FILENAME,
        [config.work_dir / f"diagrams_h{dimension}.npy" for dimension in dimensions],
    )


def _match_summary_paths(
    work_dir: Path, dimensions: tuple[int, int]
) -> tuple[Path, dict[int, Path], dict[int, Path], dict[int, Path]]:
    return (
        work_dir / MATCH_SUMMARY_SIGNATURE_FILENAME,
        {dim: work_dir / f"match_spec_h{dim}.npy" for dim in dimensions},
        {dim: work_dir / f"match_essential_count_h{dim}.npy" for dim in dimensions},
        {dim: work_dir / f"match_essential_birth_h{dim}.npy" for dim in dimensions},
    )


def _pivot_paths(
    work_dir: Path, dimensions: tuple[int, int]
) -> tuple[Path, Path, dict[int, Path]]:
    return (
        work_dir / PIVOT_SIGNATURE_FILENAME,
        work_dir / PIVOT_ROWS_FILENAME,
        {dim: work_dir / f"match_pivot_distance_h{dim}.npy" for dim in dimensions},
    )


def _clear_mmap_handles() -> None:
    """释放父进程持有的只读 mmap 句柄，避免覆盖文件时被 Windows 锁住。

    句柄缓存 ``_MMAP_HANDLES`` 在单次运行内不会跨 build/match 阶段冲突
    （build 先于 match 运行，彼时缓存为空），但同进程内重复运行流水线、
    或一次运行失败后再次导出时，旧句柄会阻塞 ``os.replace``。此处显式关闭并清空。
    """
    for path in list(_MMAP_HANDLES.keys()):
        handle = _MMAP_HANDLES.pop(path, None)
        if handle is None:
            continue
        base = getattr(handle, "base", None)
        if base is not None and hasattr(base, "close"):
            try:
                base.close()
            except Exception:  # pragma: no cover - 释放失败不影响正确性
                pass



def _reset_experiment(config: PipelineConfig) -> None:
    """清空实验库与 mmap 中间产物，回到全新状态（``--reset`` 使用）。

    直接删除实验库文件（含 WAL/SHM）与所有 mmap 产物；下次 ``connect`` 会按
    当前 schema 重建空库。与 ``--force-rebuild`` 不同，这里是“物理删除从头来过”。
    """
    db_path = config.database_path
    for path in (db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    index_path, count_path, dimension_paths = _diagram_mmap_paths(config)
    for path in (index_path, count_path, *dimension_paths,
                 config.work_dir / DIAGRAM_SIGNATURE_FILENAME):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    summary_signature, spec_paths, ess_count_paths, ess_birth_paths = _match_summary_paths(
        config.work_dir, tuple(config.distance_dimensions)
    )
    pivot_signature, pivot_rows, pivot_distance_paths = _pivot_paths(
        config.work_dir, tuple(config.distance_dimensions)
    )
    for path in (
        summary_signature,
        *spec_paths.values(),
        *ess_count_paths.values(),
        *ess_birth_paths.values(),
        pivot_signature,
        pivot_rows,
        *pivot_distance_paths.values(),
    ):
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass
    LOGGER.warning("--reset：实验库与 mmap 中间产物已清空，将从头重算")


def _export_diagrams_to_mmap(config: PipelineConfig) -> dict[str, int]:
    """把 SQLite 中的持久图导出为 memory-mapped .npy 文件（原子发布）。

    产出文件（位于 ``config.work_dir``）：

    - ``diagram_index.json``  ``{cloud_id: row_index}`` 映射，按 cloud_id 排序保证确定性
    - ``diagram_counts.npy``  ``(N, D) int32``，每个点云在各维度的有效持久对数量
    - ``diagrams_h{d}.npy``   ``(N, max_pairs_d, 2) float64``，不足部分以 NaN 填充
    - ``diagram_signature.txt``  源持久图集合的廉价签名，供后续 ``_ensure`` 判断是否需要重导出

    所有文件先写入唯一的临时目录，确认完整后再 ``os.replace`` 到最终位置。
    这样即便导出过程被中断、或与其它进程并发重导出，最终文件永远不会是 0 字节或半成品，
    避免后续 ``np.load(..., mmap_mode='r')`` 在 Windows 上抛出 ``[Errno 22] Invalid argument``。

    Returns:
        ``{cloud_id: row_index}`` 映射。
    """
    # 释放父进程可能仍持有的旧 mmap 句柄（同进程内重复导出/重跑场景），
    # 否则 Windows 上 os.replace 会因文件被占用而 PermissionError。
    _clear_mmap_handles()
    dimensions = list(range(config.max_homology_dimension + 1))
    config.work_dir.mkdir(parents=True, exist_ok=True)

    with closing(connect(config.database_path)) as db:
        all_diagrams = load_diagrams(db, dimensions)

    sorted_ids = sorted(all_diagrams)
    id_to_row = {cloud_id: row for row, cloud_id in enumerate(sorted_ids)}
    total = len(sorted_ids)

    # 一次遍历同时算出每维最大持久对数量和每个点云的有效对数。
    max_pairs = dict.fromkeys(dimensions, 0)
    counts = np.zeros((total, len(dimensions)), dtype=np.int32)
    for row, cloud_id in enumerate(sorted_ids):
        diagram = all_diagrams[cloud_id]
        for column, dimension in enumerate(dimensions):
            pairs = diagram.get(dimension)
            length = 0 if pairs is None else int(pairs.shape[0])
            counts[row, column] = length
            if length > max_pairs[dimension]:
                max_pairs[dimension] = length

    # 原子发布：先全部写到【唯一】临时目录，确认完整后再整批 replace 到最终位置。
    # 使用每次不同的临时目录可避免复用上一次导出可能残留的（带未释放句柄的）目录，
    # 从而在 Windows 上规避 os.replace 因源/目标文件被占用而报 PermissionError。
    work_dir = config.work_dir
    work_dir.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix="_mmap_stage_", dir=str(work_dir)))
    # 尽力清理历史遗留的临时目录，避免长期运行后堆积。
    for stale in work_dir.glob("_mmap_stage_*"):
        if stale != stage:
            shutil.rmtree(stale, ignore_errors=True)
    try:
        counts_path = stage / DIAGRAM_COUNTS_FILENAME
        np.save(str(counts_path), counts)
        del counts

        for dimension in dimensions:
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
            for row, cloud_id in enumerate(sorted_ids):
                pairs = all_diagrams[cloud_id].get(dimension)
                if pairs is not None and pairs.size:
                    block[row, : pairs.shape[0], :] = pairs
            block.flush()
            # 显式关闭底层 mmap 句柄，避免句柄延迟释放导致随后的 os.replace 在 Windows 上失败。
            _close_mmap(block)

        index_path = stage / DIAGRAM_INDEX_FILENAME
        index_path.write_text(json.dumps(id_to_row, ensure_ascii=False), encoding="utf-8")

        # 源数据签名：供后续 _ensure 智能判断“源是否变化”，未变化则跳过重导出。
        signature_path = stage / DIAGRAM_SIGNATURE_FILENAME
        signature_path.write_text(_compute_source_signature(config), encoding="utf-8")

        # 先发布持久图与计数，最后发布索引（索引即“就绪”标记）。
        for dimension in dimensions:
            _replace_atomic(
                str(stage / f"diagrams_h{dimension}.npy"),
                str(work_dir / f"diagrams_h{dimension}.npy"),
            )
        _replace_atomic(str(counts_path), str(work_dir / DIAGRAM_COUNTS_FILENAME))
        _replace_atomic(str(signature_path), str(work_dir / DIAGRAM_SIGNATURE_FILENAME))
        _replace_atomic(str(index_path), str(work_dir / DIAGRAM_INDEX_FILENAME))
    finally:
        shutil.rmtree(stage, ignore_errors=True)

    LOGGER.info("mmap 持久图导出完成：%d 个点云，维度 %s", total, dimensions)
    return id_to_row


def _count_exportable_diagrams(config: PipelineConfig) -> int:
    """统计数据库中可导出的持久图数量，用于判断 mmap 文件是否过期。"""
    dimensions = tuple(range(config.max_homology_dimension + 1))
    placeholders = ",".join("?" for _ in dimensions)
    with closing(connect(config.database_path)) as db:
        row = db.execute(
            f"""
            SELECT COUNT(*) AS total FROM (
                SELECT d.cloud_id
                FROM diagrams d JOIN clouds c ON c.cloud_id = d.cloud_id
                WHERE c.status = 'complete' AND d.dimension IN ({placeholders})
                GROUP BY d.cloud_id
                HAVING COUNT(DISTINCT d.dimension) = ?
            )
            """,
            (*dimensions, len(dimensions)),
        ).fetchone()
    return int(row["total"])


def _compute_source_signature(config: PipelineConfig) -> str:
    """计算源持久图集合的廉价签名，用于判断既有 mmap 文件是否仍然有效。

    仅做 SQL 聚合，不把持久图 BLOB 载入内存，开销为毫秒级。签名由「可导出点云数 +
    持久对总数 + diagram rowid 版本和 + 所有 cloud_id 的排序哈希」组成。正常 build
    会先删除再插入 diagram 行，因此即使持久对数量不变，内容重算也会改变 rowid 版本和，
    从而触发重导出；否则可安全复用既有文件，避免冗余重导出。
    """
    dimensions = tuple(range(config.max_homology_dimension + 1))
    placeholders = ",".join("?" for _ in dimensions)
    with closing(connect(config.database_path)) as db:
        rows = db.execute(
            f"""
            SELECT d.cloud_id, SUM(d.pair_count) AS pc, SUM(d.rowid) AS row_version
            FROM diagrams d JOIN clouds c ON c.cloud_id = d.cloud_id
            WHERE c.status = 'complete' AND d.dimension IN ({placeholders})
            GROUP BY d.cloud_id
            HAVING COUNT(DISTINCT d.dimension) = ?
            ORDER BY d.cloud_id
            """,
            (*dimensions, len(dimensions)),
        ).fetchall()
    cloud_count = len(rows)
    total_pairs = sum(int(row["pc"]) for row in rows)
    row_version = sum(int(row["row_version"]) for row in rows)
    id_blob = "\n".join(row["cloud_id"] for row in rows).encode("utf-8")
    id_hash = hashlib.sha256(id_blob).hexdigest()[:16]
    return f"{cloud_count}:{total_pairs}:{row_version}:{id_hash}"


def _cleanup_stale_mmap_artifacts(work_dir: Path) -> None:
    """清理历史遗留的临时导出目录，避免长期运行后堆积重复文件。"""
    if not work_dir.is_dir():
        return
    for pattern in ("_mmap_stage_*", "_match_stage_*", "_pivot_stage_*"):
        for stale in work_dir.glob(pattern):
            shutil.rmtree(stale, ignore_errors=True)


@contextmanager
def _export_lock(work_dir: Path, timeout: float = 1800.0):
    """跨进程导出锁：保证同一时刻只有一个进程执行 mmap 导出。

    使用 ``work_dir/.mmap_export.lock`` 锁文件（``O_CREAT | O_EXCL`` 原子创建）。
    持锁进程把自身 PID 写入文件；若发现锁已被占用且持有者仍存活，则轮询等待；
    获锁后会重新检查文件是否已由其它进程导出完毕（通常可直接复用，避免重复重导出）。
    若锁的持有者已不存在（崩溃遗留），则视为过期锁并抢占，保证不会死锁。
    """
    lock_path = work_dir / ".mmap_export.lock"
    work_dir.mkdir(parents=True, exist_ok=True)
    pid = os.getpid()
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{pid}\n".encode("utf-8"))
            break
        except FileExistsError:
            owner = _read_lock_owner(lock_path)
            if owner and _pid_alive(owner) and time.monotonic() < deadline:
                time.sleep(0.5)
                continue
            # 过期锁或持有者不可判定：抢占
            try:
                os.unlink(str(lock_path))
            except OSError:
                pass
            continue
    try:
        yield
    finally:
        try:
            os.close(fd)
        except OSError:
            pass
        try:
            os.unlink(str(lock_path))
        except OSError:
            pass


def _read_lock_owner(lock_path: Path) -> int:
    try:
        with open(str(lock_path), "r") as handle:
            return int(handle.read().strip() or 0)
    except (OSError, ValueError):
        return 0


def _pid_alive(pid: int) -> bool:
    """跨平台判断进程是否存活。"""
    if pid <= 0:
        return False
    if sys.platform == "win32":
        ctypes = __import__("ctypes")
        kernel32 = ctypes.windll.kernel32  # type: ignore[attr-defined]
        PROCESS_QUERY_INFORMATION = 0x0400
        handle = kernel32.OpenProcess(PROCESS_QUERY_INFORMATION, False, pid)
        if not handle:
            return False
        exit_code = ctypes.c_uint32()
        kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code))
        kernel32.CloseHandle(handle)
        return exit_code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except OSError:
        return False
    return True


def _mmap_is_current(
    config: PipelineConfig,
    index_path: Path,
    count_path: Path,
    dimension_paths: list[Path],
    expected: int,
) -> bool:
    """判断既有 mmap 文件是否仍有效：存在 + 结构完好 + 源数据签名一致。"""
    if not (index_path.is_file() and count_path.is_file()
            and all(p.is_file() for p in dimension_paths)):
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


def _ensure_diagram_mmap(config: PipelineConfig) -> dict[str, int]:
    """确保 mmap 持久图存在且与数据库一致，必要时自动重新导出。

    三层防冗余设计：

    1. **持锁后重新检查**——若另一并发进程已导出完毕，直接复用，不再重导出；
    2. **源数据签名比对**——源持久图集合未变化则跳过重导出（即使索引略旧）；
    3. **npy 头部强校验**——文件被截断/写坏时仍能及时发现并重导出，
       而不会在 worker 的 ``np.load(mmap_mode='r')`` 处抛出 ``[Errno 22]``。
    """
    index_path, count_path, dimension_paths = _diagram_mmap_paths(config)
    _cleanup_stale_mmap_artifacts(config.work_dir)
    _clear_mmap_handles()

    expected = _count_exportable_diagrams(config)
    if expected == 0:
        raise DataError("数据库中没有可用的持久图，请先运行 build 阶段")

    with _export_lock(config.work_dir):
        if _mmap_is_current(config, index_path, count_path, dimension_paths, expected):
            LOGGER.info("mmap 持久图仍然有效，复用既有文件，跳过重导出")
            return _load_diagram_index(index_path)
        LOGGER.warning("mmap 持久图需要（重新）导出……")
        return _export_diagrams_to_mmap(config)


def _load_diagram_index(index_path: Path) -> dict[str, int]:
    """解析 ``diagram_index.json``，返回 ``{cloud_id: row}`` 字典。"""
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("索引不是字典")
    return {str(key): int(value) for key, value in raw.items()}


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
    dimensions = list(range(config.max_homology_dimension + 1))

    count_shape, count_dtype = _read_npy_header(count_path)
    if count_shape != (n, len(dimensions)) or count_dtype != np.int32:
        raise ValueError(
            f"diagram_counts 形状/类型不符：{count_shape}/{count_dtype}"
        )

    # 文件已通过头部校验且体积很小，这里整份读入以核对各维最大持久对数。
    counts = np.load(str(count_path))
    for dimension, path in zip(dimensions, dimension_paths):
        shape, dtype = _read_npy_header(path)
        if shape[0] != n or shape[2] != 2 or dtype != np.float64:
            raise ValueError(
                f"diagrams_h{dimension} 形状/类型不符：{shape}/{dtype}"
            )
        if shape[1] < int(counts[:, dimension].max()):
            raise ValueError(
                f"diagrams_h{dimension} 宽度不足以容纳最大持久对数"
            )


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
    shape = tuple(int(x) for x in header["shape"])
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
        try:
            base.close()
        except Exception:  # pragma: no cover - 释放失败不影响正确性
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
    assert last_exc is not None
    raise last_exc


def _match_summary_signature(config: PipelineConfig, dimensions: tuple[int, int]) -> str:
    dims = ",".join(str(dim) for dim in dimensions)
    return f"{_compute_source_signature(config)}:summary-v2:ranks={_SPEC_RANKS}:dims={dims}"


def _match_summary_is_current(
    config: PipelineConfig,
    dimensions: tuple[int, int],
    total: int,
) -> bool:
    signature_path, spec_paths, count_paths, birth_paths = _match_summary_paths(
        config.work_dir, dimensions
    )
    try:
        if signature_path.read_text(encoding="utf-8").strip() != _match_summary_signature(
            config, dimensions
        ):
            return False
        for dim in dimensions:
            spec_shape, spec_dtype = _read_npy_header(spec_paths[dim])
            count_shape, count_dtype = _read_npy_header(count_paths[dim])
            birth_shape, birth_dtype = _read_npy_header(birth_paths[dim])
            if spec_shape != (total, _SPEC_RANKS) or spec_dtype != np.float64:
                return False
            if count_shape != (total,) or count_dtype != np.int32:
                return False
            if (
                len(birth_shape) != 2
                or birth_shape[0] != total
                or birth_shape[1] < 1
                or birth_dtype != np.float64
            ):
                return False
    except (OSError, ValueError, EOFError):
        return False
    return True


def _ensure_match_summary(
    config: PipelineConfig,
    dimensions: tuple[int, int],
    total: int,
) -> tuple[dict[int, Path], dict[int, Path], dict[int, Path]]:
    """一次生成并持久化所有 worker 共用的剪枝摘要。"""
    signature_path, spec_paths, count_paths, birth_paths = _match_summary_paths(
        config.work_dir, dimensions
    )
    with _export_lock(config.work_dir):
        if _match_summary_is_current(config, dimensions, total):
            return spec_paths, count_paths, birth_paths

        counts = np.load(str(config.work_dir / DIAGRAM_COUNTS_FILENAME), mmap_mode="r")
        stage = Path(tempfile.mkdtemp(prefix="_match_stage_", dir=str(config.work_dir)))
        try:
            for dim in dimensions:
                block = np.load(
                    str(config.work_dir / f"diagrams_h{dim}.npy"), mmap_mode="r"
                )
                spec = np.full((total, _SPEC_RANKS), np.nan, dtype=np.float64)
                essential_count = np.zeros(total, dtype=np.int32)
                births: list[np.ndarray | None] = [None] * total
                width = 1
                for row in range(total):
                    count = int(counts[row, dim])
                    if count == 0:
                        continue
                    points = block[row, :count, :]
                    finite_birth = np.isfinite(points[:, 0])
                    finite_death = np.isfinite(points[:, 1])
                    finite_mask = finite_birth & finite_death
                    spec[row, :] = 0.0
                    if finite_mask.any():
                        finite = points[finite_mask]
                        ranked = np.sort(finite[:, 1] - finite[:, 0])[::-1]
                        take = min(_SPEC_RANKS, ranked.shape[0])
                        spec[row, :take] = ranked[:take]
                    essential_mask = finite_birth & ~finite_death
                    size = int(essential_mask.sum())
                    if size:
                        essential_count[row] = size
                        births[row] = np.sort(
                            np.asarray(points[essential_mask, 0], dtype=np.float64)
                        )[::-1]
                        width = max(width, size)
                essential_birth = np.zeros((total, width), dtype=np.float64)
                for row, values in enumerate(births):
                    if values is not None:
                        essential_birth[row, : values.shape[0]] = values
                np.save(str(stage / spec_paths[dim].name), spec)
                np.save(str(stage / count_paths[dim].name), essential_count)
                np.save(str(stage / birth_paths[dim].name), essential_birth)
                del block

            for dim in dimensions:
                for destination in (spec_paths[dim], count_paths[dim], birth_paths[dim]):
                    _replace_atomic(str(stage / destination.name), str(destination))
            staged_signature = stage / signature_path.name
            staged_signature.write_text(
                _match_summary_signature(config, dimensions), encoding="utf-8"
            )
            _replace_atomic(str(staged_signature), str(signature_path))
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    LOGGER.info("匹配剪枝摘要已持久化：%d 个点云", total)
    return spec_paths, count_paths, birth_paths


def _init_distance_worker_mmap(
    h0_path: str,
    h1_path: str,
    count_path: str,
    dimensions: tuple[int, int],
    spec_paths: dict[int, str],
    essential_count_paths: dict[int, str],
    essential_birth_paths: dict[int, str],
) -> None:
    global _MMAP_PATHS, _MMAP_COUNT_PATH, _MMAP_DIMS, _MMAP_HANDLES
    global _MMAP_SPEC, _MMAP_ESS_CNT, _MMAP_ESS_B
    _MMAP_PATHS = dict(zip(dimensions, (h0_path, h1_path)))
    _MMAP_COUNT_PATH = count_path
    _MMAP_DIMS = dimensions
    _MMAP_HANDLES = {}
    _MMAP_SPEC = {
        dim: _mmap_handle(spec_paths[dim]) for dim in dimensions
    }
    _MMAP_ESS_CNT = {
        dim: _mmap_handle(essential_count_paths[dim]) for dim in dimensions
    }
    _MMAP_ESS_B = {
        dim: _mmap_handle(essential_birth_paths[dim]) for dim in dimensions
    }


def _init_match_worker_mmap(
    h0_path: str,
    h1_path: str,
    count_path: str,
    target_index: dict[str, int],
    candidates: tuple[tuple[str, int], ...],
    dimensions: tuple[int, int],
    threshold: float,
    top_k: int,
    spec_paths: dict[int, str],
    essential_count_paths: dict[int, str],
    essential_birth_paths: dict[int, str],
    pivot_rows_path: str,
    pivot_distance_paths: dict[int, str],
) -> None:
    """匹配 worker 初始化：只接收文件路径与整数索引，不搬运任何持久图数据。"""
    global _MMAP_TARGET_IDX, _MMAP_CANDIDATES, _MMAP_CANDIDATE_ROWS
    global _MMAP_THRESHOLD, _MMAP_TOP_K, _MMAP_PIVOT_ROWS, _MMAP_PIVOT_DIST
    _init_distance_worker_mmap(
        h0_path,
        h1_path,
        count_path,
        dimensions,
        spec_paths,
        essential_count_paths,
        essential_birth_paths,
    )
    _MMAP_TARGET_IDX = target_index
    _MMAP_CANDIDATES = candidates
    _MMAP_CANDIDATE_ROWS = np.asarray(
        [row for _, row in candidates], dtype=np.int64
    )
    _MMAP_THRESHOLD = threshold
    _MMAP_TOP_K = top_k
    _MMAP_PIVOT_ROWS = _mmap_handle(pivot_rows_path)
    _MMAP_PIVOT_DIST = {
        dim: _mmap_handle(pivot_distance_paths[dim]) for dim in dimensions
    }


def _mmap_handle(path: str) -> np.ndarray:
    """按需打开并缓存只读内存映射。

    ``np.load(mmap_mode='r')`` 是 O(1) 的虚拟地址映射，不搬运数据；
    进程内缓存句柄可避免为每个目标点云重复打开文件。
    """
    handle = _MMAP_HANDLES.get(path)
    if handle is None:
        try:
            handle = np.load(path, mmap_mode="r")
        except (OSError, EOFError, ValueError) as exc:
            raise DataError(
                f"无法以只读内存映射打开持久图文件 {path}：{exc}。"
                f"该文件可能为空或被截断，请重新运行 build 阶段以触发自动重导出。"
            ) from exc
        _MMAP_HANDLES[path] = handle
    return handle


def _mmap_pairs(block: np.ndarray, row: int, count: int) -> np.ndarray:
    """从 mmap 中取出有限持久对，拷贝成 C 连续数组交给 Topp。"""
    points = block[row, :count, :]
    finite = np.isfinite(points[:, 0]) & np.isfinite(points[:, 1])
    return np.ascontiguousarray(points[finite], dtype=np.float64)


def _exact_bottleneck(
    finite_a: np.ndarray,
    finite_b: np.ndarray,
    essential_a: np.ndarray,
    essential_b: np.ndarray,
) -> float:
    """合成有限持久对和本质类的精确瓶颈距离。"""
    if essential_a.shape[0] != essential_b.shape[0]:
        return float("inf")
    distance = float(finite_bottleneck_distance(finite_a, finite_b))
    if essential_a.shape[0]:
        essential_distance = float(np.abs(essential_a - essential_b).max())
        distance = max(distance, essential_distance)
    return distance


def _exact_distances_between_rows(rows: tuple[int, int]) -> tuple[float, float]:
    left_row, right_row = rows
    counts = _mmap_handle(_MMAP_COUNT_PATH)
    distances: list[float] = []
    for dim in _MMAP_DIMS:
        block = _mmap_handle(_MMAP_PATHS[dim])
        left_essential_count = int(_MMAP_ESS_CNT[dim][left_row])
        right_essential_count = int(_MMAP_ESS_CNT[dim][right_row])
        distances.append(
            _exact_bottleneck(
                _mmap_pairs(block, left_row, int(counts[left_row, dim])),
                _mmap_pairs(block, right_row, int(counts[right_row, dim])),
                _MMAP_ESS_B[dim][left_row, :left_essential_count],
                _MMAP_ESS_B[dim][right_row, :right_essential_count],
            )
        )
    return float(distances[0]), float(distances[1])


def _pivot_signature(
    config: PipelineConfig,
    dimensions: tuple[int, int],
    candidate_ids: tuple[str, ...],
    pivot_count: int,
) -> str:
    candidate_hash = hashlib.sha256("\n".join(candidate_ids).encode("utf-8")).hexdigest()[:16]
    return (
        f"{_compute_source_signature(config)}:pivot-v3:topp-0.1.0-exact:"
        f"dims={dimensions}:candidates={candidate_hash}:count={pivot_count}"
    )


def _pivot_cache_is_current(
    config: PipelineConfig,
    dimensions: tuple[int, int],
    candidate_ids: tuple[str, ...],
    pivot_count: int,
) -> bool:
    signature_path, rows_path, distance_paths = _pivot_paths(config.work_dir, dimensions)
    try:
        if signature_path.read_text(encoding="utf-8").strip() != _pivot_signature(
            config, dimensions, candidate_ids, pivot_count
        ):
            return False
        rows_shape, rows_dtype = _read_npy_header(rows_path)
        if rows_shape != (pivot_count,) or rows_dtype != np.int64:
            return False
        for dim in dimensions:
            shape, dtype = _read_npy_header(distance_paths[dim])
            if shape != (len(candidate_ids), pivot_count) or dtype != np.float64:
                return False
    except (OSError, ValueError, EOFError):
        return False
    return True


def _ensure_pivot_cache(
    config: PipelineConfig,
    dimensions: tuple[int, int],
    candidate_ids: tuple[str, ...],
    candidate_rows: np.ndarray,
    h0_path: Path,
    h1_path: Path,
    count_path: Path,
    spec_paths: dict[int, Path],
    essential_count_paths: dict[int, Path],
    essential_birth_paths: dict[int, Path],
) -> tuple[Path, dict[int, Path]]:
    """构建可复用的历史图到 pivot 精确距离矩阵。"""
    signature_path, rows_path, distance_paths = _pivot_paths(config.work_dir, dimensions)
    pivot_count = min(config.matching_pivots, len(candidate_ids))
    with _export_lock(config.work_dir):
        if _pivot_cache_is_current(
            config, dimensions, candidate_ids, pivot_count
        ):
            LOGGER.info("pivot 距离缓存仍然有效：%d 个 pivot", pivot_count)
            return rows_path, distance_paths

        stage = Path(tempfile.mkdtemp(prefix="_pivot_stage_", dir=str(config.work_dir)))
        matrices = {
            dim: np.empty((len(candidate_ids), pivot_count), dtype=np.float64)
            for dim in dimensions
        }
        pivot_rows = np.empty(pivot_count, dtype=np.int64)
        executor: ProcessPoolExecutor | None = None
        try:
            if pivot_count:
                initargs = (
                    str(h0_path),
                    str(h1_path),
                    str(count_path),
                    dimensions,
                    {dim: str(spec_paths[dim]) for dim in dimensions},
                    {dim: str(essential_count_paths[dim]) for dim in dimensions},
                    {dim: str(essential_birth_paths[dim]) for dim in dimensions},
                )
                worker_count = config.resolved_matching_workers
                if worker_count == 1:
                    _init_distance_worker_mmap(*initargs)
                else:
                    executor = ProcessPoolExecutor(
                        max_workers=worker_count,
                        initializer=_init_distance_worker_mmap,
                        initargs=initargs,
                    )

                coverage = np.full(len(candidate_ids), np.inf, dtype=np.float64)
                selected = np.zeros(len(candidate_ids), dtype=bool)
                next_index = 0
                for column in range(pivot_count):
                    pivot_row = int(candidate_rows[next_index])
                    pivot_rows[column] = pivot_row
                    tasks = ((int(row), pivot_row) for row in candidate_rows)
                    results = (
                        map(_exact_distances_between_rows, tasks)
                        if executor is None
                        else executor.map(_exact_distances_between_rows, tasks, chunksize=32)
                    )
                    values = np.asarray(list(results), dtype=np.float64)
                    matrices[dimensions[0]][:, column] = values[:, 0]
                    matrices[dimensions[1]][:, column] = values[:, 1]
                    combined = np.maximum(values[:, 0], values[:, 1])
                    coverage = np.minimum(coverage, combined)
                    selected[next_index] = True
                    coverage[selected] = -np.inf
                    if column + 1 < pivot_count:
                        next_index = int(np.argmax(coverage))
                    LOGGER.info("pivot 缓存进度：%d/%d", column + 1, pivot_count)

            np.save(str(stage / rows_path.name), pivot_rows)
            for dim in dimensions:
                np.save(str(stage / distance_paths[dim].name), matrices[dim])
            _replace_atomic(str(stage / rows_path.name), str(rows_path))
            for dim in dimensions:
                _replace_atomic(
                    str(stage / distance_paths[dim].name), str(distance_paths[dim])
                )
            staged_signature = stage / signature_path.name
            staged_signature.write_text(
                _pivot_signature(config, dimensions, candidate_ids, pivot_count),
                encoding="utf-8",
            )
            _replace_atomic(str(staged_signature), str(signature_path))
        finally:
            if executor is not None:
                executor.shutdown()
            shutil.rmtree(stage, ignore_errors=True)
    return rows_path, distance_paths


def _match_one_mmap(target_id: str) -> tuple[str, int, int, list[tuple[str, float, float]]]:
    """单个目标点云对全部候选的瓶颈匹配（读共享 mmap）。"""
    dim0, dim1 = _MMAP_DIMS
    block0 = _mmap_handle(_MMAP_PATHS[dim0])
    block1 = _mmap_handle(_MMAP_PATHS[dim1])
    counts = _mmap_handle(_MMAP_COUNT_PATH)

    target_row = _MMAP_TARGET_IDX[target_id]
    target_count0 = int(counts[target_row, dim0])
    target_count1 = int(counts[target_row, dim1])
    target0 = _mmap_pairs(block0, target_row, target_count0)
    target1 = _mmap_pairs(block1, target_row, target_count1)

    candidate_rows = _MMAP_CANDIDATE_ROWS
    mask = candidate_rows != target_row
    target_essential: dict[int, np.ndarray] = {}
    lower_bounds = {
        dim: np.zeros(candidate_rows.shape[0], dtype=np.float64)
        for dim in _MMAP_DIMS
    }
    for dim in _MMAP_DIMS:
        target_essential_count = int(_MMAP_ESS_CNT[dim][target_row])
        target_essential[dim] = _MMAP_ESS_B[dim][
            target_row, :target_essential_count
        ]
        candidate_essential_count = _MMAP_ESS_CNT[dim][candidate_rows]
        mask &= candidate_essential_count == target_essential_count

        target_spec = _MMAP_SPEC[dim][target_row]
        candidate_spec = _MMAP_SPEC[dim][candidate_rows]
        valid_spec = np.isfinite(candidate_spec[:, 0]) & np.isfinite(target_spec[0])
        spectrum_lower_bound = np.abs(candidate_spec - target_spec).max(axis=1) / 2.0
        lower_bounds[dim] = np.maximum(
            lower_bounds[dim], np.where(valid_spec, spectrum_lower_bound, 0.0)
        )

        if target_essential_count:
            essential_lower_bound = np.abs(
                _MMAP_ESS_B[dim][candidate_rows, :target_essential_count]
                - target_essential[dim]
            ).max(axis=1)
            lower_bounds[dim] = np.maximum(
                lower_bounds[dim], essential_lower_bound
            )
        mask &= lower_bounds[dim] < _MMAP_THRESHOLD

    if mask.any() and _MMAP_PIVOT_ROWS.size:
        pivot_query = np.asarray(
            [
                _exact_distances_between_rows((target_row, int(pivot_row)))
                for pivot_row in _MMAP_PIVOT_ROWS
            ],
            dtype=np.float64,
        )
        for column, dim in enumerate(_MMAP_DIMS):
            history = _MMAP_PIVOT_DIST[dim]
            query = pivot_query[:, column]
            valid = np.isfinite(history) & np.isfinite(query)[None, :]
            difference = np.zeros_like(history)
            np.subtract(history, query, out=difference, where=valid)
            np.abs(difference, out=difference)
            lower_bounds[dim] = np.maximum(
                lower_bounds[dim], difference.max(axis=1)
            )
            mask &= lower_bounds[dim] < _MMAP_THRESHOLD

    candidate_positions = np.flatnonzero(mask)
    qualified: list[tuple[str, float, float]] = []
    candidate_total = 0
    for position in candidate_positions:
        candidate_id, row = _MMAP_CANDIDATES[int(position)]
        count0 = int(counts[row, dim0])
        count1 = int(counts[row, dim1])
        candidate_total += 1
        d0 = _exact_bottleneck(
            target0,
            _mmap_pairs(block0, row, count0),
            target_essential[dim0],
            _MMAP_ESS_B[dim0][row, : int(_MMAP_ESS_CNT[dim0][row])],
        )
        if not d0 < _MMAP_THRESHOLD:
            continue
        d1 = _exact_bottleneck(
            target1,
            _mmap_pairs(block1, row, count1),
            target_essential[dim1],
            _MMAP_ESS_B[dim1][row, : int(_MMAP_ESS_CNT[dim1][row])],
        )
        if d1 < _MMAP_THRESHOLD:
            qualified.append((candidate_id, d0, d1))

    qualified_count = len(qualified)
    if qualified_count < _MMAP_TOP_K:
        return target_id, candidate_total, qualified_count, []
    qualified.sort(key=lambda value: (value[2], value[1], value[0]))
    return target_id, candidate_total, qualified_count, qualified[:_MMAP_TOP_K]


def match_clouds(
    config: PipelineConfig,
    progress: ProgressCallback | None = None,
) -> dict[str, int]:
    """瓶颈匹配（mmap 零拷贝版本）。

    所有 worker 共享同一份只读内存映射的持久图，进程初始化参数只有文件路径和
    整数索引，因此总内存占用与并发数无关。
    """
    dimensions = tuple(config.distance_dimensions)
    dim0, dim1 = dimensions

    cloud_id_to_row = _ensure_diagram_mmap(config)
    count_path = config.work_dir / DIAGRAM_COUNTS_FILENAME
    h0_path = config.work_dir / f"diagrams_h{dim0}.npy"
    h1_path = config.work_dir / f"diagrams_h{dim1}.npy"
    counts = np.load(str(count_path))
    spec_paths, essential_count_paths, essential_birth_paths = _ensure_match_summary(
        config, dimensions, len(cloud_id_to_row)
    )

    def usable(cloud_id: str) -> bool:
        row = cloud_id_to_row.get(cloud_id)
        if row is None:
            return False
        return int(counts[row, dim0]) > 0 and int(counts[row, dim1]) > 0

    with closing(connect(config.database_path)) as db:
        records = load_cloud_records(db)
        target_ids = sorted(
            cloud_id
            for cloud_id, record in records.items()
            if record.cloud_date == config.as_of_date and usable(cloud_id)
        )
        candidate_ids = sorted(
            cloud_id
            for cloud_id, record in records.items()
            if record.cloud_date < config.as_of_date and usable(cloud_id)
        )
        if not target_ids:
            raise DataError(f"没有日期为 {config.as_of_date} 的可用点云")
        if not candidate_ids:
            raise DataError("没有历史候选点云")

        target_index = {cloud_id: cloud_id_to_row[cloud_id] for cloud_id in target_ids}
        candidates = tuple((cloud_id, cloud_id_to_row[cloud_id]) for cloud_id in candidate_ids)
        candidate_id_tuple = tuple(candidate_ids)
        candidate_rows = np.asarray([row for _, row in candidates], dtype=np.int64)
        pivot_rows_path, pivot_distance_paths = _ensure_pivot_cache(
            config,
            dimensions,
            candidate_id_tuple,
            candidate_rows,
            h0_path,
            h1_path,
            count_path,
            spec_paths,
            essential_count_paths,
            essential_birth_paths,
        )

        set_metadata(db, "matching_signature", f"in_progress:{config.matching_signature()}")
        db.commit()

        worker_count = config.resolved_matching_workers
        _progress(
            progress,
            "matching",
            0,
            len(target_ids),
            {"selected": 0, "workers": worker_count},
        )

        # initargs 只含路径、整数索引和标量参数，spawn 时的传输量以 MB 计而非 GB。
        initargs = (
            str(h0_path),
            str(h1_path),
            str(count_path),
            target_index,
            candidates,
            dimensions,
            config.distance_threshold,
            config.top_k,
            {dim: str(spec_paths[dim]) for dim in dimensions},
            {dim: str(essential_count_paths[dim]) for dim in dimensions},
            {dim: str(essential_birth_paths[dim]) for dim in dimensions},
            str(pivot_rows_path),
            {dim: str(pivot_distance_paths[dim]) for dim in dimensions},
        )
        if worker_count == 1:
            _init_match_worker_mmap(*initargs)
            results = map(_match_one_mmap, target_ids)
            executor = None
        else:
            executor = ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_match_worker_mmap,
                initargs=initargs,
            )
            results = executor.map(_match_one_mmap, target_ids, chunksize=1)

        selected = 0
        try:
            for index, result in enumerate(results, 1):
                target_id, candidate_total, qualified_count, top_matches = result
                save_matches(db, target_id, candidate_total, qualified_count, top_matches)
                selected += int(bool(top_matches))
                if index % 20 == 0:
                    db.commit()
                    LOGGER.info("匹配进度：%d/%d，已选 %d", index, len(target_ids), selected)
                _progress(
                    progress,
                    "matching",
                    index,
                    len(target_ids),
                    {"selected": selected, "workers": worker_count},
                )
        finally:
            if executor is not None:
                executor.shutdown()
        set_metadata(db, "matching_signature", config.matching_signature())
        db.commit()
        return {
            "targets": len(target_ids),
            "candidates": len(candidate_ids),
            "selected": selected,
            "workers": worker_count,
        }


def forecast(
    config: PipelineConfig,
    progress: ProgressCallback | None = None,
) -> dict[str, int]:
    with closing(connect(config.database_path)) as db:
        stored_signature = get_metadata(db, "matching_signature")
        if stored_signature != config.matching_signature():
            raise DataError("匹配参数已变化或尚未运行 match，请先重新执行匹配阶段")
        records = load_cloud_records(db)
        targets = db.execute(
            "SELECT target_id FROM match_runs WHERE status='selected' ORDER BY target_id"
        ).fetchall()
        majority = config.top_k // 2 + 1
        worker_count = config.resolved_forecast_workers
        _progress(
            progress,
            "forecast",
            0,
            len(targets),
            {"complete": 0, "error": 0, "workers": worker_count},
        )

        tasks: list[tuple[str, list[str]]] = []
        for target_row in targets:
            target_id = str(target_row["target_id"])
            match_rows = db.execute(
                "SELECT similar_id FROM matches WHERE target_id=? ORDER BY rank", (target_id,)
            ).fetchall()
            tasks.append((target_id, [str(row["similar_id"]) for row in match_rows]))

        @lru_cache(maxsize=None)
        def future(source_path: Path, cloud_date: object):
            return load_future_directions(source_path, cloud_date, config.forecast_horizon)

        def forecast_one(task: tuple[str, list[str]]):
            target_id, similar_ids = task
            try:
                if len(similar_ids) != config.top_k:
                    raise DataError(f"匹配数为 {len(similar_ids)}，期望 {config.top_k}")
                target_record = records[target_id]
                target_dates, target_diffs, target_directions, target_log_returns = future(
                    target_record.source_path, target_record.cloud_date
                )
                analog_directions = np.vstack([
                    future(records[item].source_path, records[item].cloud_date)[2]
                    for item in similar_ids
                ])
                vote_up = analog_directions.sum(axis=0)
                predictions = (vote_up >= majority).astype(np.int8)
                rows: list[tuple[object, ...]] = []
                for horizon in range(config.forecast_horizon):
                    actual = int(target_directions[horizon])
                    predicted = int(predictions[horizon])
                    rows.append((
                        horizon + 1,
                        target_dates[horizon].isoformat(),
                        float(target_diffs[horizon]),
                        float(target_log_returns[horizon]),
                        actual,
                        predicted,
                        int(vote_up[horizon]),
                        config.top_k,
                        int(actual == predicted),
                    ))
                return target_id, rows, None
            except Exception as exc:
                return target_id, [], str(exc)

        if worker_count == 1:
            results = map(forecast_one, tasks)
            executor = None
        else:
            executor = ThreadPoolExecutor(max_workers=worker_count)
            results = executor.map(forecast_one, tasks)

        complete = 0
        errors = 0
        try:
            for index, (target_id, rows, error) in enumerate(results, 1):
                if error is None:
                    save_forecasts(db, target_id, rows)
                    complete += 1
                else:
                    save_forecast_error(db, target_id, error)
                    errors += 1
                    LOGGER.warning("目标 %s 预测失败：%s", target_id, error)
                if index % 50 == 0:
                    db.commit()
                _progress(
                    progress,
                    "forecast",
                    index,
                    len(targets),
                    {"complete": complete, "error": errors, "workers": worker_count},
                )
        finally:
            if executor is not None:
                executor.shutdown()
        set_metadata(db, "forecast_signature", config.forecast_signature())
        db.commit()
        return {
            "selected": len(targets),
            "complete": complete,
            "error": errors,
            "workers": worker_count,
        }


def run_all(
    config: PipelineConfig,
    progress: ProgressCallback | None = None,
    reset: bool = False,
    force_rebuild: bool = False,
) -> dict[str, dict[str, int]]:
    from .reporting import generate_outputs

    return {
        "topology": build_topology(config, progress, reset=reset, force_rebuild=force_rebuild),
        "matching": match_clouds(config, progress),
        "forecast": forecast(config, progress),
        "report": generate_outputs(config),
    }
