from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from contextlib import closing, nullcontext

import numpy as np

from .build import (
    _BUILD_COMPLETED,  # noqa: F401  # re-export：保留 pipeline._BUILD_COMPLETED 兼容引用
    _BUILD_CONFIG,  # noqa: F401  # re-export：保留 pipeline._BUILD_CONFIG 兼容引用
    _build_stock,  # noqa: F401  # re-export：保留 pipeline._build_stock 兼容引用（test_cloud_finite_guard 经 pipeline.X 访问）
    _init_build_worker,  # noqa: F401  # re-export：保留 pipeline._init_build_worker 兼容引用（test_cloud_finite_guard 经 pipeline.X 访问）
    _reset_experiment,  # noqa: F401  # re-export：保留 pipeline._reset_experiment 兼容引用（test_refactor_wave2 经 pipeline.X 访问）
    build_topology,
)
from .config import PipelineConfig
from .data import (
    DataError,
    load_future_directions,  # noqa: F401  # re-export：保留 pipeline.load_future_directions 兼容引用（test_core 经 pipeline.X monkeypatch）
)
from .db_utils import (
    COMMIT_RETRY_ATTEMPTS,  # noqa: F401  # re-export：保留 pipeline.COMMIT_RETRY_ATTEMPTS 兼容引用
    COMMIT_RETRY_BASE_DELAY,  # noqa: F401  # re-export：保留 pipeline.COMMIT_RETRY_BASE_DELAY 兼容引用
    _clear_incomplete_marker,  # noqa: F401  # re-export：保留 pipeline._clear_incomplete_marker 兼容引用（test_refactor_wave2 经 pipeline.X 访问）
    _commit_with_retry,  # noqa: F401  # re-export：保留 pipeline._commit_with_retry 兼容引用（test_refactor_wave1/3 经 pipeline.X 访问）
    _is_sqlite_busy,  # noqa: F401  # re-export：保留 pipeline._is_sqlite_busy 兼容引用（test_refactor_wave3 经 pipeline.X 访问）
)
from .forecast import (
    forecast,
)
from .locks import (
    _EXPORT_LOCK_BOOT_TOKEN,  # noqa: F401  # re-export：保留 pipeline._EXPORT_LOCK_BOOT_TOKEN 兼容引用
    EXPORT_LOCK_HEARTBEAT_INTERVAL,  # noqa: F401  # re-export：供 pipeline.EXPORT_LOCK_HEARTBEAT_INTERVAL 兼容引用
    _export_lock_heartbeat,  # noqa: F401  # re-export：保留 pipeline._export_lock_heartbeat 兼容引用
    _lock_file_age,  # noqa: F401  # re-export：保留 pipeline._lock_file_age 兼容引用
    _lock_owner_is_valid,  # noqa: F401  # re-export：保留 pipeline._lock_owner_is_valid 兼容引用
    _pid_alive,  # noqa: F401  # re-export：保留 pipeline._pid_alive 兼容引用
    _read_lock_owner,  # noqa: F401  # re-export：保留 pipeline._read_lock_owner 兼容引用
    _read_lock_owner_and_token,  # noqa: F401  # re-export：保留 pipeline._read_lock_owner_and_token 兼容引用
)
from .matching import (
    PIVOT_PROGRESS_MIN_INTERVAL,  # noqa: F401  # re-export：保留 pipeline.PIVOT_PROGRESS_MIN_INTERVAL 兼容引用
    PIVOT_ROWS_FILENAME,  # noqa: F401  # re-export：保留 pipeline.PIVOT_ROWS_FILENAME 兼容引用（test_pivot_parallel_bit_exact 经 pipeline.X 访问）
    PIVOT_SIGNATURE_FILENAME,  # noqa: F401  # re-export：保留 pipeline.PIVOT_SIGNATURE_FILENAME 兼容引用
    Candidate,  # noqa: F401  # re-export：保留 pipeline.Candidate 兼容引用
    _candidate_diagram,  # noqa: F401  # re-export：保留 pipeline._candidate_diagram 兼容引用（test_pivot_parallel_bit_exact 经 pipeline.X 访问）
    _ensure_pivot_cache,  # noqa: F401  # re-export：保留 pipeline._ensure_pivot_cache 兼容引用（matching 已收敛为同模块直接调用）
    _exact_bottleneck,  # noqa: F401  # re-export：保留 pipeline._exact_bottleneck 兼容引用
    _exact_distances_between_rows,  # noqa: F401  # re-export：保留 pipeline._exact_distances_between_rows 兼容引用（test_pivot_parallel_bit_exact 经 pipeline.X 访问）
    _init_match_worker_mmap,  # noqa: F401  # re-export：保留 pipeline._init_match_worker_mmap 兼容引用
    _init_pivot_worker,  # noqa: F401  # re-export：保留 pipeline._init_pivot_worker 兼容引用（test_pivot_parallel_bit_exact 经 pipeline.X 访问）
    _load_done_target_ids,  # noqa: F401  # re-export：保留 pipeline._load_done_target_ids 兼容引用
    _match_one_mmap,  # noqa: F401  # re-export：保留 pipeline._match_one_mmap 兼容引用
    _mmap_handle,  # noqa: F401  # re-export：保留 pipeline._mmap_handle 兼容引用
    _mmap_pairs,  # noqa: F401  # re-export：保留 pipeline._mmap_pairs 兼容引用
    _persist_matches,  # noqa: F401  # re-export：保留 pipeline._persist_matches 兼容引用
    _pivot_cache_is_current,  # noqa: F401  # re-export：保留 pipeline._pivot_cache_is_current 兼容引用
    _pivot_column_distances,  # noqa: F401  # re-export：保留 pipeline._pivot_column_distances 兼容引用（test_pivot_parallel_bit_exact 经 pipeline.X 访问）
    _pivot_layer_accepts,  # noqa: F401  # re-export：保留 pipeline._pivot_layer_accepts 兼容引用
    _pivot_paths,  # noqa: F401  # re-export：保留 pipeline._pivot_paths 兼容引用
    _reselect_completed_match_targets,  # noqa: F401  # re-export：保留 pipeline._reselect_completed_match_targets 兼容引用
    _select_next_pivot,  # noqa: F401  # re-export：保留 pipeline._select_next_pivot 兼容引用（test_pivot_selection 经 pipeline.X 访问）
    match_clouds,
)
from .mmap_io import (
    DIAGRAM_COUNTS_FILENAME,  # noqa: F401  # re-export：保留 pipeline.DIAGRAM_COUNTS_FILENAME 兼容引用
    DIAGRAM_INDEX_FILENAME,  # noqa: F401  # re-export：保留 pipeline.DIAGRAM_INDEX_FILENAME 兼容引用
    DIAGRAM_SIGNATURE_FILENAME,  # noqa: F401  # re-export：保留 pipeline.DIAGRAM_SIGNATURE_FILENAME 兼容引用
    EXPORT_LOCK_POLL_INTERVAL,  # noqa: F401  # re-export：供 pipeline.EXPORT_LOCK_POLL_INTERVAL 兼容引用
    EXPORT_LOCK_STALE_SECONDS,  # noqa: F401  # re-export：供 pipeline.EXPORT_LOCK_STALE_SECONDS 兼容引用
    EXPORT_PROGRESS_MIN_INTERVAL,  # noqa: F401  # re-export：保留 pipeline.EXPORT_PROGRESS_MIN_INTERVAL 兼容引用
    _cleanup_stale_mmap_artifacts,  # noqa: F401  # re-export：保留 pipeline._cleanup_stale_mmap_artifacts 兼容引用
    _clear_mmap_handles,  # noqa: F401  # re-export：保留 pipeline._clear_mmap_handles 兼容引用
    _close_mmap,  # noqa: F401  # re-export：保留 pipeline._close_mmap 兼容引用
    _diagram_mmap_paths,  # noqa: F401  # re-export：保留 pipeline._diagram_mmap_paths 兼容引用
    _ensure_diagram_mmap,  # noqa: F401  # re-export：保留 pipeline._ensure_diagram_mmap 兼容引用
    _export_diagrams_to_mmap,  # noqa: F401  # re-export：保留 pipeline._export_diagrams_to_mmap 兼容引用（tests/_diag_e1.py 经 pipeline.X 访问）
    _export_dimensions,  # noqa: F401  # re-export：保留 pipeline._export_dimensions 兼容引用
    _export_lock,  # noqa: F401  # re-export：保留 pipeline._export_lock 兼容引用
    _load_diagram_index,  # noqa: F401  # re-export：保留 pipeline._load_diagram_index 兼容引用
    _mmap_is_current,  # noqa: F401  # re-export：保留 pipeline._mmap_is_current 兼容引用
    _read_npy_header,  # noqa: F401  # re-export：保留 pipeline._read_npy_header 兼容引用
    _remove_unused_dimension_files,  # noqa: F401  # re-export：保留 pipeline._remove_unused_dimension_files 兼容引用
    _replace_atomic,  # noqa: F401  # re-export：保留 pipeline._replace_atomic 兼容引用
    _validate_diagram_mmap_files,  # noqa: F401  # re-export：保留 pipeline._validate_diagram_mmap_files 兼容引用
)
from .persistence import (
    CommitBatch,  # noqa: F401  # re-export：保留 pipeline.CommitBatch 兼容引用
)
from .policy import (
    POLICY,
    StagePolicy,  # noqa: F401  # re-export：保留 pipeline.StagePolicy 兼容引用（测试经 pipeline.StagePolicy 访问）
)
from .pooling import (
    CONSUME_SLOW_WARN_SECONDS,  # noqa: F401  # re-export：保留 pipeline.CONSUME_SLOW_WARN_SECONDS 兼容引用
    STAGE_TASK_TIMEOUT,  # noqa: F401  # re-export：保留 pipeline.STAGE_TASK_TIMEOUT 兼容引用（test_refactor_wave2 经 pipeline.X 访问）
    STAGE_WATCHDOG,  # noqa: F401  # re-export：保留 pipeline.STAGE_WATCHDOG 兼容引用
    STAGE_WATCHDOG_INTERVAL,  # noqa: F401  # re-export：保留 pipeline.STAGE_WATCHDOG_INTERVAL 兼容引用
    _make_pool,  # noqa: F401  # re-export：保留 pipeline._make_pool 兼容引用
    _run_consumer,  # noqa: F401  # re-export：保留 pipeline._run_consumer 兼容引用
    _run_pool_stage,  # noqa: F401  # re-export：保留 pipeline._run_pool_stage 兼容引用
    _StageWatchdog,  # noqa: F401  # re-export：保留 pipeline._StageWatchdog 兼容引用
    _terminate_pool,  # noqa: F401  # re-export：保留 pipeline._terminate_pool 兼容引用
)
from .signatures import (
    _compute_source_signature,  # noqa: F401  # re-export：保留 pipeline._compute_source_signature 兼容引用
    _count_exportable_diagrams,  # noqa: F401  # re-export：保留 pipeline._count_exportable_diagrams 兼容引用
    _exportable_diagram_rows,  # noqa: F401  # re-export：保留 pipeline._exportable_diagram_rows 兼容引用
    _pivot_signature,  # noqa: F401  # re-export：保留 pipeline._pivot_signature 兼容引用
    _stage_signature,
    matching_resume_key,  # noqa: F401  # re-export：保留 pipeline.matching_resume_key 兼容引用（test_dir_derivation/test_match_signature_finalization/test_storage 经 pipeline.X 访问）
)
from .storage import (
    connect,
    get_stage_status,
)
from .topology import (
    backend_info,
    set_topology_backend,
)

LOGGER = logging.getLogger(__name__)


# ── 计算阶段挂死防护参数（P0）────────────────────────────────────
# 持续同调与瓶颈距离最终都落到 GUDHI 的 C 扩展里：极端输入下它可能长时间不返回，
# 甚至直接把 worker 进程打成段错误。这两种情况 Python 的 try/except 都拦不住
# （前者不抛异常，后者进程已经没了），表现就是父进程在 executor.map 上永久阻塞。
# 因此在父进程侧统一加「单任务超时 → 记错 → 强杀并重建进程池 → 继续剩余任务」。
# 两个阈值都可用环境变量覆盖，默认值取得比较宽松，正常的慢任务不会被误杀。


# 候选图跨目标缓存（WP3）：worker 内不同目标扫同一候选时复用已拷贝的有限点数组，
# 减少 numpy 拷贝。per-worker 进程级状态，跨目标复用；受内存上限约束，超限退化为逐候选直算。
_MMAP_CAND_DIAG: dict[int, tuple[np.ndarray, np.ndarray]] = {}
MAX_CAND_CACHE = 40_000
# 瓶颈早停下界（WP4，可选，默认关闭）：d0 实算前用持久度谱下界跳过明显不达标对。
# 默认 False——绝不改变既有默认路径行为；开启后须 self-test 全绿方视为无损。
ENABLE_LB_PRUNE: bool = False
_LB_PRUNE_COUNT: int = 0


# ── mmap 零拷贝匹配：worker 进程内的轻量全局状态 ──────────────────
# 这里只存文件路径与整数索引，持久图数据始终留在 OS page cache 中，
# 由所有 worker 通过只读内存映射共享，内存占用不随并发数增长。
_MMAP_PATHS: dict[int, str] = {}
_MMAP_COUNT_PATH: str = ""
_MMAP_TARGET_IDX: dict[str, int] = {}
_MMAP_CANDIDATES: tuple[tuple[str, int], ...] = ()
_MMAP_DIMS: tuple[int, int] = (0, 1)
# 候选有限点对跨目标缓存阈值（H0=目标维度、H1=候选维度）：低于此值的瓶颈距离
# 视为「有限点对齐」，触发 WP3 跨目标持久图缓存复用（见 _match_one_mmap）。
_MATCH_MMAP_THRESHOLD_H0: float = 0.1
_MATCH_MMAP_THRESHOLD_H1: float = 0.1
_MMAP_HANDLES: dict[str, np.ndarray] = {}
# 完整候选排序落盘的安全上限（步骤 A）：worker 把排序后的合格候选截到该上限，
# 远超出常规 top_k（1~数十），足以覆盖改 top_k 后的「重新截断」复用；
# 同时为有大量合格邻居的目标封顶存储，避免 match_candidates 无界膨胀。
# 若业务需要更大的 top_k，调高此常量即可（超出部分将退化为重算距离）。
MATCH_CANDIDATE_CAP: int = POLICY.match_candidate_cap  # 兼容别名（R5）
# 瓶颈距离算法标识（v3）：标签由 config._distance_algo_label 按后端生成——
# 源码运行 = exact-essential-v3-topp；便携 EXE（native_c_dll）= exact-essential-v3-native。
# matching_resume_key 等签名统一引用该函数（避免硬编码漂移）。
# 每个点云的有限持续度谱(H0)：_MMAP_SPEC[row, i] = 第 i+1 大的持续度(death-birth)，
# 降序、不足 _SPEC_RANKS 秩补 0。本质类不参与。用于瓶颈距离的逐秩下界剪枝。
# 第 0 列即最大持续度，是原 pmax；nan 表示该点云无有限持久对。
_SPEC_RANKS: int = 8

_MMAP_PIVOT_ROWS: np.ndarray = np.array([], dtype=np.int64)
_MMAP_PIVOT_DIST: dict[int, np.ndarray] = {}
_MMAP_SPEC: np.ndarray = np.array([])
# 本质同调类(death=+inf)的 birth：按维度存放
#   _MMAP_ESS_CNT[dim][row] = 该点云在该维度的本质类个数
#   _MMAP_ESS_B[dim][row, :cnt] = 对应 birth，按降序排列（用于最优逐秩匹配）
_MMAP_ESS_CNT: dict[int, np.ndarray] = {}
_MMAP_ESS_B: dict[int, np.ndarray] = {}

ProgressCallback = Callable[[str, int, int, Mapping[str, int]], None]


class StageFailure(Exception):
    """编排层异常：任一阶段失败时抛出，携带已完成阶段的结果，绝不吞首因（R4-A）。

    成功路径不受影响；本异常仅用于「附加可观测信息后原样上抛」（保留 ``__cause__``），
    不引入任何兜底逻辑。``cli.py`` / ``run.py`` 顶层 ``except Exception`` 可自然兜住。
    """

    def __init__(self, stage: str, completed: Mapping[str, dict[str, int]]) -> None:
        self.stage = stage
        self.completed: dict[str, dict[str, int]] = dict(completed)
        super().__init__(f"流水线在阶段 {stage!r} 失败；已完成阶段：{sorted(self.completed)}")


def run_all(
    config: PipelineConfig,
    progress: ProgressCallback | None = None,
    reset: bool = False,
    force_rebuild: bool = False,
    force_content_hash: bool = False,
    log_queue=None,
) -> dict[str, dict[str, int]]:
    set_topology_backend(config.topology_backend)
    # 后端初始化 fail-fast：切换后端后立即加载并校验其依赖就绪（如 Topp 是否安装）。
    # 后端提供 backend_info 入口（ripser_topp）时，缺依赖在此处即抛 TopologyError，
    # 避免首笔瓶颈距离计算时才暴露；无该入口的后端（gudhi / native_c_dll）返回 None，
    # 静默跳过，不影响其既有惰性报错路径。
    backend_init = backend_info()
    if backend_init is not None:
        LOGGER.info(
            "拓扑后端初始化：name=%s version=%s source=%s",
            getattr(backend_init, "name", "?"),
            getattr(backend_init, "version", "?"),
            getattr(backend_init, "source", "?"),
        )
    from .reporting import generate_outputs

    # 阶段编排（R4-A）：拆字典字面量为显式阶段循环，仅 topology 收 kwargs。
    # 任一阶段失败时抛 StageFailure 携带已完成阶段结果，绝不吞首因；成功路径返回值不变。
    results: dict[str, dict[str, int]] = {}
    # R4-B：阶段级幂等跳过——查 stage_runs 直接判定是否跳过已完成阶段。
    # reset / force_rebuild 会改变底层数据，此时禁用跳过（宁可多跑，避免讹用陈旧 stage_runs 状态）。
    skip_enabled = not (reset or force_rebuild)
    # R4-B：标记 forecast 是否在本轮实际执行；仅当被跳过（依赖 stage_runs 历史 completed）时，
    # report 守卫才需校验其历史完成数据是否真实存在。
    _forecast_ran = False
    # 兼容无 database_path 的轻量配置（如单元测试桩）：无库则跳过判定与守卫整体禁用。
    _db_path = getattr(config, "database_path", None)
    _skip_db_cm = closing(connect(_db_path)) if _db_path is not None else nullcontext()
    with _skip_db_cm as _skip_db:
        for name, fn in (
            (
                "topology",
                lambda cfg, prog: build_topology(
                    cfg,
                    prog,
                    reset=reset,
                    force_rebuild=force_rebuild,
                    force_content_hash=force_content_hash,
                    log_queue=log_queue,
                ),
            ),
            (
                "matching",
                lambda cfg, prog: match_clouds(cfg, prog, log_queue=log_queue),
            ),
            ("forecast", forecast),
            # report(generate_outputs) 透传 progress：让结果输出阶段回弹进度，
            # 避免预测天数很大时该阶段整段同步落盘无反馈、GUI 误判卡死/阻塞。
            ("report", lambda cfg, prog: generate_outputs(cfg, progress=prog)),
        ):
            # R4-B：跳过判定（topology/matching/forecast）。status=completed 且 signature 与当前配置一致才跳过。  # noqa: E501
            if (
                _skip_db is not None
                and skip_enabled
                and name in ("topology", "matching", "forecast")
            ):
                _status, _stored_sig = get_stage_status(_skip_db, name)
                if _status == "completed" and _stored_sig == _stage_signature(config, name):
                    LOGGER.info("阶段 %s 已完成且签名一致，跳过（幂等）", name)
                    results[name] = {"skipped": True}
                    continue
            if name == "forecast":
                _forecast_ran = True
            if name == "report":  # noqa: SIM102
                # R4-B：report 前置校验——仅当 forecast 本轮被跳过（未实际执行）时，
                # 才校验其历史完成数据是否真实存在，避免产出空报告；
                # forecast 本轮已实际执行则信任其产出。
                if not _forecast_ran and _skip_db is not None:
                    _fcount = _skip_db.execute(
                        "SELECT COUNT(*) AS n FROM forecast_runs WHERE status='complete'"
                    ).fetchone()["n"]
                    if _fcount == 0:
                        raise DataError(
                            "forecast 阶段被跳过但其完成数据为空（forecast_runs 无 complete 记录），"  # noqa: E501
                            "拒绝产出空报告；请重新运行 forecast 阶段"
                        )
            try:
                results[name] = fn(config, progress)
            except Exception as exc:
                raise StageFailure(stage=name, completed=results) from exc
    return results
