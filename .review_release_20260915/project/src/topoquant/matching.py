"""mmap 零拷贝匹配原语（Phase 3 从 god-module 抽取）。

机械搬迁（byte-for-byte）：原 ``pipeline.py`` 中的 mmap 匹配 worker 初始化
（_init_match_worker_mmap）、句柄/点对读取（_mmap_handle/_mmap_pairs）、候选图
跨目标缓存（_candidate_diagram）与精确瓶颈距离（_exact_bottleneck）平移至此；
``pipeline.py`` 以 ``from .matching import (...)`` 重导出，外部 ``pipeline.X``
引用零破坏。

共享可变全局（_MMAP_* / _MATCH_* / Candidate 等）**保留在 pipeline.py**（单一
真相源）：这些全局会被 _init_match_worker_mmap 重绑定，若随迁将导致跨模块别名
分裂（重导出引用变陈旧）；本模块相关函数经惰性 ``from . import pipeline`` 在
运行时访问，保证单一真相、无导入期循环（与 mmap_io._clear_mmap_handles 同模式）。
pivot 簇（_pivot_paths/_ensure_pivot_cache/_pivot_layer_accepts 等）迁入后已收敛为
同模块直接调用，仅 _MMAP_* 全局仍经惰性 pipeline.X 访问。
函数体除全局引用改写为 pipeline.X 外逐字节一致，无循环依赖（仅依赖
_logging/data/topology 与标准库）。
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import os
import shutil
import socket
import sqlite3
import tempfile
import time
from collections import namedtuple
from collections.abc import Callable, Mapping
from contextlib import closing, suppress
from pathlib import Path

import numpy as np

from ._logging import configure_logging
from .config import PipelineConfig
from .data import DataError
from .db_utils import _clear_incomplete_marker, _commit_with_retry, _write_incomplete_marker
from .mmap_io import (
    DIAGRAM_COUNTS_FILENAME,
    _ensure_diagram_mmap,
    _export_lock,
    _progress,
    _read_npy_header,
    _replace_atomic,
)
from .persistence import CommitBatch
from .policy import POLICY
from .pooling import STAGE_TASK_TIMEOUT, _run_consumer, _run_pool_stage
from .signatures import _pivot_signature, matching_resume_key
from .storage import (
    connect,
    get_stage_status,
    load_cloud_records,
    save_match_candidates,
    save_matches,
    upsert_stage_run,
)
from .topology import TopologyError, finite_bottleneck_distance, set_topology_backend

LOGGER = logging.getLogger(__name__)

# 与 pipeline.py 同口径的进度回调类型别名（match_clouds 签名注解使用）。
ProgressCallback = Callable[[str, int, int, Mapping[str, int]], None]


def _init_match_worker_mmap(  # noqa: PLR0913, PLR0917
    h0_path: str,
    h1_path: str,
    count_path: str,
    target_index: dict[str, int],
    candidates: tuple[Candidate, ...],  # 注解 Candidate 定义于本模块（namedtuple）
    dimensions: tuple[int, int],
    threshold_h0: float,
    threshold_h1: float,
    pivot_rows_path: str = "",
    pivot_distance_paths: dict[int, str] | None = None,
    log_queue=None,
) -> None:
    """匹配 worker 初始化：只接收文件路径与整数索引，不搬运任何持久图数据。

    top_k 不再传入 worker（步骤 A）：它只影响落盘截断几条，与距离计算无关，
    故从 worker 初始化参数中彻底移除，确保 top_k 不进入距离计算的任何缓存键。
    """
    # 共享可变全局保留在 pipeline.py（单一真相源）；此处经惰性导入访问，
    # 避免重绑定全局随迁导致跨模块别名分裂（Phase 3 决策，见模块 docstring）。
    from . import pipeline

    pipeline._MMAP_PATHS = dict(zip(dimensions, (h0_path, h1_path), strict=False))
    pipeline._MMAP_COUNT_PATH = count_path
    pipeline._MMAP_TARGET_IDX = target_index
    pipeline._MMAP_CANDIDATES = candidates
    pipeline._MMAP_DIMS = dimensions
    pipeline._MATCH_MMAP_THRESHOLD_H0 = threshold_h0
    pipeline._MATCH_MMAP_THRESHOLD_H1 = threshold_h1
    pipeline._MMAP_HANDLES = {}

    # 预计算两类派生量（复用已打开的 mmap 句柄，避免重复加载）：
    #   1) 有限点的持续度谱(H0，降序前 _SPEC_RANKS 秩) —— 供瓶颈距离逐秩下界剪枝
    #   2) 各维度本质类(death=+inf)的 birth 降序表 —— 供精确瓶颈距离使用
    # 派生量同样写入 pipeline 命名空间（单一真相）。
    dim0, _ = dimensions
    counts_full = _mmap_handle(pipeline._MMAP_COUNT_PATH)
    spec, ess_cnt, ess_b = _precompute_derived(dim0, dimensions, counts_full)
    pipeline._MMAP_SPEC = spec
    pipeline._MMAP_ESS_CNT = ess_cnt
    pipeline._MMAP_ESS_B = ess_b
    # pivot 距离矩阵（可选）：供第 4 层剪枝读取；未提供（如仅构建缓存时）置空。
    if pivot_rows_path:
        pipeline._MMAP_PIVOT_ROWS = _mmap_handle(pivot_rows_path)
        pipeline._MMAP_PIVOT_DIST = {
            dim: _mmap_handle(pivot_distance_paths[dim]) for dim in dimensions
        }
    else:
        pipeline._MMAP_PIVOT_ROWS = np.array([], dtype=np.int64)
        pipeline._MMAP_PIVOT_DIST = {}
    configure_logging("worker-silent", log_queue=log_queue)
    # P0-2 防回归：pivot 矩阵行数必须等于候选数（ordinal 与候选枚举序号同序）。
    # 以 _MMAP_CANDIDATES 守卫：_ensure_pivot_cache 会以空候选 () 调用本函数建缓存，
    # 此时矩阵也为空，若不加守卫会误炸。未来任何 ordinal/row 错配立即抛错，而非静默误读。
    if pipeline._MMAP_CANDIDATES:
        for dim in pipeline._MMAP_DIMS:
            assert pipeline._MMAP_PIVOT_DIST[dim].shape[0] == len(pipeline._MMAP_CANDIDATES), (  # noqa: S101
                f"pivot 矩阵行数 {pipeline._MMAP_PIVOT_DIST[dim].shape[0]} "
                f"!= 候选数 {len(pipeline._MMAP_CANDIDATES)}：ordinal/row 映射错配"
            )


def _precompute_derived(
    dim0: int,
    dimensions: tuple[int, int],
    counts_full: np.ndarray,
) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    """预计算派生量：有限点持续度谱 + 各维度本质类 birth 降序表（M4：提取自
    _init_match_worker_mmap，逻辑逐字平移；_MMAP_* 全局仍经惰性 pipeline.X 访问）。
    """
    from . import pipeline  # _MMAP_* 驻留 pipeline.py（单一真相源）

    n = counts_full.shape[0]
    spec = np.full((n, pipeline._SPEC_RANKS), np.nan)
    ess_cnt: dict[int, np.ndarray] = {}
    ess_b: dict[int, np.ndarray] = {}
    for dim in dimensions:
        block = _mmap_handle(pipeline._MMAP_PATHS[dim])
        col = counts_full[:, dim]
        cnt = np.zeros(n, dtype=np.int32)
        births: list[np.ndarray | None] = [None] * n
        width = 0
        for row_idx in range(n):
            c = int(col[row_idx])
            if c == 0:
                continue
            pts = block[row_idx, :c, :]
            fin_b = np.isfinite(pts[:, 0])
            fin_d = np.isfinite(pts[:, 1])
            if dim == dim0:
                m = fin_b & fin_d
                # 秩不足时补 0：等价于"该秩上没有持久对"，持续度视为 0，
                # 与定理中 nu_i (i 超出图规模时为 0) 的约定一致。
                # 非空图但完全没有有限点（只含本质类）时整行即为全 0，
                # 这同样是合法的谱，不能留成 NaN——留 NaN 只会白白放弃剪枝。
                spec[row_idx, :] = 0.0
                if m.any():
                    finite = pts[m]
                    ranked = np.sort(finite[:, 1] - finite[:, 0])[::-1]
                    take = min(pipeline._SPEC_RANKS, ranked.shape[0])
                    spec[row_idx, :take] = ranked[:take]
            em = fin_b & ~fin_d
            k = int(em.sum())
            if k:
                cnt[row_idx] = k
                births[row_idx] = np.sort(np.asarray(pts[em, 0], dtype=np.float64))[::-1]
                width = max(width, k)
        arr = np.zeros((n, max(width, 1)), dtype=np.float64)
        for row_idx, birth_values in enumerate(births):
            if birth_values is not None:
                arr[row_idx, : birth_values.shape[0]] = birth_values
        ess_cnt[dim] = cnt
        ess_b[dim] = arr
    return spec, ess_cnt, ess_b


def _mmap_handle(path: str) -> np.ndarray:
    """按需打开并缓存只读内存映射。

    ``np.load(mmap_mode='r')`` 是 O(1) 的虚拟地址映射，不搬运数据；
    进程内缓存句柄可避免为每个目标点云重复打开文件。
    """
    from . import pipeline  # _MMAP_HANDLES 驻留 pipeline.py（单一真相源）

    handle = pipeline._MMAP_HANDLES.get(path)
    if handle is None:
        try:
            handle = np.load(path, mmap_mode="r")
        except (OSError, EOFError, ValueError) as exc:
            raise DataError(
                f"无法以只读内存映射打开持久图文件 {path}：{exc}。"
                f"该文件可能为空或被截断，请重新运行 build 阶段以触发自动重导出。"
            ) from exc
        pipeline._MMAP_HANDLES[path] = handle
    return handle


def _mmap_pairs(block: np.ndarray, row: int, count: int) -> np.ndarray:
    """从 mmap 中取出**有限**持久对，拷贝成 C 连续数组交给 GUDHI。

    本质同调类（essential class，death=+inf）在这里被排除，改由 ``_exact_bottleneck``
    单独精确处理——见该函数的推导。直接把 inf 喂给 GUDHI 会让本质类个数不同的图对
    返回 inf（正确），但个数相同的图对结果依赖实现细节；拆开处理语义更明确。
    """
    pts = block[row, :count, :]
    mask = np.isfinite(pts[:, 0]) & np.isfinite(pts[:, 1])
    return np.ascontiguousarray(pts[mask], dtype=np.float64)


def _candidate_diagram(row: int) -> tuple[np.ndarray, np.ndarray]:
    """返回某行的 (H0, H1) 有限持久点对（C 连续拷贝），并跨目标 memoize（WP3）。

    仅避免重复 ``np.ascontiguousarray`` 拷贝，**不改变** ``_exact_bottleneck`` 的输入口径，
    数值与直接调用 ``_mmap_pairs`` 逐位一致。缓存为 per-worker 进程级状态，受
    ``MAX_CAND_CACHE`` 上限约束；超限自动退化为逐候选直算（与原行为等价）。
    """
    from . import pipeline  # _MMAP_* 驻留 pipeline.py（单一真相源）

    cached = pipeline._MMAP_CAND_DIAG.get(row)
    if cached is not None:
        return cached
    dim0, dim1 = pipeline._MMAP_DIMS
    counts = _mmap_handle(pipeline._MMAP_COUNT_PATH)
    block0 = _mmap_handle(pipeline._MMAP_PATHS[dim0])
    block1 = _mmap_handle(pipeline._MMAP_PATHS[dim1])
    c0 = int(counts[row, dim0])
    c1 = int(counts[row, dim1])
    pair = (_mmap_pairs(block0, row, c0), _mmap_pairs(block1, row, c1))
    if len(pipeline._MMAP_CAND_DIAG) < pipeline.MAX_CAND_CACHE:
        pipeline._MMAP_CAND_DIAG[row] = pair
    return pair


def _exact_bottleneck(
    finite_a: np.ndarray,
    finite_b: np.ndarray,
    ess_a: np.ndarray,
    ess_b: np.ndarray,
) -> float:
    """含本质类的**精确**瓶颈距离 W_inf。

    本质类是 (b, +inf)。在 L_inf 意义下：
      - 本质类 ↔ 有限点：|d - (+inf)| = inf；
      - 本质类 ↔ 对角线：代价 = (+inf - b)/2 = inf。
    故本质类只能与对方的本质类一一配对。由此：

      1) 两图本质类**个数不同** ==> 不存在有限代价的匹配 ==> W_inf = +inf；
      2) 个数相同时，本质类之间的最优配对代价 = min over 双射 σ: max_i |b_i - b'_{σ(i)}|，
         这是经典的"瓶颈指派"问题，在一维实数集上**按序配对即最优**
         （交换论证：若 σ 存在逆序对，交换后最大代价不增）。故降序排列后逐位取差即可。
      3) 有限点部分与本质类部分互不干扰（跨类配对代价为 inf），
         整体瓶颈距离 = max(有限部分 W_inf, 本质部分逐位最大差)。

    注意这里必须用 ``finite_bottleneck_distance`` 而不是 ``bottleneck_distance``：
    后者带有"整图为空即 inf"的业务约定，而**有限部分**为空是完全正常的
    （例如 H1 只含一个本质类），误用会把这类图对全部判为不合格。
    整图为空的情形已由调用方的 ``count == 0`` 前置判断拦下。
    """
    if ess_a.shape[0] != ess_b.shape[0]:
        return float("inf")
    distance = float(finite_bottleneck_distance(finite_a, finite_b))
    if ess_a.shape[0]:
        essential = float(np.abs(ess_a - ess_b).max())
        distance = max(distance, essential)
    return distance


# ────────────────────────────────────────────────────────────────────────
# 融合升级：瓶颈匹配第 4 层（pivot）剪枝的全部实现
# ────────────────────────────────────────────────────────────────────────


def _match_one_mmap(target_id: str) -> tuple[str, int, int, list[tuple[str, float, float]]]:  # noqa: PLR0912,PLR0915
    """单个目标点云对全部候选的瓶颈匹配（读共享 mmap）。"""
    from . import pipeline  # 共享全局驻留 pipeline.py（单一真相源，避免别名分裂）

    dim0, dim1 = pipeline._MMAP_DIMS
    counts = _mmap_handle(pipeline._MMAP_COUNT_PATH)

    target_row = pipeline._MMAP_TARGET_IDX[target_id]
    target_count0 = int(counts[target_row, dim0])
    target_count1 = int(counts[target_row, dim1])
    if target_count0 == 0 or target_count1 == 0:
        return target_id, 0, 0, []

    # WP3：目标自身的有限点对也走跨目标缓存（目标行往往也是其它目标的候选，可复用）。
    target0, target1 = _candidate_diagram(target_row)

    target_spec = pipeline._MMAP_SPEC[target_row]
    target_spec_ok = bool(np.isfinite(target_spec[0]))
    two_thr = 2.0 * pipeline._MATCH_MMAP_THRESHOLD_H0
    ess_cnt0 = pipeline._MMAP_ESS_CNT[dim0]
    ess_cnt1 = pipeline._MMAP_ESS_CNT[dim1]
    ess_b0 = pipeline._MMAP_ESS_B[dim0]
    ess_b1 = pipeline._MMAP_ESS_B[dim1]
    t_k0 = int(ess_cnt0[target_row])
    t_k1 = int(ess_cnt1[target_row])
    t_ess0 = ess_b0[target_row, :t_k0]
    t_ess1 = ess_b1[target_row, :t_k1]

    # 第 4 层（pivot）剪枝的准备：预计算目标到各 pivot 的精确距离（与在线实算同口径）。
    # 之后每个候选用三角不等式下界 |W(A,B)-W(B,P)| 判断是否必被拒。
    pivot_query: dict[int, np.ndarray] = {}
    if pipeline._MMAP_PIVOT_ROWS.size:
        # P1 合并双维计算：每个 pivot 只算一次精确距离，再拆给两维，
        # 避免原双层循环对每维各重算一遍（pivot 数 × 目标数级别，量级小但应修）。
        w0 = np.empty(len(pipeline._MMAP_PIVOT_ROWS), dtype=np.float64)
        w1 = np.empty(len(pipeline._MMAP_PIVOT_ROWS), dtype=np.float64)
        for k, pivot_row in enumerate(pipeline._MMAP_PIVOT_ROWS):
            # pivot 簇已与 _exact_distances_between_rows 同模块（Phase 3 收敛），直接调用。
            pair = _exact_distances_between_rows((target_row, int(pivot_row)))
            w0[k], w1[k] = float(pair[0]), float(pair[1])
        dim0, dim1 = pipeline._MMAP_DIMS
        pivot_query[dim0] = w0
        pivot_query[dim1] = w1

    qualified: list[tuple[str, float, float]] = []
    candidate_total = 0
    # 退化候选聚合计数：候选循环里持久图退化的候选原本会逐条 LOGGER.warning，候选规模大时
    # （数千~数万）会刷屏。改为计数、循环结束后对当前目标汇总一条（见下方 degenerate_skipped），
    # 既不丢失"多少候选因退化被跳过"的信息，又把刷屏从 O(候选数) 降到 O(目标数)。计算逻辑不变。
    degenerate_skipped = 0
    for ordinal, (candidate_id, row) in enumerate(pipeline._MMAP_CANDIDATES):
        if candidate_id == target_id:
            continue
        count0 = int(counts[row, dim0])
        count1 = int(counts[row, dim1])
        if count0 == 0 or count1 == 0:
            continue
        # 本质类个数剪枝——严格无损：个数不同 ==> d0 = +inf ==> 必被拒。
        # 见 _exact_bottleneck 的推导；这是零成本的整数比较。
        if int(ess_cnt0[row]) != t_k0:
            continue
        # 逐秩持续度谱下界剪枝——严格无损，不会误杀任何合格对。
        #
        # 定理: 设 mu_1 >= mu_2 >= ... 为 D1 有限点的持续度降序谱（秩超出图规模时取 0），
        #       nu 同理，delta = W_inf(D1,D2)，则对每个秩 i 都有 |mu_i - nu_i| <= 2*delta。
        # 证明: 设 eta 为达到 delta 的最优部分匹配。取 D1 中持续度最大的 i 个点。
        #       - 若某点 p 满足 pers(p) > 2*delta，它不可能被 eta 匹配到对角线
        #         （代价 = pers(p)/2 > delta），只能匹配到某个 q ∈ D2；
        #         由 ||p-q||_inf <= delta 得 pers(q) >= pers(p) - 2*delta。
        #       - 于是 D2 中持续度 >= mu_i - 2*delta 的点至少有 i 个
        #         （前 i 大的点里，pers > 2*delta 的都有像；pers <= 2*delta 的点
        #          其 mu_i - 2*delta <= 0，条件平凡成立）。按定义 nu_i >= mu_i - 2*delta。
        #       - 交换 D1/D2 得 mu_i >= nu_i - 2*delta。合并即 |mu_i - nu_i| <= 2*delta。QED
        #       i=1 即退化为原来的"最大持续度"下界 delta >= |P1-P2|/2。
        #
        # 推论: 若存在秩 i 使 |mu_i - nu_i| > 2*threshold，则 delta > threshold，
        #       该候选必被下面的 `d0 < threshold` 拒绝，跳过与实算完全等价。
        #       又因合格判定是 (d0 < thr) AND (d1 < thr) 的合取，d0 单侧被拒即整对被拒，
        #       所以只用 H0 的下界剪枝同样安全。
        #
        # 实测(40383 个点云, 阈值 0.1): 相比只用 i=1，实算量再降 83.3%，
        # 累计剪枝率 86.7% -> 97.8%；1500 个被剪对暴力实算误杀 0，最小真实 d0=0.2002。
        if target_spec_ok:
            cand_spec = pipeline._MMAP_SPEC[row]
            if np.isfinite(cand_spec[0]) and np.abs(target_spec - cand_spec).max() > two_thr:
                continue
        # 本质类 birth 下界剪枝——严格无损：d0 >= max_i|b_i-b'_i|，
        # 该值已 >= 阈值则必被拒，可省掉昂贵的有限部分瓶颈计算。
        if (
            t_k0
            and float(np.abs(t_ess0 - ess_b0[row, :t_k0]).max())
            >= pipeline._MATCH_MMAP_THRESHOLD_H0
        ):
            continue
        # H1 本质类个数剪枝——严格无损（与 H0 同理）：dim1 本质类个数不同
        # ==> d1 = +inf ==> 必被拒。放在昂贵的 d0 实算之前，可省去一批 d0 计算（P2-6）。
        k1 = int(ess_cnt1[row])
        if k1 != t_k1:
            continue
        # 第 4 层（pivot）三角不等式下界剪枝——严格无损：
        # W_inf(A,B) >= |W_inf(A,P) - W_inf(B,P)|；若任一维度该下界 >= 阈值，
        # 则该维度真实距离 >= 阈值，候选必被 (d0<thr) AND (d1<thr) 拒。
        # history[dim][ordinal] 为预计算的「该候选到各 pivot」精确距离（按候选枚举序号 ordinal 建表），  # noqa: E501
        # 与 query 同口径；注意此处用 ordinal 而非全局行号 row（P0-1 索引修复）。
        if pipeline._MMAP_PIVOT_ROWS.size:
            pruned = False
            for dim in pipeline._MMAP_DIMS:
                _pd = pipeline._MMAP_PIVOT_DIST[dim]
                # P-1 防回归：ordinal 是候选枚举序号，也是 pivot 矩阵的列/行索引。
                # 越界立即报错，杜绝「把全局行号 row 误用回 pivot 矩阵」导致的静默误读。
                assert 0 <= ordinal < _pd.shape[0], (  # noqa: S101
                    f"pivot 行索引 ordinal={ordinal} 越界 [0,{_pd.shape[0]})：ordinal/row 错位"
                )
                history = _pd[ordinal]
                query = pivot_query[dim]
                thr = (
                    pipeline._MATCH_MMAP_THRESHOLD_H0
                    if dim == dim0
                    else pipeline._MATCH_MMAP_THRESHOLD_H1
                )
                # P0-3 / P5：NaN 安全化第 4 层下界剪枝（严格无损），逻辑抽至
                # _pivot_layer_accepts 以便单测四组合语义（pivot 簇已同模块，Phase 3 收敛）。
                if _pivot_layer_accepts(history, query, thr):
                    pruned = True
                    break
            if pruned:
                continue
        # 注意语义: candidate_total 统计的是"剪枝后实际计算距离的候选数"，
        # 而非"扫描到的候选总数"。被剪掉的候选已被证明不可能合格，故不计入。
        # 落库字段 match_runs.candidate_count 沿用此语义（仅供审计，不参与任何计算）。
        # WP4（可选，默认关闭）：持久度谱盒式下界早停。
        # lb = max_i |(d-b)/2(target_i) - (d-b)/2(cand_i)| <= W_inf(目标,候选)
        # （严格无损：lb >= 阈值 ⇒ 真实距离 >= 阈值 ⇒ 候选必被 (d0<thr) 拒）。
        # 仅 pipeline.ENABLE_LB_PRUNE 开启时生效；默认关闭，绝不改变既有默认路径行为。
        if pipeline.ENABLE_LB_PRUNE and target_spec_ok:
            _lb = 0.5 * float(np.abs(target_spec - pipeline._MMAP_SPEC[row]).max())
            if _lb >= pipeline._MATCH_MMAP_THRESHOLD_H0:
                pipeline._LB_PRUNE_COUNT += 1
                continue
        candidate_total += 1
        # WP3：候选有限点对走跨目标缓存（同候选被每个目标重复扫描）。
        # B4 follow-up（决策登记册）：退化候选（含非有限 birth-death 对）会让
        # _exact_bottleneck 经 finite_bottleneck_distance 抛 TopologyError。
        # 守卫下沉到候选粒度——仅跳过该候选并 LOGGER.warning 留痕，目标的其余
        # 有效候选照常产出，避免单候选异常拖累整个目标的有效匹配。
        t0_h, t1_h = _candidate_diagram(row)
        try:
            d0 = _exact_bottleneck(target0, t0_h, t_ess0, ess_b0[row, :t_k0])
            d1 = _exact_bottleneck(target1, t1_h, t_ess1, ess_b1[row, :k1])
        except TopologyError as exc:
            # 退化候选逐条打印会刷屏（候选规模数千~数万），改为计数，循环结束后按目标汇总一条。
            degenerate_skipped += 1
            continue
        if not d0 < pipeline._MATCH_MMAP_THRESHOLD_H0:
            continue
        if d1 < pipeline._MATCH_MMAP_THRESHOLD_H1:
            qualified.append((candidate_id, d0, d1))

    if degenerate_skipped:
        LOGGER.warning(
            "目标 %s 共跳过 %d 个退化持久图候选（不计入匹配），不影响其余匹配",
            target_id,
            degenerate_skipped,
        )
    qualified_count = len(qualified)
    # 排序：以 H1 距离(d1)为主键、H0 距离(d0)与候选 id 为次级键的稳定排序。
    # 注：曾尝试改成仅按 d1（对齐流程图"top-5 by H1 distance"），实测 49.35%→48.88%，
    # 并非两份报告差异的根因（根因是标签口径：notebook 样本内 vs topoquant 样本外），故回退。
    qualified.sort(key=lambda item: (item[2], item[1], item[0]))
    # 步骤 A：不再按 top_k 截断，返回完整候选排序（至安全上限），供落盘后按需截断复用。
    # 选中判定（合格候选数 >= top_k）延后到 match_clouds 落盘期按 config.top_k 决定，
    # 因此改 top_k 不会触发瓶颈距离重算。
    return target_id, candidate_total, qualified_count, qualified[: pipeline.MATCH_CANDIDATE_CAP]


def _load_done_target_ids(db) -> set[str]:
    """返回 match_runs 中已有结果（status 非空）的 target_id 集合，供断点续算跳过。

    稳健判空：`status IS NOT NULL AND status != ''`——尽管该列有 NOT NULL 约束，
    仍显式排除空字符串，兼容未来可能写入空状态的边界情况。
    """
    rows = db.execute(
        "SELECT target_id FROM match_runs WHERE status IS NOT NULL AND status != ''"
    ).fetchall()
    return {str(row["target_id"]) for row in rows}


def _persist_matches(  # noqa: PLR0913,PLR0917
    db: sqlite3.Connection,
    config: PipelineConfig,
    target_id: str,
    candidate_total: int,
    qualified_count: int,
    full_ranking: list[tuple[str, float, float]],
) -> None:
    """落盘单个匹配目标：完整候选排序 + 按当前 top_k 截断的 matches + match_runs。

    ``full_ranking`` 由 worker 产出、已按 (d1, d0, id) 稳定排序（至 ``MATCH_CANDIDATE_CAP``）。
    选中判定（合格候选数 >= top_k）在此按 ``config.top_k`` 决定，使得改 top_k 时仅截断口径不同、
    瓶颈距离无需重算（步骤 A）。``matches`` 仅存 top_k 条，下游 forecast/report 消费口径与
    改动前一致。
    """
    is_selected = qualified_count >= config.top_k
    top_matches = full_ranking[: config.top_k] if is_selected else []
    # 完整候选排序：供续算时按新 top_k 廉价重新截断（不重算距离）。
    save_match_candidates(db, target_id, full_ranking)
    save_matches(db, target_id, candidate_total, qualified_count, top_matches, is_selected)


def _reselect_completed_match_targets(
    db: sqlite3.Connection,
    config: PipelineConfig,
    done_target_ids: set[str],
) -> None:
    """续算时对已完成目标：不重算距离，仅按当前 top_k 从已存完整候选排序重新截断落盘。

    场景：仅改 top_k（步骤 A）或挪动 source_dir / 改无关拓扑参数（步骤 B，图内容不变）。
    既有的 ``match_candidates`` 仍有效，直接截断即可，瓶颈距离计算被完全跳过。
    为保留「同参数续跑不误删 forecasts」的既有行为：仅当目标已选中且现有 matches 恰好等于
    当前 top_k 时跳过（matches 即最优前缀，无需重写）；若现有 matches 少于（增量补全）或多于
    （缩减 top_k）当前 top_k，则按当前 top_k 重新截断落盘（缩减时旧 forecasts 由 forecast 阶段
    按新邻居集重算覆盖）。
    """
    if not done_target_ids:
        return
    # ── 分批查询（本次加固）───────────────────────────────────────────
    # 沿用 _export_diagrams_to_mmap 里已经确立的稳健口径：SQLite 对单条语句的宿主
    # 变量数有上限（SQLITE_LIMIT_VARIABLE_NUMBER，旧版本 999，3.32+ 默认 32766），
    # 而 done_target_ids 的规模随目标数线性增长（本项目当前 4970，更大数据集会继续涨）。
    # 把全部 id 塞进单个 IN (...) 迟早会抛 OperationalError: too many SQL variables，
    # 而且它恰好发生在「续算入口」——本该最省事的续跑反而会整轮失败。
    # 按固定安全批大小切分后，单条语句的变量数恒定，与运行规模彻底解耦。
    #
    # 正确性说明：切分只按 target_id 进行，因此同一个 target 的全部 match_candidates
    # 必定落在同一批内；批内依旧 ORDER BY target_id, rank，所以下面 by_target 中每个
    # target 的候选顺序与不分批时**完全一致**（下游只依赖「同一 target 内 rank 递增」）。
    # 这里只替换取数方式并把结果拼回同名列表，下游聚合与落盘逻辑保持原样不动。
    ordered_ids = list(done_target_ids)
    _chunk = 900
    cand_rows: list[sqlite3.Row] = []
    run_rows: list[sqlite3.Row] = []
    for _start in range(0, len(ordered_ids), _chunk):
        batch = ordered_ids[_start : _start + _chunk]
        placeholders = ",".join("?" for _ in batch)
        cand_rows.extend(
            db.execute(
                f"SELECT target_id, similar_id, rank, distance_dim0, distance_dim1 "
                f"FROM match_candidates WHERE target_id IN ({placeholders}) "
                f"ORDER BY target_id, rank",
                batch,
            ).fetchall()
        )
        run_rows.extend(
            db.execute(
                f"SELECT target_id, candidate_count, qualified_count FROM match_runs "
                f"WHERE target_id IN ({placeholders})",
                batch,
            ).fetchall()
        )
    by_target: dict[str, list[tuple[str, float, float]]] = {}
    for row in cand_rows:
        by_target.setdefault(str(row["target_id"]), []).append(
            (str(row["similar_id"]), float(row["distance_dim0"]), float(row["distance_dim1"]))
        )
    qualified = {
        str(row["target_id"]): (int(row["candidate_count"]), int(row["qualified_count"]))
        for row in run_rows
    }
    with CommitBatch(
        db, every=POLICY.commit_every_reselect, what="reselect 分批落盘"
    ) as _reselect_batch:
        _written = 0
        for target_id in done_target_ids:
            candidate_count, qc = qualified.get(target_id, (0, 0))
            is_selected = qc >= config.top_k
            if is_selected:
                existing = db.execute(
                    "SELECT COUNT(*) AS c FROM matches WHERE target_id = ?", (target_id,)
                ).fetchone()["c"]
                # 恰好等于当前 top_k → 最优前缀，无需重写，保留既有 forecasts。
                # 大于（缩减 top_k）或小于（增量补全）都必须按当前 top_k 重新截断落盘，
                # 否则下游 forecast 的 len(similar_ids)==top_k 断言会因 matches 行数不符而失败。
                if existing == config.top_k:
                    continue
            ranking = by_target.get(target_id, [])
            top_matches = ranking[: config.top_k] if is_selected else []
            save_matches(db, target_id, candidate_count, qc, top_matches, is_selected)
            _written += 1
            # 分批提交：避免把全部重写塞进单一大事务（WAL 长时间持有、崩溃回滚面过大）。
            # 提交节奏收敛到 CommitBatch（每 commit_every_reselect 项一次 + 退出时尾批补提交），
            # 与 _consume_target 的每 100 项提交同一意图——把事务粒度控制在可控范围，
            # 崩溃时仅回滚最后一批而非全部 reselect。
            _reselect_batch.tick()


# 候选记录：cloud_id + 全局 mmap 行号。
# 契约（P-2 根因消除）：pivot 距离矩阵按「候选枚举序号 ordinal」建表（见 _ensure_pivot_cache），
# 与全局 mmap 行号 row 是两套索引。凡访问 pivot 矩阵必须用 ordinal，绝不可用 row。
# Candidate 同时是 2-元组，循环体 `for ordinal, (cid, row) in enumerate(_MMAP_CANDIDATES)` 兼容不变。  # noqa: E501
Candidate = namedtuple("Candidate", ["cid", "row"])


def match_clouds(  # noqa: PLR0915
    config: PipelineConfig,
    progress: ProgressCallback | None = None,
    log_queue=None,
) -> dict[str, int]:
    set_topology_backend(config.topology_backend)
    _clear_incomplete_marker(config.work_dir, "match")
    """瓶颈匹配（mmap 零拷贝版本）。

    所有 worker 共享同一份只读内存映射的持久图，进程初始化参数只有文件路径和
    整数索引，因此总内存占用与并发数无关。
    """
    dimensions = tuple(config.distance_dimensions)
    dim0, dim1 = dimensions

    # 透传 progress：等待跨进程导出锁（export_wait）与实际导出（export）都会
    # 实时上报，避免匹配阶段一开始就长时间静默、与挂死无法区分。
    cloud_id_to_row = _ensure_diagram_mmap(config, progress)
    count_path = config.work_dir / DIAGRAM_COUNTS_FILENAME
    h0_path = config.work_dir / f"diagrams_h{dim0}.npy"
    h1_path = config.work_dir / f"diagrams_h{dim1}.npy"
    # 计数表只有几百 KB，父进程整份读入即可；不保留 mmap 句柄，
    # 以免在 Windows 上妨碍后续 build 阶段重写同名文件。
    try:
        counts = np.load(str(count_path))
    except (ValueError, OSError) as _exc:
        raise DataError(f"diagram_counts 读取失败（可能文件被截断）：{count_path}") from _exc

    def usable(cloud_id: str) -> bool:
        row = cloud_id_to_row.get(cloud_id)
        if row is None:
            return False
        return int(counts[row, dim0]) > 0 and int(counts[row, dim1]) > 0

    with closing(connect(config.database_path)) as db:
        _progress(progress, "load_clouds", 0, 1, {"status": "读取点云元数据"})
        records = load_cloud_records(db)
        _progress(
            progress,
            "load_clouds",
            1,
            1,
            {"status": f"已读 {len(records)} 个点云"},
        )
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

        # ── 断点续算：跳过已完成匹配的目标（P1-2）──
        # 续算键改为「图内容签名 + 距离参数」（matching_resume_key，步骤 A+B）：
        #   · 图内容不变（挪 source_dir / 改不影响匹配输入的拓扑参数 / 仅改 top_k）→ 键不变 → 复用；
        #   · 图内容或距离参数变化（含 as_of_date 滑动导致图内容变化）→ 键变化 → 全量重算。
        # 键一致时：已完成目标不再重算瓶颈距离，仅按当前 top_k 从已存完整候选排序
        # （match_candidates）重新截断落盘（_reselect_completed_match_targets），实现「改 top_k
        # 只改截断、不重算距离」的复用（步骤 A）。键不一致或首次运行：重跑全部目标，
        # 由 save_matches 的 DELETE+INSERT 幂等覆盖，不留陈旧行。
        # 续算状态唯一权威来源为 ``stage_runs``（get_stage_status 仅读该表；旧库经
        # _upgrade_v2_to_v3 在 connect 时回填，故不再依赖 legacy matching_signature）。
        # 运行中硬杀残留的 running 状态，由 ``status == "running"`` 识别为「可续」，
        # 斩断「硬杀→签名中毒→反复冷重算」死循环；下游 forecast 亦同步对齐该口径。
        status, stored_signature = get_stage_status(db, "matching")
        current_signature = matching_resume_key(config)
        if status is not None and stored_signature != current_signature:
            LOGGER.warning(
                "匹配输入（图内容+距离参数）已变化（已存=%s / 当前=%s），既有匹配结果不可信，将重新计算全部目标",  # noqa: E501
                stored_signature,
                current_signature,
            )
        resume_ok = bool(stored_signature) and stored_signature == current_signature
        if resume_ok:
            done_target_ids = _load_done_target_ids(db)
            if done_target_ids:
                LOGGER.info(
                    "复用既有距离结果，跳过 %d 个已完成目标的瓶颈距离计算（按当前 top_k 重新截断落盘）",  # noqa: E501
                    len(done_target_ids),
                )
                _reselect_completed_match_targets(db, config, done_target_ids)
                target_ids = [tid for tid in target_ids if tid not in done_target_ids]
            if status == "running":
                LOGGER.info("检测到上一轮中断残留的 running 状态，将从已完成目标续算而非全量重算")

        target_index = {cloud_id: cloud_id_to_row[cloud_id] for cloud_id in target_ids}
        candidates = tuple(
            Candidate(cloud_id, cloud_id_to_row[cloud_id]) for cloud_id in candidate_ids
        )
        # 第 4 层（pivot）剪枝的预计算缓存（代表候选 × 全部候选精确距离），一次性成本。
        # _ensure_pivot_cache 已随 pivot 簇迁入本模块（Phase 3 收敛），直接调用。
        candidate_rows = np.asarray([row for _, row in candidates], dtype=np.int64)
        pivot_rows_path, pivot_distance_paths = _ensure_pivot_cache(
            config,
            dimensions,
            tuple(candidate_ids),
            candidate_rows,
            h0_path,
            h1_path,
            count_path,
            progress=progress,
            log_queue=log_queue,
        )

        # R1c 收敛收尾：续算状态仅落 stage_runs（不再双写 legacy matching_signature）。
        # 旧库兼容由 _upgrade_v2_to_v3 回填保证，测试已改为断言 stage_runs 行。
        _commit_with_retry(db, what="匹配阶段起始标记")
        _matching_started_at = dt.datetime.now(dt.timezone.utc).isoformat()
        upsert_stage_run(
            db,
            stage="matching",
            status="running",
            signature=current_signature,
            pid=os.getpid(),
            host=socket.gethostname(),
            started_at=_matching_started_at,
            updated_at=_matching_started_at,
        )

        worker_count = config.resolved_matching_workers
        _progress(
            progress,
            "matching",
            0,
            len(target_ids),
            {"selected": 0, "workers": worker_count},
        )

        # initargs 只含路径、整数索引和标量参数，spawn 时的传输量以 MB 计而非 GB。
        # 关于 candidates：每个 worker 都需要扫描完整候选集（任一目标可能匹配任意候选），
        # 因此候选列表必须随 initializer 下发，无法按 worker 切分。
        #   · fork 启动方式（Linux/macOS）：candidates 经由写时复制(COW)继承，不重复占内存；
        #   · spawn 启动方式（Windows 默认）：每个子进程各 pickle 一次，但 payload 仅为
        #     (cloud_id, 行号) 整数对，体量极小，开销可忽略（P3-8）。
        # 注意：top_k 已不在 initargs 中（步骤 A）——距离计算与 top_k 无关。
        initargs = (
            str(h0_path),
            str(h1_path),
            str(count_path),
            target_index,
            candidates,
            dimensions,
            config.distance_threshold_h0,
            config.distance_threshold_h1,
            str(pivot_rows_path),
            {dim: str(pivot_distance_paths[dim]) for dim in dimensions},
            log_queue,
        )
        selected = 0
        errors = 0

        def _consume_target(index: int, result: object) -> None:
            """原 ``for index, result in enumerate(results, 1):`` 的整段循环体（逐字保留）。"""
            nonlocal selected
            target_id, candidate_total, qualified_count, full_ranking = result
            # 落盘：完整候选排序（match_candidates）+ 按当前 top_k 截断的 matches；
            # 选中判定延迟到此处按 config.top_k 决定（步骤 A）。
            _persist_matches(db, config, target_id, candidate_total, qualified_count, full_ranking)
            selected += 1 if qualified_count >= config.top_k else 0
            # R2-S3：提交节奏收敛到 CommitBatch（每 commit_every_matching 项一次 + 退出时尾批补提交），  # noqa: E501
            # 末项必落盘由 __exit__ flush 兜底；锁争抢自愈语义不变。
            _commit_batch.tick()
            # 进度日志仍按原节奏（每 100 项 + 末项）输出；已有 rich 进度条降级 DEBUG，
            # 无进度条（headless / --json）保留 INFO。
            if index % POLICY.commit_every_matching == 0 or index == len(target_ids):
                if progress is None:
                    LOGGER.info("匹配进度：%d/%d，已选 %d", index, len(target_ids), selected)
                else:
                    LOGGER.debug("匹配进度：%d/%d，已选 %d", index, len(target_ids), selected)
            _progress(
                progress,
                "matching",
                index,
                len(target_ids),
                {"selected": selected, "workers": worker_count},
            )

        def _fail_target(index: int, label: str, reason: str, timed_out: bool) -> None:
            """超时/异常导致某个目标没有结果：记错 + 日志 + 推进进度，不写 match_runs。

            不落 match_runs 是刻意的——重跑 match 时 ``_load_done_target_ids`` 查不到它，
            该目标会被自动重算，断点续跑语义保持幂等。
            """
            nonlocal errors
            errors += 1
            LOGGER.error("目标 %s 未产出匹配结果（%s），已记为错误并继续后续目标", label, reason)
            _progress(
                progress,
                "matching",
                index,
                len(target_ids),
                {"selected": selected, "workers": worker_count},
            )

        # ── 完成标记落地保障（本次修复）──────────────────────────────────
        # 原实现把「签名改成完成态」放在顺利跑完之后的直线代码里。一旦中途抛异常
        # （实测现场即 _consume_target 中的 commit 撞上 database is locked），
        # 这一行永远不会执行，matching_signature 便永久停留在 "in_progress:<key>"。
        # 后果是双重的，且互相放大：
        #   1) 已算完并落库的 match_runs（现场 2160 条）全部无法被续算复用——
        #      下次运行 stored != current，只能从零重算全部 4970 个目标；
        #   2) forecast 阶段的守卫 `stored != matching_resume_key(config)` 必然成立，
        #      直接抛 DataError，整条流水线被永久卡在匹配阶段，反复重算、反复失败。
        # 因此用 try/finally 把完成标记钉死：正常结束或异常退出都落成完成态。
        # 这是安全且幂等的——失败/未跑的目标没有 match_runs 行，
        # _load_done_target_ids 查不到它们，重跑 match 只会补算这部分。
        # 这正是 _fail_target 文档里既有的设计意图，此处只是让异常路径同样遵守。
        with CommitBatch(
            db, every=POLICY.commit_every_matching, what="匹配中间结果"
        ) as _commit_batch:
            try:
                # 模式 E-2/3：单/多 worker 统一走 _run_consumer，消除分支隔离漂移；
                # 单 worker 也包 try/except，任一目标异常记 _fail_target 后继续，不再穿透整阶段。
                _run_consumer(
                    stage_label="瓶颈匹配",
                    worker_count=worker_count,
                    initializer=_init_match_worker_mmap,
                    initargs=initargs,
                    worker_fn=_match_one_mmap,
                    items=list(target_ids),
                    labels=[str(target_id) for target_id in target_ids],
                    consume=_consume_target,
                    fail_handler=_fail_target,
                )
                if errors:
                    # 签名照常落成完成态：失败目标没有 match_runs 行，重跑 match 会自动重试它们，
                    # 同时 forecast 不会因为个别目标挂死而被整体拦下（这正是本次修复的目的）。
                    LOGGER.warning(
                        "匹配阶段有 %d/%d 个目标超时或异常未完成；重新执行 match 可自动重试这些目标",  # noqa: E501
                        errors,
                        len(target_ids),
                    )
            finally:
                try:
                    upsert_stage_run(
                        db,
                        stage="matching",
                        status="completed",
                        signature=current_signature,
                        pid=os.getpid(),
                        host=socket.gethostname(),
                        started_at=_matching_started_at,
                        updated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
                    )
                    _commit_with_retry(db, what="匹配阶段完成标记")
                except Exception as exc:
                    # 连完成标记都写不进去，说明库已不可写。这里只记录，绝不 raise：
                    # 否则会用这个二次异常替换掉正在传播的真正首因，让排查失去线索。
                    LOGGER.error(
                        "写入匹配完成标记失败：%s（已算出的结果仍在库中，"
                        "下次运行会因签名未落地而重算全部目标）",
                        exc,
                    )
                    # 显式落一个告警标记，使「库不可写」这一严重状态可被监控/运维捕获
                    # （仅写文件系统，不依赖 DB；失败则忽略，不影响主流程）。
                    with suppress(Exception):
                        _write_incomplete_marker(config.work_dir, "match", exc)
        return {
            "targets": len(target_ids),
            "candidates": len(candidate_ids),
            "selected": selected,
            "errors": errors,
            "workers": worker_count,
        }


# 融合升级：瓶颈匹配第 4 层（pivot）剪枝。
# 选 matching_pivots 个代表候选（pivot），预计算每个候选到它们的精确瓶颈距离；
# 匹配时利用三角不等式下界 |W(A,B)-W(B,P)| <= W(A,B) 提前剪掉必不合格候选，
# 该下界严格无损（被剪候选的真实距离必 >= 阈值）。pivot 数为 0 时整层禁用。
PIVOT_ROWS_FILENAME = "match_pivot_rows.npy"
PIVOT_SIGNATURE_FILENAME = "match_pivot_signature.txt"
# pivot 距离缓存单线程重计算期间的最小进度刷新间隔（秒）。
# 列内逐次回调会让进度渲染本身成为新瓶颈，故采用「分块 map + 时间节流」：
# 每列约 64 次刷新、且任意两次刷新间隔 >= 该值，兼顾可见性与零拷贝计算性能。
PIVOT_PROGRESS_MIN_INTERVAL = 0.2


def _pivot_paths(
    work_dir: Path,
    dimensions: tuple[int, int],
) -> tuple[Path, Path, dict[int, Path]]:
    """pivot 缓存文件路径：签名、pivot 行号、各维度「候选×pivot」精确距离矩阵。"""
    return (
        work_dir / PIVOT_SIGNATURE_FILENAME,
        work_dir / PIVOT_ROWS_FILENAME,
        {dim: work_dir / f"match_pivot_distance_h{dim}.npy" for dim in dimensions},
    )


def _exact_distances_between_rows(rows: tuple[int, int]) -> tuple[float, float]:
    """两行持久图之间的精确瓶颈距离 (d0, d1)，复用已打开的 mmap 全局句柄。

    与 ``_match_one_mmap`` 内联的实算完全一致（同样走 ``_exact_bottleneck``），
    保证 pivot 缓存与在线匹配的距离口径逐位一致——这是「无损」的前提。
    """
    from . import pipeline  # _MMAP_* 驻留 pipeline.py（单一真相源，避免别名分裂）

    left_row, right_row = rows
    dim0, dim1 = pipeline._MMAP_DIMS
    counts = _mmap_handle(pipeline._MMAP_COUNT_PATH)
    block0 = _mmap_handle(pipeline._MMAP_PATHS[dim0])
    block1 = _mmap_handle(pipeline._MMAP_PATHS[dim1])
    c0l = int(counts[left_row, dim0])
    c1l = int(counts[left_row, dim1])
    c0r = int(counts[right_row, dim0])
    c1r = int(counts[right_row, dim1])
    el0 = pipeline._MMAP_ESS_B[dim0][left_row, : int(pipeline._MMAP_ESS_CNT[dim0][left_row])]
    er0 = pipeline._MMAP_ESS_B[dim0][right_row, : int(pipeline._MMAP_ESS_CNT[dim0][right_row])]
    el1 = pipeline._MMAP_ESS_B[dim1][left_row, : int(pipeline._MMAP_ESS_CNT[dim1][left_row])]
    er1 = pipeline._MMAP_ESS_B[dim1][right_row, : int(pipeline._MMAP_ESS_CNT[dim1][right_row])]
    d0 = _exact_bottleneck(
        _mmap_pairs(block0, left_row, c0l),
        _mmap_pairs(block0, right_row, c0r),
        el0,
        er0,
    )
    d1 = _exact_bottleneck(
        _mmap_pairs(block1, left_row, c1l),
        _mmap_pairs(block1, right_row, c1r),
        el1,
        er1,
    )
    return d0, d1


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


def _select_next_pivot(coverage: np.ndarray) -> int:
    """P2 退化修复：把 inf 视为"落后"(-1)，避免 argmax 永远优先选中 inf 候选。

    与原内联三行逐位等价：``np.isinf`` 同时覆盖 ``+inf``（尚未被任何 pivot 覆盖）
    与 ``-inf``（已被选为 pivot，见 ``coverage[selected] = -np.inf``），两者都降为 -1.0；
    ``coverage`` 本身不被修改（先 ``copy()``）。
    """
    pick = coverage.copy()
    pick[np.isinf(pick)] = -1.0
    return int(np.argmax(pick))


# ────────────────────────────────────────────────────────────────────────
# WP1：pivot 距离缓存并行化（方案 A+B）——新增 worker，不改变任何既有口径
# ────────────────────────────────────────────────────────────────────────


def _init_pivot_worker(  # noqa: PLR0913,PLR0917
    h0_path: str,
    h1_path: str,
    count_path: str,
    dimensions: tuple[int, int],
    thr_h0: float,
    thr_h1: float,
    log_queue=None,
) -> None:
    """pivot 预计算 worker 初始化：复用 :func:`_init_match_worker_mmap` 建好 mmap 全局句柄。

    语义与 ``_ensure_pivot_cache`` 内既有的空候选调用完全一致——只设全局句柄，
    不传入真实目标/候选索引，也不加载 pivot 距离矩阵（``pivot_rows_path`` 取默认空串）。
    新增函数，不替换任何既有符号。
    """
    # log_queue 必须走关键字参数：_init_match_worker_mmap 签名第 9 个位置形参是
    # pivot_rows_path（默认空串），若把 log_queue 当第 9 个位置参数传入，会被误当作
    # pivot_rows_path 路径，触发 np.load(Queue) 的 TypeError。关键字传参可确保
    # pivot_rows_path 取默认空串（不加载 pivot 距离矩阵），log_queue 正确接入日志。
    _init_match_worker_mmap(
        h0_path, h1_path, count_path, {}, (), dimensions, thr_h0, thr_h1,
        log_queue=log_queue,
    )


def _pivot_column_distances(task):
    """pivot 缓存单 worker 任务：固定 pivot 的有限点**算一次**，再循环候选复用（去重拷贝）。

    与 :func:`_exact_distances_between_rows` 同口径（同样走 ``_exact_bottleneck`` +
    ``_mmap_pairs`` + ``_MMAP_ESS_*``），故结果逐位一致、无损。``task`` 为
    ``(column, pivot_row, ordinals, cand_global_rows)``：``ordinals`` 是候选枚举序号
    （= 结果回写 ``values`` 的行索引），``cand_global_rows`` 是对应的持久图全局行号。
    """
    from . import pipeline  # _MMAP_* 驻留 pipeline.py（单一真相源，避免别名分裂）

    column, pivot_row, ordinals, cand_rows = task
    dim0, dim1 = pipeline._MMAP_DIMS
    counts = _mmap_handle(pipeline._MMAP_COUNT_PATH)
    block0 = _mmap_handle(pipeline._MMAP_PATHS[dim0])
    block1 = _mmap_handle(pipeline._MMAP_PATHS[dim1])
    c0p = int(counts[pivot_row, dim0])
    c1p = int(counts[pivot_row, dim1])
    # 去重拷贝：pivot 有限点每列只算一次（原单线程版本每个候选都重算一遍）。
    p_h0 = _mmap_pairs(block0, pivot_row, c0p)
    p_h1 = _mmap_pairs(block1, pivot_row, c1p)
    p_e0 = pipeline._MMAP_ESS_B[dim0][pivot_row, : int(pipeline._MMAP_ESS_CNT[dim0][pivot_row])]
    p_e1 = pipeline._MMAP_ESS_B[dim1][pivot_row, : int(pipeline._MMAP_ESS_CNT[dim1][pivot_row])]
    out = np.empty((len(ordinals), 2), dtype=np.float64)
    for i, _o in enumerate(ordinals):
        cr = int(cand_rows[i])
        c0 = int(counts[cr, dim0])
        c1 = int(counts[cr, dim1])
        out[i, 0] = _exact_bottleneck(
            p_h0,
            _mmap_pairs(block0, cr, c0),
            p_e0,
            pipeline._MMAP_ESS_B[dim0][cr, : int(pipeline._MMAP_ESS_CNT[dim0][cr])],
        )
        out[i, 1] = _exact_bottleneck(
            p_h1,
            _mmap_pairs(block1, cr, c1),
            p_e1,
            pipeline._MMAP_ESS_B[dim1][cr, : int(pipeline._MMAP_ESS_CNT[dim1][cr])],
        )
    return column, np.asarray(ordinals, dtype=np.int64), out


def _pivot_layer_accepts(history: np.ndarray, query: np.ndarray, thr: float) -> bool:
    """第 4 层 pivot 三角不等式下界剪枝的接受判定（严格无损、NaN/inf 安全，可单测）。

    ``history`` / ``query`` 为某维「候选到各 pivot」与「目标到各 pivot」的精确距离向量（等长）。
    返回 ``True`` 表示该维下界 >= 阈值、候选必被拒（剪枝）。
    语义（P0-3）：有限↔有限→正常比；inf↔有限（下界 inf）→ 合法剪枝；
    inf↔inf（无信息）→ 视为 0 → 不剪。
    """
    # inf - inf 在数学上为 nan，属预期的「无信息」情况，此处静默该告警。
    with np.errstate(invalid="ignore"):
        diff = np.abs(np.asarray(history, dtype=float) - np.asarray(query, dtype=float))
    return bool(np.where(np.isnan(diff), 0.0, diff).max() >= thr)


def _ensure_pivot_cache(  # noqa: PLR0913,PLR0915,PLR0917
    config: PipelineConfig,
    dimensions: tuple[int, int],
    candidate_ids: tuple[str, ...],
    candidate_rows: np.ndarray,
    h0_path: Path,
    h1_path: Path,
    count_path: Path,
    progress: ProgressCallback | None = None,
    log_queue=None,
) -> tuple[Path, dict[int, Path]]:
    """构建可复用的「候选 × pivot」精确瓶颈距离矩阵（一次性成本，结果落盘缓存）。

    选取代表候选（pivot）后用贪心覆盖法：首个 pivot 取 0 号候选，其后每个 pivot
    取「到已有 pivot 覆盖最差（两维距离最大值最大）」的候选，使少量 pivot 覆盖尽可能
    多的候选空间。距离矩阵 shape=(n_candidates, pivot_count)，是两维精确瓶颈距离。

    **可见性（本次新增）**：本函数是匹配阶段中排在进度条之前的一段重计算——
    每个 pivot 都要单线程遍历全部候选算精确瓶颈距离（本项目规模为 8 × 35208 次），
    原先只在每列算完后打一条 ``LOGGER.info``，列与列之间可能数十秒毫无输出，
    与挂死无法区分。这里透传 ``progress``：等锁阶段上报 ``pivot_wait``，
    列内重计算以「分块 map + 时间节流」细粒度上报 ``pivot``（总量 =
    pivot_count × n_candidates，随每列逐块平滑推进，每列约 64 次刷新、
    任意两次刷新间隔 >= ``PIVOT_PROGRESS_MIN_INTERVAL``）。热路径仍用
    ``list(map(...))`` 批量计算，仅在分块边界做节流上报，进度渲染本身不构成瓶颈。
    """
    signature_path, rows_path, distance_paths = _pivot_paths(config.work_dir, dimensions)
    pivot_count = min(config.matching_pivots, len(candidate_ids))
    with _export_lock(config.work_dir, progress=progress, stage="pivot_wait"):
        if _pivot_cache_is_current(config, dimensions, candidate_ids, pivot_count):
            LOGGER.info("pivot 距离缓存仍然有效：%d 个 pivot", pivot_count)
            return rows_path, distance_paths

        stage = Path(tempfile.mkdtemp(prefix="_pivot_stage_", dir=str(config.work_dir)))
        matrices = {
            dim: np.empty((len(candidate_ids), pivot_count), dtype=np.float64) for dim in dimensions
        }
        pivot_rows = np.empty(pivot_count, dtype=np.int64)
        try:
            if pivot_count:
                # 复用匹配 worker 的 mmap 初始化（设置 _MMAP_* 全局），供
                # _exact_distances_between_rows 读取持久图计算行对距离。
                # 这里仅用其全局句柄，不传入真实目标/候选索引。
                _init_match_worker_mmap(
                    str(h0_path),
                    str(h1_path),
                    str(count_path),
                    {},
                    (),
                    dimensions,
                    config.distance_threshold_h0,
                    config.distance_threshold_h1,
                )
                coverage = np.full(len(candidate_ids), np.inf, dtype=np.float64)
                selected = np.zeros(len(candidate_ids), dtype=bool)
                next_index = 0
                # 开工即上报一帧：让「正在算 pivot」立刻出现在进度条上，
                # 而不是等第一列（可能数十秒）算完才有任何反馈。
                # 列内单线程重计算以往数十秒毫无输出、与挂死无法区分；
                # 现改为「分块 map + 时间节流」的细粒度进度：总量 =
                # pivot_count × n_candidates，随每列逐块推进平滑增长。
                n_candidates = len(candidate_ids)
                total_items = pivot_count * n_candidates
                done_items = 0
                _last_report = time.monotonic()
                _pivot_chunk = max(1, n_candidates // 64)  # 每列约 64 次刷新
                _progress(
                    progress,
                    "pivot",
                    0,
                    total_items,
                    {"pivots": pivot_count, "candidates": n_candidates},
                )
                # WP1：pivot 预计算并行化（方案 A+B）。
                # 列级「选 pivot → 聚合 coverage/selected → 选下一 pivot」保持单线程逐字不变
                # （pivot 选择是数据依赖的贪心，必须按列顺序）；仅把每列内的
                # 「候选×pivot 精确距离」下放到进程池，且 pivot 有限点每列只拷贝一次（去重拷贝）。
                workers = max(1, config.resolved_matching_workers)
                for column in range(pivot_count):
                    pivot_row = int(candidate_rows[next_index])
                    pivot_rows[column] = pivot_row
                    values = np.empty((n_candidates, 2), dtype=np.float64)
                    # 任务粒度：每列拆成 >= workers*4 个子块，使进程池本列内即被喂满
                    # （列间顺序依赖，必须每列内部达到 >= workers 并发，而非跨列累计），
                    # 避免「16 核被 8 个 pivot 列限制成 8 路」。
                    _num_chunks = max(1, workers * 4, math.ceil(n_candidates / 1024))
                    _chunk = max(1, math.ceil(n_candidates / _num_chunks))
                    _tasks = [
                        (
                            column,
                            int(pivot_row),
                            list(range(_s, min(_s + _chunk, n_candidates))),
                            [
                                int(candidate_rows[_o])
                                for _o in range(_s, min(_s + _chunk, n_candidates))
                            ],
                        )
                        for _s in range(0, n_candidates, _chunk)
                    ]
                    _chunk_labels = [
                        f"pivot 列{column + 1} 块{chunk_idx + 1}/{len(_tasks)}"
                        for chunk_idx in range(len(_tasks))
                    ]

                    def _consume_chunk(
                        _position,
                        result,
                        *,
                        _values=values,
                        _col_no=column,
                    ):
                        # 顺序无关：按候选下标 _ords 回填，与任务执行顺序无关 → 数值逐位一致。
                        _col, _ords, _out = result
                        _values[_ords, 0] = _out[:, 0]
                        _values[_ords, 1] = _out[:, 1]
                        nonlocal done_items, _last_report
                        done_items += len(_ords)
                        _now = time.monotonic()
                        if (
                            _now - _last_report >= PIVOT_PROGRESS_MIN_INTERVAL
                            or done_items >= total_items
                        ):
                            _last_report = _now
                            _progress(
                                progress,
                                "pivot",
                                done_items,
                                total_items,
                                {
                                    "pivots": pivot_count,
                                    "candidates": n_candidates,
                                    "col": _col_no + 1,
                                },
                            )

                    def _fallback_chunk(
                        position,
                        label,
                        reason,
                        timed_out,
                        *,
                        _tasks=_tasks,
                        _values=values,
                        _col_no=column,
                    ):
                        # 某块超时/异常（如 GUDHI C 挂死被看门狗强杀）：父进程同步补算该块，
                        # 父进程 mmap 句柄已在上方 _init_match_worker_mmap 设好，结果逐位一致。
                        # 这样既不在挂死时无限等待（恢复），也不会留下未写孔的损坏矩阵（正确）。
                        LOGGER.warning(
                            "pivot 缓存某块失败（%s，超时=%s），父进程同步补算：%s",
                            reason,
                            timed_out,
                            label,
                        )
                        _col, _ords, _out = _pivot_column_distances(_tasks[position - 1])
                        _values[_ords, 0] = _out[:, 0]
                        _values[_ords, 1] = _out[:, 1]
                        nonlocal done_items, _last_report
                        done_items += len(_ords)
                        _now = time.monotonic()
                        if (
                            _now - _last_report >= PIVOT_PROGRESS_MIN_INTERVAL
                            or done_items >= total_items
                        ):
                            _last_report = _now
                            _progress(
                                progress,
                                "pivot",
                                done_items,
                                total_items,
                                {
                                    "pivots": pivot_count,
                                    "candidates": n_candidates,
                                    "col": _col_no + 1,
                                },
                            )

                    # 复用 _run_pool_stage 的「per-task 超时 + 强杀子进程 + 重建池」，
                    # 不自造裸 mp.Pool（模式 E-1）。每列独立调用：列间贪心选择有数据依赖，
                    # 必须逐列串行；_run_pool_stage 在 finally 中 terminate 池，不泄漏。
                    # task_timeout 沿用 STAGE_TASK_TIMEOUT（默认 1800s）：pivot 单块通常远快于
                    # 此值，此处仅作为 GUDHI C 挂死的兜底恢复上限（模式 E-4：文档化超时）。
                    _run_pool_stage(
                        stage_label="pivot 缓存",
                        worker_count=workers,
                        initializer=_init_pivot_worker,
                        initargs=(
                            str(h0_path),
                            str(h1_path),
                            str(count_path),
                            dimensions,
                            config.distance_threshold_h0,
                            config.distance_threshold_h1,
                            log_queue,
                        ),
                        task_func=_pivot_column_distances,
                        items=_tasks,
                        labels=_chunk_labels,
                        consume=_consume_chunk,
                        on_failure=_fallback_chunk,
                        task_timeout=STAGE_TASK_TIMEOUT,
                    )
                    matrices[dimensions[0]][:, column] = values[:, 0]
                    matrices[dimensions[1]][:, column] = values[:, 1]
                    combined = np.maximum(values[:, 0], values[:, 1])
                    coverage = np.minimum(coverage, combined)
                    selected[next_index] = True
                    coverage[selected] = -np.inf
                    if column + 1 < pivot_count:
                        # 选「到最近 pivot 距离最大」的候选作为下一个 pivot，
                        # 即覆盖最差者，使少量 pivot 覆盖尽可能大的候选空间。
                        # P2 退化修复：覆盖准则含 inf，argmax 会永远优先选中 inf 候选
                        # （约半数 pivot 无增量覆盖价值）。把 inf 视为"落后"(-1)避免反复选中。
                        next_index = _select_next_pivot(coverage)
                    LOGGER.info("pivot 缓存进度：%d/%d", column + 1, pivot_count)

            np.save(str(stage / rows_path.name), pivot_rows)
            for dim in dimensions:
                np.save(str(stage / distance_paths[dim].name), matrices[dim])
            _replace_atomic(str(stage / rows_path.name), str(rows_path))
            for dim in dimensions:
                _replace_atomic(str(stage / distance_paths[dim].name), str(distance_paths[dim]))
            staged_signature = stage / signature_path.name
            staged_signature.write_text(
                _pivot_signature(config, dimensions, candidate_ids, pivot_count),
                encoding="utf-8",
            )
            _replace_atomic(str(staged_signature), str(signature_path))
        finally:
            shutil.rmtree(stage, ignore_errors=True)
    return rows_path, distance_paths
