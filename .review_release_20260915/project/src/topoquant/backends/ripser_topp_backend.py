from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .base import Diagram, TopologyError, as_finite_pairs

# 本后端 = Ripser 算持久图 + Topp 1.0.0 算精确瓶颈距离（高速后端）。
# Topp 是同仓库 bottleneck_distance_banckend 的 Python 绑定（C++ 内核 + pybind11）。
name = "ripser_topp"

# Topp 公开异常类型（见 Topp/docs/API.md）。
_TOPP_EXCEPTIONS = (TypeError, ValueError, NotImplementedError, MemoryError, OverflowError, RuntimeError)


@dataclass(frozen=True)
class ToppBackendInfo:
    """Topp 高速后端的可复现信息（用于启动日志 / 后端初始化校验）。"""

    name: str
    version: str
    source: str


def compute_persistence(
    points: np.ndarray,
    max_edge_length: float,
    max_homology_dimension: int,
) -> Diagram:
    """用 Ripser 计算持续同调（搬运 BRANCH 实现）。"""
    try:
        from ripser import ripser
    except ImportError as exc:
        raise TopologyError("缺少 Ripser 后端，请安装依赖：pip install ripser>=0.6.15") from exc
    try:
        diagrams = ripser(
            np.asarray(points, dtype=np.float64),
            maxdim=max_homology_dimension,
            thresh=max_edge_length,
        )["dgms"]
    except Exception as exc:
        raise TopologyError(f"Ripser 持续同调计算失败：{exc}") from exc
    return {
        dimension: np.ascontiguousarray(diagrams[dimension], dtype=np.float64).reshape(-1, 2)
        for dimension in range(max_homology_dimension + 1)
    }


def finite_bottleneck_distance(left: np.ndarray, right: np.ndarray) -> float:
    """两组有限持久对之间的精确瓶颈距离（Topp 1.0.0 高速内核）。

    退化情形判定与其余后端（gudhi / native_c_dll）保持一致，确保 oracle
    对拍逐位一致——这是「保持语义一致性」的硬约束，不得改动顺序或取值。
    """
    left = as_finite_pairs(left)
    right = as_finite_pairs(right)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise TopologyError("finite_bottleneck_distance 只接受有限 birth-death 对")
    if left.size == 0 and right.size == 0:
        return 0.0
    if left.size == 0:
        return float(np.max((right[:, 1] - right[:, 0]) / 2.0))
    if right.size == 0:
        return float(np.max((left[:, 1] - left[:, 0]) / 2.0))
    try:
        from topp import bottleneck_distance as topp_bottleneck_distance
    except ImportError as exc:
        raise TopologyError(
            "缺少 Topp 高速瓶颈距离后端（v1.0.0）。该后端与克隆的 "
            "bottleneck_distance_banckend 同源：可 `pip install topp==1.0.0` 安装预编译包，"
            "或 `pip install -e .` 从项目依赖构建。"
        ) from exc
    try:
        result = float(topp_bottleneck_distance(left, right))
    except _TOPP_EXCEPTIONS as exc:
        raise TopologyError(f"Topp 瓶颈距离计算失败：{exc}") from exc
    if not math.isfinite(result):
        # 有限输入（无 essential 点）下 Topp 不应返回非有限值；若出现属上游契约违例，
        # 显式熔断而非把 inf/NaN 透传给下游匹配/缓存，避免静默污染结果。
        raise TopologyError(f"Topp 返回非有限瓶颈距离（{result}），有限持久对输入不应触发此情形")
    return result


def backend_info() -> ToppBackendInfo:
    """加载并校验 Topp 后端，返回可用于启动日志的可复现信息（后端初始化入口）。

    在首次调用 ``finite_bottleneck_distance`` 前可主动调用本函数做 fail-fast 校验：
    Topp 未安装即抛 ``TopologyError``，避免计算期才暴露缺依赖。
    """
    try:
        import topp
    except ImportError as exc:
        raise TopologyError(
            "缺少 Topp 高速瓶颈距离后端（v1.0.0）。可 `pip install topp==1.0.0` "
            "或 `pip install -e .` 从项目依赖构建。"
        ) from exc
    return ToppBackendInfo(
        name=name,
        version=getattr(topp, "__version__", "unknown"),
        source="topp package (Topp 1.0.0, bottleneck_distance_banckend)",
    )
