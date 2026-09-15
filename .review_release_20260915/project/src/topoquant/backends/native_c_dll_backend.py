from __future__ import annotations

import numpy as np

from ..bottleneck_native import NativeBottleneckError, native_bottleneck_distance
from .base import Diagram, TopologyError, as_finite_pairs

name = "native_c_dll"


def compute_persistence(
    points: np.ndarray,
    max_edge_length: float,
    max_homology_dimension: int,
) -> Diagram:
    """持久图仍由 Ripser 计算（C-DLL 仅承担瓶颈距离）。

    便携 EXE 同样打包 ripser 的预编译 wheel，因此源码与便携在此复用同一持久图内核。
    """
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
    """两组有限持久对之间的精确瓶颈距离（自研 C ABI DLL 内核）。

    退化情形判定与其余后端保持一致，确保 oracle 对拍逐位一致。
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
        return native_bottleneck_distance(left, right)
    except NativeBottleneckError as exc:
        raise TopologyError(str(exc)) from exc
