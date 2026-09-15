from __future__ import annotations

import numpy as np

from .base import Diagram, TopologyError, as_finite_pairs

name = "gudhi"


def compute_persistence(
    points: np.ndarray,
    max_edge_length: float,
    max_homology_dimension: int,
) -> Diagram:
    """用 GUDHI ``RipsComplex`` 计算持续同调（保留 PURE 原实现，原样搬运）。"""
    try:
        import gudhi as gd
    except ImportError as exc:
        raise TopologyError(
            "缺少 GUDHI 后端，请安装可选依赖：pip install topoquant[gudhi]"
        ) from exc
    try:
        complex_ = gd.RipsComplex(points=points, max_edge_length=max_edge_length)
        tree = complex_.create_simplex_tree(max_dimension=max_homology_dimension + 1)
        persistence = tree.persistence()
    except Exception as exc:  # GUDHI 的 C 扩展可能抛出非 Python 异常
        raise TopologyError(f"GUDHI 持续同调计算失败：{exc}") from exc
    result: Diagram = {}
    for dimension in range(max_homology_dimension + 1):
        pairs = [(birth, death) for dim, (birth, death) in persistence if dim == dimension]
        result[dimension] = np.asarray(pairs, dtype=np.float64).reshape(-1, 2)
    return result


def finite_bottleneck_distance(left: np.ndarray, right: np.ndarray) -> float:
    """两组有限持久对之间的精确瓶颈距离（GUDHI ``bottleneck_distance``）。

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
        import gudhi as gd
    except ImportError as exc:
        raise TopologyError(
            "缺少 GUDHI 后端，请安装可选依赖：pip install topoquant[gudhi]"
        ) from exc
    return float(gd.bottleneck_distance(left, right))
