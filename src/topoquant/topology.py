from __future__ import annotations

import numpy as np
from topp import bottleneck_distance as topp_bottleneck_distance

from .domain import Diagram


class TopologyError(RuntimeError):
    """持续同调计算失败。"""


def compute_persistence(
    points: np.ndarray,
    max_edge_length: float,
    max_homology_dimension: int,
) -> Diagram:
    try:
        from ripser import ripser
    except ImportError as exc:
        raise TopologyError("缺少 Ripser，请先安装项目依赖") from exc

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


def bottleneck_distance(left: np.ndarray, right: np.ndarray) -> float:
    """整张持续图之间的瓶颈距离。

    沿用 notebook 的筛选口径：任一整图为空即判为无穷远（视为不可比）。
    """
    if left.size == 0 or right.size == 0:
        return float("inf")

    left = np.asarray(left, dtype=np.float64).reshape(-1, 2)
    right = np.asarray(right, dtype=np.float64).reshape(-1, 2)
    left_finite = np.isfinite(left).all(axis=1)
    right_finite = np.isfinite(right).all(axis=1)
    left_essential = np.isfinite(left[:, 0]) & np.isposinf(left[:, 1])
    right_essential = np.isfinite(right[:, 0]) & np.isposinf(right[:, 1])
    if not np.all(left_finite | left_essential) or not np.all(
        right_finite | right_essential
    ):
        return float("inf")

    left_births = np.sort(left[left_essential, 0])[::-1]
    right_births = np.sort(right[right_essential, 0])[::-1]
    if len(left_births) != len(right_births):
        return float("inf")
    essential_distance = (
        float(np.max(np.abs(left_births - right_births)))
        if len(left_births)
        else 0.0
    )
    finite_distance = finite_bottleneck_distance(
        left[left_finite], right[right_finite]
    )
    return max(finite_distance, essential_distance)


def finite_bottleneck_distance(left: np.ndarray, right: np.ndarray) -> float:
    """两组有限持久对之间的精确瓶颈距离。"""
    left = np.asarray(left, dtype=np.float64).reshape(-1, 2)
    right = np.asarray(right, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        raise TopologyError("finite_bottleneck_distance 只接受有限 birth-death 对")
    if left.size == 0 and right.size == 0:
        return 0.0
    if left.size == 0:
        return float(np.max((right[:, 1] - right[:, 0]) / 2.0))
    if right.size == 0:
        return float(np.max((left[:, 1] - left[:, 0]) / 2.0))
    try:
        return float(topp_bottleneck_distance(left, right))
    except (RuntimeError, ValueError) as exc:
        raise TopologyError(f"Topp Bottleneck distance 计算失败：{exc}") from exc
