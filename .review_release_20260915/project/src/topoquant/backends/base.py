from __future__ import annotations

from typing import Protocol

import numpy as np

Diagram = dict[int, np.ndarray]


class TopologyError(RuntimeError):
    """持续同调计算或瓶颈距离失败。"""


def as_finite_pairs(x: object) -> np.ndarray:
    """把任意持久对输入规整为 ``(N, 2)`` 的有限 birth-death 对数组（形状不变量守卫）。

    拒绝裸 ``reshape(-1, 2)`` 会静默吞掉的形状畸变：输入必须是二维且第二维恰为 2，
    否则抛 ``ValueError``。空图 ``(0, 2)`` 是合法的退化输入（有限部分为空），正常放行，
    交由各后端的退化分支处理（整图空 / 半寿命最大值）。

    注意：本函数只校验**形状**，不校验**有限性**——有限性由各后端入口的
    ``np.isfinite(...).all()`` 守卫负责（非有限必须抛 ``TopologyError``）。
    """
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"持久对必须是 (N, 2) 的二维数组，收到形状 {arr.shape}")
    return arr


class TopologyBackend(Protocol):
    """统一后端契约（文档型，非运行时校验）。

    后端以**模块**形式实现，而非类实例。``topology.py`` 门面经
    ``backends.module_for(name)`` 惰性加载模块，再调用其中的两个模块级函数：

    - ``compute_persistence(points, max_edge_length, max_homology_dimension) -> Diagram``
    - ``finite_bottleneck_distance(left, right) -> float``

    注意：本 Protocol 仅是静态类型/文档约定。后端是模块对象，无法在运行时
    经 ``isinstance`` 满足方法式 Protocol，故不做 ``@runtime_checkable`` 修饰，
    也不提供运行时校验；分派正确性由 ``tests/test_backend_oracle.py`` 的三方
    退化对拍（含整图空 / 有限部分为空）保证。

    退化情形（有限部分为空、整图空）的判定语义在各后端内**保持一致**，
    以保证多后端 oracle 对拍在退化情形逐位一致（见 ``tests/test_backend_oracle.py``）。

    算法隔离声明：``v4-topp`` 与 ``v4-native`` 是**两套被隔离的算法变体**，
    不应期望二者的匹配结果逐位相同（缓存已用不同 ``distance_algo_label`` 隔离）。
    非退化（一般）情形只保证在数值容差内一致，不保证逐位相同。
    """

    name: str

    def compute_persistence(
        self,
        points: np.ndarray,
        max_edge_length: float,
        max_homology_dimension: int,
    ) -> Diagram: ...

    def finite_bottleneck_distance(self, left: np.ndarray, right: np.ndarray) -> float: ...


__all__ = ["Diagram", "TopologyBackend", "TopologyError"]
