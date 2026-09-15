from __future__ import annotations

import contextvars
import importlib
from typing import Any

import numpy as np

from .backends.base import Diagram, TopologyError

# ── 后端分派（融合升级核心）────────────────────────────────────────────
# PURE 不再硬依赖 GUDHI，且默认即采用 Topp 高速瓶颈距离后端（已替换自研 C-DLL 内核）：
#   - 默认 ``ripser_topp``（Ripser 算持久图 + Topp 1.0.0 算精确瓶颈距离）；
#   - 自研 ``native_c_dll``（C ABI DLL 内核）保留为**已注册的可选回退**，不再作为默认；
#   - GUDHI 仅作为可选 ``[gudhi]`` extra，用于对拍/兜底。
# ``pipeline.py`` 唯一经由本模块耦合，不直接 import 任何具体后端。
# 若一个便携/冻结 EXE 不打包 topp，请在配置显式设 ``topology_backend: native_c_dll``。
_DEFAULT_BACKEND = "ripser_topp"

# 模式 G：后端选择改为 ``contextvars.ContextVar``，使后端切换随任务上下文隔离。
# 在 ``spawn`` 多进程下，子进程重新 import 本模块会得到默认的 ``_DEFAULT_BACKEND``，
# 不会继承父进程的临时切换；并行任务（线程池 / 协程）各自 ``set`` 互不污染，
# 解决了模块级全局在并发切换后端时「串台」的隐患。公开 API（set/get）保持不变。
_ACTIVE_BACKEND: contextvars.ContextVar[str] = contextvars.ContextVar(
    "topoquant.topology.backend", default=_DEFAULT_BACKEND
)

_REGISTRY: dict[str, str] = {
    "gudhi": "topoquant.backends.gudhi_backend",
    "ripser_topp": "topoquant.backends.ripser_topp_backend",
    "native_c_dll": "topoquant.backends.native_c_dll_backend",
}

# 默认后端优先使用 Topp 高速内核；自研 native_c_dll 作为已注册回退。


def available_backends() -> tuple[str, ...]:
    """返回所有已注册后端名称。"""
    return tuple(_REGISTRY)


def set_topology_backend(name: str) -> None:
    """切换同调/瓶颈内核后端（唯一耦合点：仅经由本门面）。

    在**当前上下文**生效（``ContextVar.set``），不影响其他任务上下文，亦不会被
    ``spawn`` 子进程继承（子进程以默认后端启动）。
    """
    if name not in _REGISTRY:
        raise TopologyError(f"未知拓扑后端：{name!r}（可选：{', '.join(_REGISTRY)}）")
    # 早失败：入口即导入后端模块，捕获模块级导入/语法错误，避免在 spawn 子进程
    # 首次调用时才暴露。重后端库（gudhi）仍按设计惰性导入，其可用性在首次 compute
    # 时由后端函数抛清晰 TopologyError 报告（gudhi 可选，不应在入口硬依赖）。
    importlib.import_module(_REGISTRY[name])
    _ACTIVE_BACKEND.set(name)


def active_backend_name() -> str:
    """当前上下文生效的后端名称。"""
    return _ACTIVE_BACKEND.get()


def backend_info():
    """返回当前后端初始化信息（若后端提供 ``backend_info`` 入口）。

    用于启动期 fail-fast 校验后端依赖是否就绪（如 Topp 是否安装）。后端模块
    未提供该入口时返回 ``None``，调用方不应依赖其返回值做关键分支。
    """
    module = _backend_module()
    factory = getattr(module, "backend_info", None)
    if factory is None:
        return None
    return factory()


def _backend_module() -> Any:
    return importlib.import_module(_REGISTRY[_ACTIVE_BACKEND.get()])


def compute_persistence(
    points: np.ndarray,
    max_edge_length: float,
    max_homology_dimension: int,
) -> Diagram:
    """计算 Rips 复形的持续同调，返回 ``0..max_homology_dimension`` 各维的持久对。

    内部按 ``active_backend_name()`` 分派到具体后端实现。
    """
    return _backend_module().compute_persistence(points, max_edge_length, max_homology_dimension)


def finite_bottleneck_distance(left: np.ndarray, right: np.ndarray) -> float:
    """两组**有限**持久对之间的精确瓶颈距离（严格数学定义，空集合法）。

    按 ``active_backend_name()`` 分派到具体后端（GUDHI / Topp / C-DLL）。
    与其余后端共享同一退化情形判定语义。

    不变量守卫（模式 G）：
    - 形状：``left`` / ``right`` 必须是 ``(N, 2)`` 的二维有限 birth-death 对数组，
      畸形形状（如 ``(k, 3)``、非二维）由各后端入口抛 ``ValueError``，不再被
      裸 ``reshape(-1, 2)`` 静默重塑成错误数据。
    - 有限性：含 ``NaN`` / ``±inf`` 的出生-死亡对必须抛 ``TopologyError``，不得
      静默返回 ``inf`` / ``NaN``。
    - 退化（有限部分为空）：空图 ``(0, 2)`` 是合法输入，按半寿命最大值处理，详见各后端。
    """
    return _backend_module().finite_bottleneck_distance(left, right)


def bottleneck_distance(left: np.ndarray, right: np.ndarray) -> float:
    """整张持续图之间的瓶颈距离。

    保留：承载「整图空→∞」的显式业务语义，被测试引用；请勿删除，与
    ``finite_bottleneck_distance``（严格数学定义，空集合法）区分使用。
    当前流水线只调用 ``finite_bottleneck_distance``，本函数暂无生产调用方（R3）。

    「整图空→+∞」判定口径（模式 G 文档化）：
    - 判定依据是**整图是否为空**——``left.size == 0 or right.size == 0`` 即判为
      无穷远（视为不可比）。这里的「空」指**整张图没有任何持久对**（含本质类），
      即传入数组 ``size == 0``；它与 ``finite_bottleneck_distance`` 中「有限部分
      为空但图非全空」的退化分支（返回半寿命最大值）语义不同，二者不可混用。
    - 注意区分「含 inf 死亡值的本质类」：本质类持久对的死亡值在数学上是 ``+∞``，
      但它是**图内的合法持久对**（``size != 0``），不应触发「整图空→+∞」；
      ``finite_bottleneck_distance`` 只接受剔除本质类后的有限对，故不会出现 inf 死亡值。
      这条口径由 ``tests/test_backend_oracle.py`` 的三方 oracle 对拍锁死。
    只应作用于"整张图"，不要用于图的某个子部分——见 ``finite_bottleneck_distance``。
    """
    if left.size == 0 or right.size == 0:
        return float("inf")
    return finite_bottleneck_distance(left, right)
