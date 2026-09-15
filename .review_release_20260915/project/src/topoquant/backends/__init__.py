"""同调/瓶颈内核后端注册表。

PURE 默认采用 Topp 高速瓶颈距离后端（已替换自研 C-DLL 内核）：
- ``ripser_topp``：Ripser 算持久图 + Topp 1.0.0 算精确瓶颈距离（默认）；
- ``native_c_dll``：自研 C ABI DLL 内核，保留为已注册的可选回退；
- ``gudhi``：可选 ``[gudhi]`` extra，用于对拍/兜底。

所有后端都实现同一契约（见 ``base.TopologyBackend``），由 ``topoquant.topology``
门面按名称分派。``pipeline.py`` 唯一经由 ``topology.py`` 耦合，不直接 import 任何具体后端。
"""

from __future__ import annotations

from .base import Diagram, TopologyBackend, TopologyError

_REGISTRY: dict[str, str] = {
    "gudhi": "topoquant.backends.gudhi_backend",
    "ripser_topp": "topoquant.backends.ripser_topp_backend",
    "native_c_dll": "topoquant.backends.native_c_dll_backend",
}


def available_backends() -> tuple[str, ...]:
    """返回所有已注册后端名称。"""
    return tuple(_REGISTRY)


def module_for(name: str):
    """返回指定后端的模块对象（惰性加载，避免无关后端被导入）。"""
    import importlib

    if name not in _REGISTRY:
        raise TopologyError(f"未知拓扑后端：{name!r}（可选：{', '.join(_REGISTRY)}）")
    return importlib.import_module(_REGISTRY[name])


__all__ = [
    "Diagram",
    "TopologyBackend",
    "TopologyError",
    "available_backends",
    "module_for",
]
