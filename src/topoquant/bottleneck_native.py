from __future__ import annotations

import ctypes
import hashlib
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np


class NativeBottleneckError(RuntimeError):
    """本地 Bottleneck distance 内核不可用或计算失败。"""


@dataclass(frozen=True)
class NativeBackendInfo:
    path: Path
    sha256: str
    solver_config: tuple[int, int, int, int, int, int]


_DLL_PATH = Path(__file__).resolve().parent / "native" / "bottleneck_core_c.dll"
_EXPECTED_SHA256 = "63C0B18EE32193AEF587B22BE6BE0B8A5C0A9D0501C5FDD48F912A9A0CF3208D"
# F:\bottleneck 当前默认 dispatcher：clipped candidates、adaptive threshold、
# dense AoS distance、adaptive adjacency/matcher、degree-ascending order。
_SOLVER_CONFIG = (2, 7, 0, 9, 9, 1)
_DOUBLE_POINTER = ctypes.POINTER(ctypes.c_double)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


@lru_cache(maxsize=1)
def _load_distance_function():
    if not _DLL_PATH.is_file():
        raise NativeBottleneckError(f"缺少本地 Bottleneck 内核：{_DLL_PATH}")
    actual_sha256 = _file_sha256(_DLL_PATH)
    if actual_sha256 != _EXPECTED_SHA256:
        raise NativeBottleneckError(
            "本地 Bottleneck 内核校验失败："
            f"期望 {_EXPECTED_SHA256}，实际 {actual_sha256}"
        )
    try:
        library = ctypes.CDLL(str(_DLL_PATH))
        function = library.bottleneck_core_distance
    except (OSError, AttributeError) as exc:
        raise NativeBottleneckError(f"无法加载本地 Bottleneck 内核：{exc}") from exc
    function.argtypes = [
        _DOUBLE_POINTER,
        ctypes.c_size_t,
        _DOUBLE_POINTER,
        ctypes.c_size_t,
        *([ctypes.c_int] * 6),
    ]
    function.restype = ctypes.c_double
    # function 持有其来源 DLL 的引用；缓存后可安全供当前进程重复调用。
    function._topoquant_library = library
    return function


def backend_info() -> NativeBackendInfo:
    """加载并校验内核，返回可用于启动日志的可复现信息。"""
    _load_distance_function()
    return NativeBackendInfo(
        path=_DLL_PATH,
        sha256=_EXPECTED_SHA256,
        solver_config=_SOLVER_CONFIG,
    )


def native_bottleneck_distance(left: np.ndarray, right: np.ndarray) -> float:
    """通过项目内置的 C ABI DLL 计算 exact Bottleneck distance。"""
    first = np.ascontiguousarray(left, dtype=np.float64).reshape(-1, 2)
    second = np.ascontiguousarray(right, dtype=np.float64).reshape(-1, 2)
    if not np.isfinite(first).all() or not np.isfinite(second).all():
        raise NativeBottleneckError("本地 Bottleneck 内核只接受有限 birth-death 对")

    function = _load_distance_function()
    result = function(
        first.ctypes.data_as(_DOUBLE_POINTER),
        first.shape[0],
        second.ctypes.data_as(_DOUBLE_POINTER),
        second.shape[0],
        *_SOLVER_CONFIG,
    )
    if math.isnan(result):
        raise NativeBottleneckError("本地 Bottleneck 内核返回 NaN")
    return float(result)
