"""共享终端颜色与打印助手（解决重复 A1）。

原先 ``run.py`` 与 ``setup_env.py`` 各自维护了一份几乎相同的 ``C`` 字典与
``_c`` / ``_sep`` / ``_ok`` / ``_warn`` / ``_err`` / ``_info`` / ``_hint`` 助手，
导致重复与漂移。现将它们收敛到此处作为唯一真源。

历史说明：两个脚本对 ``C`` 字典使用了不同的键名
（``run.py`` 用描述性键 ``cyan`` / ``green`` …；``setup_env.py`` 用单字母键
``C`` / ``G`` / ``R_`` …），为兼容二者现有调用点，``C`` 同时保留两套键，
指向相同的 ANSI 转义序列。
"""

from __future__ import annotations

import warnings

# 废弃的单字母颜色别名（仅保留作过渡兼容；新代码一律用描述性键）。
_DEPRECATED_COLOR_ALIASES = frozenset({"R", "B", "D", "G", "Y", "R_", "C"})


class _ColorDict(dict):
    """颜色码字典；访问废弃的单字母别名（如 ``C["R"]``）时发出 DeprecationWarning。

    描述性键（reset/bold/dim/cyan/green/yellow/red/magenta/blue）正常使用、不告警。
    """

    def __getitem__(self, key):
        if key in _DEPRECATED_COLOR_ALIASES:
            warnings.warn(
                f"颜色键 {key!r} 是废弃的单字母别名，请改用描述性键"
                f"（如 'reset'/'bold'/'cyan'/'green'）；将在未来版本移除",
                DeprecationWarning,
                stacklevel=2,
            )
        return super().__getitem__(key)


C = _ColorDict(
    {
        "reset": "\033[0m",
        "bold": "\033[1m",
        "dim": "\033[2m",
        "cyan": "\033[36m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "red": "\033[31m",
        "magenta": "\033[35m",
        "blue": "\033[34m",
        # setup_env 兼容别名（单字母键，已废弃）
        "R": "\033[0m",
        "B": "\033[1m",
        "D": "\033[2m",
        "G": "\033[32m",
        "Y": "\033[33m",
        "R_": "\033[31m",
        "C": "\033[36m",
    }
)


def c(text: str, *codes: str) -> str:
    """包裹 ANSI 颜色码。"""
    prefix = "".join(codes)
    return f"{prefix}{text}{C['reset']}"


def sep(title: str = "") -> None:
    width = 60
    if title:
        # 分隔线 + 标题居左；尾部按 width 补齐（负数乘法得空串，与内联版一致）。
        title_bar = "─" * 4 + " " + title + " " + "─" * (width - len(title) - 6)
        print(f"\n{c(title_bar, C['dim'], C['bold'])}")
    else:
        print(f"{c('─' * width, C['dim'])}")


def ok(text: str) -> str:
    return c(f"✓ {text}", C["green"])


def warn(text: str) -> str:
    return c(f"⚠ {text}", C["yellow"])


def err(text: str) -> str:
    return c(f"✗ {text}", C["red"])


def info(text: str) -> str:
    return c(text, C["cyan"])


def hint(text: str) -> str:
    return c(text, C["dim"])


# ── 兼容别名（deprecated）────────────────────────────
# run.py 旧代码使用下划线前缀的私有名，保留别名以兼容可能的外部引用。
_c = c  # deprecated: use c
_sep = sep  # deprecated: use sep
_ok = ok  # deprecated: use ok
_warn = warn  # deprecated: use warn
_err = err  # deprecated: use err
_info = info  # deprecated: use info
_hint = hint  # deprecated: use hint
