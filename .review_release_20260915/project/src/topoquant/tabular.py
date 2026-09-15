"""表格输出统一出口（TopoQuant 输出规范）。

项目约定：所有"表格数据"终产物必须保存为 ``.csv``（utf-8-sig 编码，带 BOM 以便
Excel 直接打开），**禁止** ``.xls`` / ``.xlsx``。

本模块是该约定的唯一代码出口。任何模块要写出表格数据，都必须经过
:func:`write_table` / :func:`write_table_dicts`，从而从机制上保证：

* 产物恒为 ``.csv`` —— 即便调用方误传 ``.xls`` / ``.xlsx`` 扩展名，也会被自动改写
  为 ``.csv`` 并告警，杜绝"误写 Excel 格式"的回归；
* 编码恒为 ``utf-8-sig``（中文表头/内容在 Excel 中直接可读，不乱码）；
* 列顺序由调用方显式给定，稳定、可跨 run 对比。

散落各处的 ``csv.writer`` / ``csv.DictWriter`` 调用是约定被悄悄改回 Excel 的温床，
已全部收敛到本模块（见 ``reporting.py`` / ``validation.py`` / ``tools/compare_runs.py``）。

规范的适用范围、例外与生效条件见 ``docs/OUTPUT_FORMAT.md``。
提交前与 CI 会通过 ``scripts/check_tabular_format.py`` 扫描禁用信号
（``to_excel`` / ``ExcelWriter`` / ``openpyxl`` / ``xlsxwriter``）做硬阻断。
"""
from __future__ import annotations

import csv
import warnings
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

# 本项目表格输出的唯一规范格式。集中在此，便于全局检索与未来可能的扩展。
TABLE_FORMAT = "csv"

# 被禁止的表格输出格式：任何调用方传入此类扩展名都会被改写为 .csv 并告警。
_FORBIDDEN_EXTENSIONS = {".xls", ".xlsx", ".xlsm", ".xlsb"}


def _coerce_csv(path: str | Path) -> Path:
    """把任意路径强制规范为 ``.csv``。

    若传入 ``.xls`` / ``.xlsx`` 等被禁止的扩展名，自动改写为 ``.csv`` 并返回，
    同时打印告警 —— 这保证即使调用方误写 Excel 扩展名，落盘产物也不会偏离规范。
    其它非 ``.csv`` 扩展名同样统一改写，避免约定被悄悄绕过。
    """
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix in _FORBIDDEN_EXTENSIONS:
        coerced = p.with_suffix(".csv")
        warnings.warn(
            f"表格输出被强制规范为 .csv：{p.name!r} -> {coerced.name!r}",
            stacklevel=3,
        )
        return coerced
    if suffix != ".csv":
        # 例如 .tsv 等也统一为 .csv，保证约定的唯一性。
        return p.with_suffix(".csv")
    return p


def write_table(
    path: str | Path,
    *,
    header: Sequence[str],
    rows: Iterable[Sequence[object]],
    encoding: str = "utf-8-sig",
) -> Path:
    """写出列表型表格（每行为序列）。

    这是项目表格输出的唯一出口，强制 ``.csv`` + ``utf-8-sig``。

    :param path: 目标路径；扩展名若非 ``.csv`` 会被自动规范为 ``.csv``。
    :param header: 表头（列名），顺序即落盘顺序。
    :param rows: 数据行，每行是与 ``header`` 等长的序列。
    :param encoding: 落盘编码，默认 ``utf-8-sig``（带 BOM，Excel 友好）。
    :returns: 实际写入的 ``.csv`` 路径。
    """
    out = _coerce_csv(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding=encoding, newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(header))
        writer.writerows(rows)
    return out


def write_table_dicts(
    path: str | Path,
    *,
    rows: Sequence[Mapping[str, object]],
    encoding: str = "utf-8-sig",
) -> Path:
    """写出字典型表格（每个元素为 dict，键即表头）。

    这是项目表格输出的唯一出口，强制 ``.csv`` + ``utf-8-sig``。表头由首行 dict 的键
    顺序决定（需保证各行键一致）。调用方应至少传入一行样本以稳定表头。

    :param path: 目标路径；扩展名若非 ``.csv`` 会被自动规范为 ``.csv``。
    :param rows: 数据行，dict 列表；表头取自首行键。
    :param encoding: 落盘编码，默认 ``utf-8-sig``。
    :returns: 实际写入的 ``.csv`` 路径。
    :raises ValueError: 当 ``rows`` 为空（无法推导表头）时。
    """
    if not rows:
        raise ValueError("write_table_dicts 至少需要一行样本以推导表头")
    fieldnames = list(rows[0].keys())
    out = _coerce_csv(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding=encoding, newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return out
