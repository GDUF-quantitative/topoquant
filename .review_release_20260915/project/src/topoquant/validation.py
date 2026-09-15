from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from .config import PipelineConfig
from .data import (
    STOCK_NAME,
    list_stock_files,
    read_csv_flexibly,
    required_columns,
    stock_code_from_path,
)
from .tabular import write_table, write_table_dicts

ValidationProgress = Callable[[str, int, int, Mapping[str, int]], None]


def _validate_one(path: Path, config: PipelineConfig) -> dict[str, object]:
    row: dict[str, object] = {
        "file": path.name,
        "stock_code": "",
        "status": "error",
        "rows": 0,
        "invalid_dates": 0,
        "duplicate_dates": 0,
        "invalid_numeric_rows": 0,
        "history_rows": 0,
        "eligible_windows": 0,
        "future_rows": 0,
        "message": "",
    }
    if STOCK_NAME.fullmatch(path.stem) is None:
        row["message"] = "文件名应为六位代码加 SZ/SH"
        return row
    row["stock_code"] = stock_code_from_path(path)
    required = required_columns(config)
    try:
        frame = read_csv_flexibly(path)
    except Exception as exc:
        row["message"] = f"读取失败：{exc}"
        return row
    row["rows"] = len(frame)
    missing = sorted(required - set(frame.columns))
    if missing:
        row["message"] = f"缺少列：{', '.join(missing)}"
        return row

    dates = pd.to_datetime(frame["EventDate"], errors="coerce")
    invalid_dates = int(dates.isna().sum())
    valid_dates = dates.dropna().dt.date
    duplicate_dates = int(valid_dates.duplicated().sum())
    numeric_columns = list(dict.fromkeys((*config.features, "prev_close")))
    numeric = frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    invalid_numeric_rows = int(numeric.isna().any(axis=1).sum())
    history_rows = int((valid_dates <= config.as_of_date).sum())
    eligible_windows = min(
        history_rows // config.window_size,
        config.lookback_trading_days // config.window_size,
    )
    future_rows = len({value for value in valid_dates if value > config.as_of_date})
    has_target_date = config.as_of_date in set(valid_dates)

    row.update({
        "invalid_dates": invalid_dates,
        "duplicate_dates": duplicate_dates,
        "invalid_numeric_rows": invalid_numeric_rows,
        "history_rows": history_rows,
        "eligible_windows": eligible_windows,
        "future_rows": future_rows,
    })
    problems: list[str] = []
    if invalid_dates:
        problems.append(f"{invalid_dates} 行日期无效")
    if duplicate_dates:
        problems.append(f"{duplicate_dates} 个重复交易日")
    if invalid_numeric_rows:
        problems.append(f"{invalid_numeric_rows} 行数值字段无效")
    if has_target_date and future_rows < config.forecast_horizon:
        problems.append(f"截止日后仅 {future_rows} 个交易日")
    if problems:
        row["message"] = "；".join(problems)
    elif eligible_windows < config.min_windows:
        row["status"] = "excluded"
        row["message"] = f"历史数据不足 {config.min_windows} 个完整窗口"
    else:
        row["status"] = "valid"
    return row


def validate_dataset(
    config: PipelineConfig,
    progress: ValidationProgress | None = None,
) -> dict[str, int]:
    files = list_stock_files(config.source_dir)
    worker_count = max(1, min(8, config.resolved_forecast_workers))
    counts = {"valid": 0, "excluded": 0, "error": 0, "workers": worker_count}
    if progress is not None:
        progress("validation", 0, len(files), counts)

    if worker_count == 1:
        results = (_validate_one(path, config) for path in files)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=worker_count)
        results = executor.map(lambda path: _validate_one(path, config), files)

    rows: list[dict[str, object]] = []
    try:
        for index, row in enumerate(results, 1):
            rows.append(row)
            counts[str(row["status"])] += 1
            if progress is not None:
                progress("validation", index, len(files), counts)
    finally:
        if executor is not None:
            executor.shutdown()

    config.output_dir.mkdir(parents=True, exist_ok=True)
    # 经统一出口写出为 CSV（utf-8-sig），列顺序稳定。
    # 零文件校验为退化场景：保留「仅表头」空文件行为（与改造前一致），
    # 因为 write_table_dicts 在空行时无法推导表头。
    if rows:
        write_table_dicts(config.output_dir / "data_validation.csv", rows=rows)
    else:
        write_table(
            config.output_dir / "data_validation.csv",
            header=[
                "file", "stock_code", "status", "rows", "invalid_dates",
                "duplicate_dates", "invalid_numeric_rows", "history_rows",
                "eligible_windows", "future_rows", "message",
            ],
            rows=[],
        )
    summary = {"total": len(files), **counts}
    (config.output_dir / "data_validation.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary
