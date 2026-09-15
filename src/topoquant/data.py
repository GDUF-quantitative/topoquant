from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PipelineConfig
from .domain import CloudWindow


STOCK_NAME = re.compile(r"^(?P<code>\d{6})[._-]?(?P<market>SZ|SH)$", re.IGNORECASE)


class DataError(ValueError):
    """输入行情数据不满足流水线契约。"""


def stock_code_from_path(path: Path) -> str:
    match = STOCK_NAME.fullmatch(path.stem)
    if match:
        return f"{match.group('code')}.{match.group('market').upper()}"
    compact = re.sub(r"[^0-9A-Za-z]", "", path.stem).upper()
    if not compact:
        raise DataError(f"无法从文件名提取股票代码：{path.name}")
    return compact


def list_stock_files(source_dir: Path) -> list[Path]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"行情目录不存在：{source_dir}")
    files = sorted(source_dir.glob("*.csv"))
    if not files:
        raise DataError(f"行情目录中没有 CSV：{source_dir}")
    return files


def source_signature(files: Sequence[Path]) -> str:
    """基于文件内容的稳定签名，忽略文件名修改时间与文件大小。

    采用流式 sha256：逐个文件读取全部字节参与哈希。因此「重新下载 / 解压导致
    mtime 变化但内容相同」不会触发重算；只有内容真正变化才会改变签名。
    文件名仍参与哈希，避免不同股票（不同文件名）被误判为同一来源。
    """
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def read_csv_flexibly(path: Path, usecols: Sequence[str] | None = None) -> pd.DataFrame:
    last_error: Exception | None = None
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return pd.read_csv(path, encoding=encoding, usecols=usecols, low_memory=False)
        except UnicodeDecodeError as exc:
            last_error = exc
        except ValueError:
            raise
    raise DataError(f"无法识别 CSV 编码：{path}") from last_error


def iter_cloud_windows(config: PipelineConfig, files: Sequence[Path]) -> Iterator[CloudWindow]:
    required = {"EventDate", *config.features}
    max_windows = config.lookback_trading_days // config.window_size
    min_rows = config.min_windows * config.window_size

    for path in files:
        try:
            frame = read_csv_flexibly(path)
        except Exception as exc:
            raise DataError(f"读取 {path.name} 失败：{exc}") from exc
        missing = sorted(required - set(frame.columns))
        if missing:
            raise DataError(f"{path.name} 缺少字段：{', '.join(missing)}")

        frame = frame.copy()
        frame["EventDate"] = pd.to_datetime(frame["EventDate"], errors="coerce")
        frame = frame.loc[frame["EventDate"].notna()]
        frame = frame.loc[frame["EventDate"].dt.date <= config.as_of_date]
        frame = frame.sort_values("EventDate", ascending=False, kind="stable")
        frame = frame.iloc[: config.lookback_trading_days].reset_index(drop=True)
        if len(frame) < min_rows:
            continue

        stock_code = stock_code_from_path(path)
        window_count = min(len(frame) // config.window_size, max_windows)
        for index in range(window_count):
            start = index * config.window_size
            stop = start + config.window_size
            window = frame.iloc[start:stop].copy()
            cloud_date = window.iloc[0]["EventDate"].date()
            compact_code = re.sub(r"[^0-9A-Za-z]", "", stock_code)
            cloud_id = f"{cloud_date:%Y%m%d}_{compact_code}"
            yield CloudWindow(cloud_id, cloud_date, stock_code, path.resolve(), window)


def standardized_points(window: CloudWindow, features: Sequence[str]) -> np.ndarray:
    numeric = window.frame.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64, copy=True)
    values = values[np.isfinite(values).all(axis=1)]
    if len(values) < 2:
        raise DataError(f"{window.cloud_id} 清理无效值后不足 2 个点")
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale[scale == 0.0] = 1.0
    return (values - mean) / scale


def load_future_directions(
    path: Path,
    cloud_date: date,
    horizon: int,
) -> tuple[list[date], np.ndarray, np.ndarray, np.ndarray]:
    try:
        frame = read_csv_flexibly(path, usecols=["EventDate", "prev_close", "close"])
    except ValueError as exc:
        raise DataError(f"{path.name} 缺少 EventDate/prev_close/close") from exc
    frame["EventDate"] = pd.to_datetime(frame["EventDate"], errors="coerce")
    frame["prev_close"] = pd.to_numeric(frame["prev_close"], errors="coerce")
    frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
    frame = frame.dropna(subset=["EventDate", "prev_close", "close"])
    frame = frame.loc[frame["EventDate"].dt.date > cloud_date]
    frame = frame.sort_values("EventDate", ascending=True, kind="stable")
    if frame["EventDate"].dt.date.duplicated().any():
        raise DataError(f"{path.name} 在 {cloud_date} 后存在重复交易日")
    frame = frame.iloc[:horizon]
    if len(frame) != horizon:
        raise DataError(f"{path.name} 在 {cloud_date} 后只有 {len(frame)} 个有效交易日，需要 {horizon} 个")
    baseline = float(frame.iloc[0]["prev_close"])
    closes = frame["close"].to_numpy(dtype=np.float64)
    if baseline <= 0.0 or np.any(closes <= 0.0):
        raise DataError(f"{path.name} 在 {cloud_date} 后存在非正价格，无法计算对数收益率")
    differences = closes - baseline
    directions = (differences >= 0.0).astype(np.int8)
    log_returns = np.log(closes / baseline)
    dates = [item.date() for item in frame["EventDate"]]
    return dates, differences, directions, log_returns

