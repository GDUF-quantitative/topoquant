from __future__ import annotations

import contextlib
import csv
import hashlib
import io
import json
import logging
import os
import re
from collections.abc import Iterator, Sequence
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

LOGGER = logging.getLogger(__name__)

from .config import PipelineConfig, iter_source_csv  # noqa: E402
from .domain import CloudWindow  # noqa: E402

STOCK_NAME = re.compile(r"^(?P<code>\d{6})[._-]?(?P<market>SZ|SH)$", re.IGNORECASE)


def clean_code(code: str) -> str:
    """把任意形式的股票代码归一化为「仅字母数字、大写」的紧凑形态。

    例如 ``"000001.SZ"`` / ``"600000.sh"`` / ``"sh600000"`` 都收敛为 ``"000001SZ"`` /
    ``"600000SH"``。供 cloud_id 拼接与文件名解析的兜底清洗共用，避免重复散落
    ``re.sub(r"[^0-9A-Za-z]", ...)`` 片段（见审计 C3）。
    """
    return re.sub(r"[^0-9A-Za-z]", "", code).upper()


class DataError(ValueError):
    """输入行情数据不满足流水线契约。"""


def stock_code_from_path(path: Path) -> str:
    match = STOCK_NAME.fullmatch(path.stem)
    if match:
        return f"{match.group('code')}.{match.group('market').upper()}"
    compact = clean_code(path.stem)
    if not compact:
        raise DataError(f"无法从文件名提取股票代码：{path.name}")
    return compact


def required_columns(config: PipelineConfig, *, with_prev_close: bool = True) -> set[str]:
    """返回单只股票 CSV 的「必需列」集合（真源）。

    契约：本函数**只读取 ``config.features``**，不访问 config 的任何其他属性。
    这样「必需列」的定义与具体的运行配置（路径、并发数、阈值等）彻底解耦，
    preflight 头检查、validation 校验、pipeline 窗口读取都统一走这里，避免散落多处导致漂移。

    - ``with_prev_close=True`` 返回 ``{"EventDate", "prev_close", *config.features}``
    - ``with_prev_close=False`` 返回 ``{"EventDate", *config.features}``
      （pipeline 读取窗口时不需要 ``prev_close``，预测阶段才单独读取）
    """
    columns: set[str] = {"EventDate", *config.features}
    if with_prev_close:
        columns.add("prev_close")
    return columns


def list_stock_files(source_dir: Path) -> list[Path]:
    if not source_dir.is_dir():
        raise FileNotFoundError(f"行情目录不存在：{source_dir}")
    # O-9：统一枚举口径，与 config 指纹 / preflight 审计一致（scandir / 不跟符号链接 /
    # 小写后缀 / 按名排序）；原 glob("*.csv") 在 Linux 大小写敏感、且跟随符号链接的
    # 不一致隐患由此消除。
    files = iter_source_csv(source_dir)
    if not files:
        raise DataError(f"行情目录中没有 CSV：{source_dir}")
    return files


def stock_quality_skip_reason(path: Path, config: PipelineConfig) -> str | None:  # noqa: PLR0911
    """构建前单文件数据质量快扫（P2-2；R2 起在两种模式下都会被调用）。

    返回跳过原因字符串，或 None（无需跳过）。只读取并解析一次文件，不做窗口划分，
    开销远小于完整 build。检查分两级：

    - **硬伤（始终拦截，与 ``data_quality_mode`` 无关）**：读取失败 / 文件为空 / 缺列 /
      **重复交易日**。重复交易日会让 ``load_future_directions`` 在 forecast 阶段直接抛
      ``DataError`` 硬崩——属于"后期才暴露"的隐患，因此提前到 build 入口一次性拦掉
      （R2；默认 ``permissive`` 也生效，但默认值本身未改）。
    - **软伤（仅 strict 拦截）**：非数值行。这类脏数据在 ``standardized_points`` 里已有
      ffill/滚动中位数兜底，不会崩，宽松模式下允许照常入云。

    调用方据此区分日志级别：硬伤统一提示"数据硬伤"，软伤提示"严格模式"。
    """
    try:
        frame = read_csv_flexibly(path)
    except Exception as exc:
        return f"读取失败：{exc}"
    if frame.empty:
        return "文件为空"
    required = required_columns(config)
    missing = sorted(required - set(frame.columns))
    if missing:
        return f"缺少列：{', '.join(missing)}"
    dates = pd.to_datetime(frame["EventDate"], errors="coerce")
    valid_dates = dates.dropna().dt.date
    duplicate_dates = int(valid_dates.duplicated().sum())
    if duplicate_dates:
        # 硬伤：无论 strict 还是 permissive 都必须拦截（R2）。
        return f"{duplicate_dates} 个重复交易日"
    if config.data_quality_mode != "strict":
        # 以下均为软伤，宽松模式（默认）到此为止，不再多做一次数值转换。
        return None
    numeric_columns = list(dict.fromkeys((*config.features, "prev_close")))
    numeric = frame.loc[:, numeric_columns].apply(pd.to_numeric, errors="coerce")
    invalid_numeric_rows = int(numeric.isna().any(axis=1).sum())
    if invalid_numeric_rows:
        ratio = invalid_numeric_rows / len(frame)
        return f"{invalid_numeric_rows} 行非数值（占比 {ratio:.1%}）"
    return None


# 进程级缓存：相同「路径 + mtime + 大小」组合的内容签名直接复用，避免重复流式读取
# 大文件。命中缓存即跳过内容哈希（第一阶段：仅比元数据）；仅当元数据变化或显式要求
# 时才进入第二阶段（流式内容哈希）。--force-content-hash 可绕过元数据缓存强制重算（P1-3）。
_SOURCE_SIGNATURE_CACHE: dict[tuple, str] = {}

# R1（P1-3 真正修复）：进程内缓存对「每进程只调用一次」的生产路径无效——每次 build 都是
# 新进程，缓存必空，仍要对全部 CSV 做完整流式 sha256。故把「元数据键 → 内容哈希」映射
# 落盘到侧车 JSON，使「内容未变即跳过哈希」能跨运行生效。
# 侧车内容按 JSON 载入后同样在进程内缓存一层（_SIDECAR_CACHE），避免同进程重复读盘。
_SIDECAR_CACHE: dict[str, dict[str, str]] = {}
# 每次行情刷新都会产生一组新的 (mtime, size) → 侧车会缓慢增长；保留最近若干条即可
# （dict 保持插入顺序，超限时按插入顺序淘汰最旧的）。
_SIDECAR_MAX_ENTRIES = 32


def _source_metadata_key(files: Sequence[Path]) -> tuple:
    """第一阶段键：文件名 + mtime + 大小 + ctime_ns。任一变化即视为可能需要重算内容签名。

    追加 ``st_ctime_ns``（POSIX=inode 变更时间，Windows=创建时间）作为额外判别维度：
    在内容/元数据变更时通常随之变化，零读取成本；仅在确有变更时才进入第二阶段全量哈希，
    不改变「未变更即复用」快路径语义，亦不产生误报（未变更时 ctime 稳定，键命中）。
    ctime 被备份工具强制改写等不稳定场景下，至多导致一次额外全量哈希（安全、非错误），
    不会漏检。

    路径统一经 ``Path.resolve()`` 标准化并转为 ``str``：既消除符号链接/相对路径/大小写
    差异带来的键漂移，也保证该键可被 JSON 序列化后写入侧车（R1）。
    """
    # 模式 H-6：每个文件只调用一次 ``resolve()`` + 一次 ``stat()``（原实现重复 stat 4 次，
    # 规模随文件数线性放大），零读取成本、键语义不变（resolve 后对真实文件取 mtime/size/ctime）。
    keys: list[tuple[str, int, int, int]] = []
    for path in files:
        resolved = path.resolve()
        info = resolved.stat()
        keys.append(
            (str(resolved), int(info.st_mtime_ns), int(info.st_size), int(info.st_ctime_ns))
        )
    return tuple(keys)


def _sidecar_key(cache_key: tuple) -> str:
    """把元数据键序列化为稳定的 JSON 字符串，作为侧车字典的键。"""
    return json.dumps(cache_key, ensure_ascii=False)


def _load_sidecar(cache_path: Path) -> dict[str, str]:
    """读取侧车：合并目录内所有 ``{stem}.*.json`` 分片（零锁，最契合 spawn）。

    每进程独立分片、读取端合并，消除多进程 build 的 lost update（模式 D①）。
    进程内再缓存一层。文件缺失/损坏一律视为空表，绝不影响主流程。
    为兼容历史单文件侧车，``cache_path`` 本身若存在（``{stem}.json``）也一并合并。
    """
    text_key = str(cache_path)
    cached = _SIDECAR_CACHE.get(text_key)
    if cached is not None:
        return cached
    mapping: dict[str, str] = {}
    shard_paths: list[Path] = []
    if cache_path.is_file():  # 历史兼容：可能存在的单文件侧车
        shard_paths.append(cache_path)
    shard_paths.extend(sorted(cache_path.parent.glob(f"{cache_path.stem}.*.json")))
    for shard in shard_paths:
        try:
            raw = json.loads(shard.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(raw, dict):
            for key, value in raw.items():
                if isinstance(value, str):
                    mapping[str(key)] = value
    _SIDECAR_CACHE[text_key] = mapping
    return mapping


def _store_sidecar(cache_path: Path, mapping: dict[str, str], key: str, signature: str) -> None:
    """原子写回本进程分片 ``{stem}.{pid}.json``（临时文件 + ``os.replace``）。

    每进程独立分片、互不覆盖，读取端合并——零锁、最契合 spawn，消除 lost update
    （模式 D①）。写盘失败仅告警不中断构建。
    """
    mapping[key] = signature
    while len(mapping) > _SIDECAR_MAX_ENTRIES:
        mapping.pop(next(iter(mapping)))
    shard = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.json")
    temporary = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp")
    try:
        shard.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(json.dumps(mapping, ensure_ascii=False), encoding="utf-8")
        os.replace(temporary, shard)
    except OSError as exc:  # 侧车纯属加速，写失败不应影响签名正确性
        LOGGER.warning("源文件指纹侧车分片写入失败（不影响构建）：%s", exc)
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)


def _hash_files_content(files: Sequence[Path]) -> str:
    """file_content_signature 的第二阶段：流式 SHA256 全部文件字节（M4：提取自
    file_content_signature，逻辑逐字平移）。文件名仍参与哈希（区分不同股票）。
    """
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    return digest.hexdigest()


def file_content_signature(
    files: Sequence[Path],
    force_content_hash: bool = False,
    cache_path: Path | None = None,
) -> str:
    """基于文件内容的稳定签名；两阶段加速（P1-3 / R1）。

    第一阶段（廉价）：仅比较每个文件的 ``(mtime, 大小)`` 元数据。若与上次计算时完全一致，
    且未要求强制重算，直接复用已缓存的内容签名，不再读取文件内容——对大行情文件可
    省去昂贵的流式 SHA256。

    第二阶段（仅必要时）：元数据变化（或 ``force_content_hash=True``）时，流式 SHA256 全部
    字节得到内容签名，并更新缓存。

    ``cache_path``（R1 新增，默认 ``None`` → 行为与改动前完全一致）：给定侧车 JSON 路径后，
    第一阶段的命中结果会**跨进程/跨 build 持久化**。这解决了「进程内缓存对每进程只调用
    一次的生产路径形同虚设」的问题。持久化采用每进程分片 ``{stem}.{pid}.json``，读取时合并
    全部分片（模式 D①，零锁、消除多进程 build 的 lost update）。注意：本参数**只改变
    「何时需要重算」，绝不改变哈希算法本身**——命中侧车时返回的就是当初用同一算法算出的
    逐字相同的哈希，签名链与增量缓存不受任何影响。

    内容哈希的语义保持不变：文件名仍参与哈希（区分不同股票），「重新下载导致 mtime 变化但
    内容相同」仍会进入第二阶段并得出相同签名，不会误触发重算。第一阶段键现含 ``ctime_ns``，
    进一步缩小「mtime + 大小同但内容异」的极罕见窗口；该窗口仍属极罕见，必要时可用
    ``force_content_hash=True`` 强制全量重算（同时绕过进程内缓存与侧车）。
    """
    cache_key = _source_metadata_key(files)
    sidecar: dict[str, str] | None = None
    sidecar_key: str | None = None
    if not force_content_hash:
        cached = _SOURCE_SIGNATURE_CACHE.get(cache_key)
        if cached is not None:
            return cached
        if cache_path is not None:
            sidecar = _load_sidecar(cache_path)
            sidecar_key = _sidecar_key(cache_key)
            persisted = sidecar.get(sidecar_key)
            if persisted is not None:
                # 跨 build 命中：直接复用上次算出的内容哈希，完全不读文件内容。
                _SOURCE_SIGNATURE_CACHE[cache_key] = persisted
                return persisted
    signature = _hash_files_content(files)
    # 始终更新缓存（含 force 后的值），确保强制重算后的新签名对后续调用生效（P1-3）。
    _SOURCE_SIGNATURE_CACHE[cache_key] = signature
    if cache_path is not None:
        # 仅在「确有新哈希算出」时才回写侧车，避免无谓写盘（R1）。
        if sidecar is None:
            sidecar = _load_sidecar(cache_path)
        if sidecar_key is None:
            sidecar_key = _sidecar_key(cache_key)
        if sidecar.get(sidecar_key) != signature:
            _store_sidecar(cache_path, sidecar, sidecar_key, signature)
    return signature


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


def read_csv_header(path: str | Path) -> list[str]:
    """读取 CSV 第一行作为列名列表，自动探测编码（utf-8-sig / utf-8 / gb18030）。

    与 ``read_csv_flexibly`` 的区别：本函数只读取文件**第一行**来取表头，
    不做全表解析、不返回 DataFrame；``read_csv_flexibly`` 负责把整张表读成 pandas，
    二者职责不同，请勿混用。

    行为：
    - 仅读取首行（上限 64KB，防御极端长行），表头永远在第一行，无需读取整张表；
    - 依次尝试三种编码，取首个能成功解码的编码的第一行，用 ``csv.reader`` 解析；
    - 空文件（或其首行无法产生任何记录）抛出 ``ValueError("空文件")``；
    - 三种编码都无法解码则抛出 ``ValueError("无法识别编码")``。

    注：预检阶段会对整目录逐个 CSV 审计表头（数千文件）。历史上此处用
    ``read(65536)`` 读了每个文件的前 64KB，冷盘/实时杀软扫描下 I/O 放大约
    300 倍、且无任何进度输出，表现如同卡死。改为只读首行后冷读量从数百 MB
    降到约 1MB，速度提升一个数量级。
    """
    with Path(path).open("rb") as _fh:
        raw = _fh.readline(65536)  # 仅首行；表头必然在第一行
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            text = raw.decode(encoding)
            return [item.strip() for item in next(csv.reader(io.StringIO(text)))]
        except UnicodeDecodeError:
            continue
        except StopIteration as exc:
            raise ValueError("空文件") from exc
    raise ValueError("无法识别编码")


def iter_cloud_windows(config: PipelineConfig, files: Sequence[Path]) -> Iterator[CloudWindow]:
    required = required_columns(config, with_prev_close=False)
    max_windows = config.lookback_trading_days // config.window_size
    min_rows = config.min_windows * config.window_size

    for path in files:
        try:
            # 只读必要列（EventDate + 特征列），避免把整张行情表读入内存（P1-5）。
            # usecols 同时隐含"列缺失即报错"的契约：缺失列会触发 ValueError，
            # 由下方 except 统一包装为 DataError。
            frame = read_csv_flexibly(path, usecols=required)
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
            compact_code = clean_code(stock_code)
            cloud_id = f"{cloud_date:%Y%m%d}_{compact_code}"
            yield CloudWindow(cloud_id, cloud_date, stock_code, path.resolve(), window)


def standardized_points(
    window: CloudWindow,
    features: Sequence[str],
    fillna_strategy: str = "ffill_median",
) -> np.ndarray:
    """把窗口数值特征标准化为点云坐标（零均值、单位方差）。

    ``fillna_strategy`` 控制原始行情中缺失/非数值（NaN/inf）的清理方式：
    - ``"ffill_median"``（默认）：逐列前向填充，再用窗口 5 的滚动中位数修补
      仍未补齐的缺口；整列全缺失则按 0 填充。相比"整行丢弃"，不会因单点脏
      数据而缩小整张点云，更能保留拓扑结构（P2-3）。
    - ``"drop"``：沿用旧行为，丢弃任一列含非有限值的整行（与历史测试契约一致）。
    """
    numeric = window.frame.loc[:, list(features)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=np.float64, copy=True)
    if fillna_strategy == "drop":
        values = values[np.isfinite(values).all(axis=1)]
    else:  # ffill_median（默认）
        values = _fillna_ffill_median(values, window.cloud_id)
    if len(values) < 2:
        raise DataError(f"{window.cloud_id} 清理无效值后不足 2 个点")
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    scale[scale == 0.0] = 1.0
    return (values - mean) / scale


def _fillna_ffill_median(values: np.ndarray, cloud_id: str) -> np.ndarray:
    """逐列填充：前向填充 → 滚动中位数(window=5) 修补剩余缺口 → 全列缺失置 0。

    返回与 ``values`` 同形状的数组；仅修改缺失单元格，不丢弃任何行，
    从而保留点云的行（交易日）规模。
    """
    _, n_cols = values.shape
    out = np.zeros_like(values)
    patched_cols = 0
    for col in range(n_cols):
        series = pd.Series(values[:, col])
        # 前向填充，再以同列末值后向填充兜底（处理首行即缺失的情形）。
        series = series.ffill().bfill()
        still_missing = series.isna()
        if still_missing.all():  # 新增：整列原始值全为 NaN/inf，将被静默填 0
            LOGGER.warning(
                "%s 第 %d 列特征整列无效（全 NaN/inf），将静默填充为全 0 常量维；"
                "permissive 模式下这会注入一个无意义维度，请核查数据源",
                cloud_id,
                col,
            )
        if still_missing.any():
            # 中心化滚动中位数；min_periods=1 保证窗口内只要有有限值即可估出。
            med = series.rolling(window=5, center=True, min_periods=1).median()
            series = series.where(~still_missing, med)
            still_missing = series.isna()
            if still_missing.any():
                series = series.fillna(0.0)
            patched_cols += 1
        out[:, col] = series.to_numpy(dtype=np.float64)
    if patched_cols:
        LOGGER.warning(
            "%s 有 %d 列特征经 ffill/滚动中位数填充（逐列，未丢弃行）", cloud_id, patched_cols
        )
    return out


def load_future_directions(
    path: Path, cloud_date: date, horizon: int
) -> tuple[list[date], np.ndarray, np.ndarray]:
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
        raise DataError(
            f"{path.name} 在 {cloud_date} 后只有 {len(frame)} 个有效交易日，需要 {horizon} 个"
        )
    baseline = float(frame.iloc[0]["prev_close"])
    differences = frame["close"].to_numpy(dtype=np.float64) - baseline
    directions = (differences >= 0.0).astype(np.int8)
    dates = [item.date() for item in frame["EventDate"]]
    return dates, differences, directions
