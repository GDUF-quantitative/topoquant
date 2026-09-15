from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """配置无效。"""


@dataclass(frozen=True)
class PipelineConfig:
    source_dir: Path
    work_dir: Path
    as_of_date: date
    window_size: int = 60
    lookback_trading_days: int = 480
    min_windows: int = 4
    features: tuple[str, ...] = ("money", "volume", "high", "close")
    max_edge_length: float = 3.0
    max_homology_dimension: int = 2
    distance_dimensions: tuple[int, int] = (0, 1)
    distance_threshold: float = 0.1
    top_k: int = 5
    matching_pivots: int = 8
    forecast_horizon: int = 5
    topology_workers: int = 0
    matching_workers: int = 0
    forecast_workers: int = 0

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        config_path = Path(path).resolve()
        try:
            raw: dict[str, Any] = json.loads(config_path.read_text(encoding="utf-8"))
        except FileNotFoundError as exc:
            raise ConfigError(f"配置文件不存在：{config_path}") from exc
        except json.JSONDecodeError as exc:
            raise ConfigError(f"配置文件不是合法 JSON：{exc}") from exc

        required = {"source_dir", "work_dir", "as_of_date"}
        missing = sorted(required - raw.keys())
        if missing:
            raise ConfigError(f"配置缺少字段：{', '.join(missing)}")

        base_dir = config_path.parent

        def resolve_path(value: str) -> Path:
            candidate = Path(value).expanduser()
            if not candidate.is_absolute():
                candidate = base_dir / candidate
            return candidate.resolve()

        try:
            parsed_date = date.fromisoformat(str(raw["as_of_date"]))
        except ValueError as exc:
            raise ConfigError("as_of_date 必须是 YYYY-MM-DD") from exc

        legacy_workers = raw.get("workers")
        config = cls(
            source_dir=resolve_path(str(raw["source_dir"])),
            work_dir=resolve_path(str(raw["work_dir"])),
            as_of_date=parsed_date,
            window_size=int(raw.get("window_size", 60)),
            lookback_trading_days=int(raw.get("lookback_trading_days", 480)),
            min_windows=int(raw.get("min_windows", 4)),
            features=tuple(raw.get("features", ["money", "volume", "high", "close"])),
            max_edge_length=float(raw.get("max_edge_length", 3.0)),
            max_homology_dimension=int(raw.get("max_homology_dimension", 2)),
            distance_dimensions=tuple(raw.get("distance_dimensions", [0, 1])),
            distance_threshold=float(raw.get("distance_threshold", 0.1)),
            top_k=int(raw.get("top_k", 5)),
            matching_pivots=int(raw.get("matching_pivots", 8)),
            forecast_horizon=int(raw.get("forecast_horizon", 5)),
            topology_workers=int(raw.get("topology_workers", 0)),
            matching_workers=int(raw.get("matching_workers", legacy_workers if legacy_workers is not None else 0)),
            forecast_workers=int(raw.get("forecast_workers", 0)),
        )
        config.validate()
        return config

    def validate(self) -> None:
        if self.window_size < 2:
            raise ConfigError("window_size 必须至少为 2")
        if self.lookback_trading_days < self.window_size:
            raise ConfigError("lookback_trading_days 不能小于 window_size")
        if self.min_windows < 1:
            raise ConfigError("min_windows 必须为正整数")
        if self.min_windows * self.window_size > self.lookback_trading_days:
            raise ConfigError("最少窗口所需交易日超过 lookback_trading_days")
        if not self.features or len(set(self.features)) != len(self.features):
            raise ConfigError("features 不能为空或重复")
        if self.max_edge_length <= 0:
            raise ConfigError("max_edge_length 必须大于 0")
        if self.max_homology_dimension < 1:
            raise ConfigError("max_homology_dimension 必须至少为 1")
        if self.distance_dimensions != (0, 1):
            raise ConfigError("当前实验契约要求 distance_dimensions 固定为 [0, 1]")
        if self.distance_threshold <= 0:
            raise ConfigError("distance_threshold 必须大于 0")
        if self.top_k < 1 or self.forecast_horizon < 1:
            raise ConfigError("top_k、forecast_horizon 必须为正整数")
        if not 0 <= self.matching_pivots <= 32:
            raise ConfigError("matching_pivots 必须在 0 到 32 之间；0 表示禁用 pivot")
        if any(value < 0 for value in (
            self.topology_workers, self.matching_workers, self.forecast_workers
        )):
            raise ConfigError("并发数不能为负数；0 表示根据 CPU 内核数自动选择")

    @property
    def database_path(self) -> Path:
        return self.work_dir / "artifacts.sqlite3"

    @property
    def output_dir(self) -> Path:
        return self.work_dir / "outputs"

    @property
    def logical_cpu_count(self) -> int:
        return max(1, os.cpu_count() or 1)

    @property
    def resolved_topology_workers(self) -> int:
        if self.topology_workers:
            return self.topology_workers
        available = self.logical_cpu_count
        return max(1, min(8, available - 1 if available > 1 else 1))

    @property
    def resolved_matching_workers(self) -> int:
        """瓶颈匹配并发数。

        mmap 零拷贝模式下所有 worker 共享同一份持久图物理内存，
        不再需要旧版为防止 Windows 多进程内存爆炸而设的 4 进程上限。
        """
        if self.matching_workers:
            return min(self.matching_workers, self.logical_cpu_count)
        available = self.logical_cpu_count
        return max(1, min(16, available - 1 if available > 1 else 1))

    @property
    def resolved_forecast_workers(self) -> int:
        if self.forecast_workers:
            return self.forecast_workers
        return max(1, min(32, self.logical_cpu_count * 2))

    def serializable(self) -> dict[str, Any]:
        data = asdict(self)
        data["source_dir"] = str(self.source_dir)
        data["work_dir"] = str(self.work_dir)
        data["as_of_date"] = self.as_of_date.isoformat()
        data["features"] = list(self.features)
        data["distance_dimensions"] = list(self.distance_dimensions)
        data["resolved_workers"] = {
            "topology": self.resolved_topology_workers,
            "matching": self.resolved_matching_workers,
            "forecast": self.resolved_forecast_workers,
        }
        return data

    def topology_signature(self) -> str:
        topology_fields = {
            "persistence_backend": "ripser-0.6-v1",
            "source_dir": str(self.source_dir),
            "as_of_date": self.as_of_date.isoformat(),
            "window_size": self.window_size,
            "lookback_trading_days": self.lookback_trading_days,
            "min_windows": self.min_windows,
            "features": self.features,
            "max_edge_length": self.max_edge_length,
            "max_homology_dimension": self.max_homology_dimension,
        }
        payload = json.dumps(topology_fields, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def matching_signature(self) -> str:
        fields = {
            "topology": self.topology_signature(),
            "distance_dimensions": self.distance_dimensions,
            "distance_threshold": self.distance_threshold,
            "top_k": self.top_k,
            "matching_pivots": self.matching_pivots,
            # 距离语义版本：改变距离定义时必须递增，否则旧结果会被误当成有效缓存。
            # Ripser 生成持续图；Topp 0.1.0 计算有限部分，本质类按 birth 精确配对。
            "distance_algo": "topp-0.1.0-exact-pivot-threshold-v6",
        }
        payload = json.dumps(fields, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def forecast_signature(self) -> str:
        fields = {
            "matching": self.matching_signature(),
            "forecast_horizon": self.forecast_horizon,
        }
        payload = json.dumps(fields, sort_keys=True).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
