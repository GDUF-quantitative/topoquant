from __future__ import annotations

import csv
import importlib.metadata
import io
import json
import os
import platform
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .config import PipelineConfig
from .data import STOCK_NAME, stock_code_from_path


@dataclass(frozen=True)
class StagePlan:
    stage: str
    mode: str
    workers: int
    requested: int


@dataclass(frozen=True)
class PreflightReport:
    platform: str
    python_version: str
    logical_cpu_count: int
    total_memory_bytes: int | None
    available_memory_bytes: int | None
    disk_free_bytes: int
    source_file_count: int
    source_size_bytes: int
    schema_valid_file_count: int
    schema_error_file_count: int
    schema_error_examples: tuple[str, ...]
    database_exists: bool
    dependency_versions: dict[str, str | None]
    stages: tuple[StagePlan, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.errors


class PreflightError(RuntimeError):
    """预启动检查未通过。"""


def _memory_status() -> tuple[int | None, int | None]:
    if os.name == "nt":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(MemoryStatus)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return int(status.total_physical), int(status.available_physical)
        except (AttributeError, OSError):
            pass
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        total = page_size * int(os.sysconf("SC_PHYS_PAGES"))
        available = page_size * int(os.sysconf("SC_AVPHYS_PAGES"))
        return total, available
    except (AttributeError, OSError, ValueError):
        return None, None


def _existing_ancestor(path: Path) -> Path:
    candidate = path.resolve()
    while not candidate.exists() and candidate != candidate.parent:
        candidate = candidate.parent
    return candidate


def _dependency_versions() -> dict[str, str | None]:
    versions: dict[str, str | None] = {}
    for distribution in ("numpy", "pandas", "ripser", "topp", "rich"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _read_csv_header(path: Path) -> list[str]:
    raw = path.open("rb").read(65536)
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            text = raw.decode(encoding)
            return [item.strip() for item in next(csv.reader(io.StringIO(text)))]
        except UnicodeDecodeError:
            continue
        except StopIteration as exc:
            raise ValueError("空文件") from exc
    raise ValueError("无法识别编码")


def _audit_headers(config: PipelineConfig, files: list[Path]) -> tuple[int, list[str]]:
    required = {"EventDate", "prev_close", *config.features}
    valid = 0
    issues: list[str] = []
    seen_codes: dict[str, Path] = {}
    for path in files:
        if STOCK_NAME.fullmatch(path.stem) is None:
            issues.append(f"{path.name}: 文件名应为六位代码加 SZ/SH")
            continue
        stock_code = stock_code_from_path(path)
        if stock_code in seen_codes:
            issues.append(f"{path.name}: 与 {seen_codes[stock_code].name} 表示同一股票")
            continue
        seen_codes[stock_code] = path
        try:
            header = _read_csv_header(path)
        except (OSError, ValueError) as exc:
            issues.append(f"{path.name}: {exc}")
            continue
        missing = sorted(required - set(header))
        if missing:
            issues.append(f"{path.name}: 缺少列 {', '.join(missing)}")
            continue
        valid += 1
    return valid, issues


def inspect_environment(
    config: PipelineConfig,
    *,
    require_source: bool = True,
    require_database: bool = False,
    require_topology_backend: bool = True,
) -> PreflightReport:
    warnings: list[str] = []
    errors: list[str] = []
    source_files: list[Path] = []
    if config.source_dir.is_dir():
        source_files = sorted(config.source_dir.glob("*.csv"))
        if require_source and not source_files:
            errors.append(f"行情目录中没有 CSV：{config.source_dir}")
    elif require_source:
        errors.append(f"行情目录不存在：{config.source_dir}")

    database_exists = config.database_path.is_file()
    if require_database and not database_exists:
        errors.append(f"实验库不存在：{config.database_path}；请先执行 build")

    versions = _dependency_versions()
    for dependency in ("numpy", "pandas", "rich"):
        if versions[dependency] is None:
            errors.append(f"缺少运行依赖 {dependency}")
    if require_topology_backend:
        for dependency in ("ripser", "topp"):
            if versions[dependency] is None:
                errors.append(f"缺少运行依赖 {dependency}")

    total_memory, available_memory = _memory_status()
    disk_probe = _existing_ancestor(config.work_dir)
    disk_free = shutil.disk_usage(disk_probe).free
    if not os.access(disk_probe, os.W_OK):
        errors.append(f"工作目录的现有上级目录不可写：{disk_probe}")
    source_size = sum(path.stat().st_size for path in source_files)
    schema_valid, schema_issues = _audit_headers(config, source_files)
    if require_source and schema_issues:
        errors.append(
            f"{len(schema_issues)} 个 CSV 未通过文件名/表头契约；"
            + "；".join(schema_issues[:3])
        )
    if source_size and disk_free < source_size:
        warnings.append("工作盘剩余空间小于原始行情总大小")
    if total_memory is not None and total_memory < 8 * 1024**3:
        warnings.append("物理内存少于 8 GiB，建议降低持续同调和匹配并发数")
    if config.matching_workers > config.logical_cpu_count:
        warnings.append("手动设置的 matching_workers 超过逻辑内核数")
    if config.topology_workers > config.logical_cpu_count:
        warnings.append("手动设置的 topology_workers 超过逻辑内核数")
    if config.top_k % 2 == 0:
        warnings.append("top_k 为偶数，多数投票可能偏向下跌类别；建议使用奇数")

    stages = (
        StagePlan("持续同调", "进程/股票", config.resolved_topology_workers, config.topology_workers),
        StagePlan("瓶颈匹配", "进程/目标点云", config.resolved_matching_workers, config.matching_workers),
        StagePlan("行情预测", "线程/目标点云", config.resolved_forecast_workers, config.forecast_workers),
    )
    return PreflightReport(
        platform=platform.platform(),
        python_version=platform.python_version(),
        logical_cpu_count=config.logical_cpu_count,
        total_memory_bytes=total_memory,
        available_memory_bytes=available_memory,
        disk_free_bytes=disk_free,
        source_file_count=len(source_files),
        source_size_bytes=source_size,
        schema_valid_file_count=schema_valid,
        schema_error_file_count=len(schema_issues),
        schema_error_examples=tuple(schema_issues[:10]),
        database_exists=database_exists,
        dependency_versions=versions,
        stages=stages,
        warnings=tuple(warnings),
        errors=tuple(errors),
    )


def ensure_ready(report: PreflightReport) -> None:
    if report.errors:
        raise PreflightError("；".join(report.errors))


def inspect_results(config: PipelineConfig) -> dict[str, object]:
    if not config.database_path.is_file():
        raise PreflightError(f"实验库不存在：{config.database_path}")
    uri = config.database_path.resolve().as_uri() + "?mode=ro"
    with sqlite3.connect(uri, uri=True) as db:
        db.row_factory = sqlite3.Row

        def scalar(sql: str) -> int:
            return int(db.execute(sql).fetchone()[0])

        summary: dict[str, object] = {
            "clouds": scalar("SELECT COUNT(*) FROM clouds WHERE status='complete'"),
            "cloud_errors": scalar("SELECT COUNT(*) FROM clouds WHERE status='error'"),
            "diagram_pairs": scalar("SELECT COALESCE(SUM(pair_count), 0) FROM diagrams"),
            "targets_checked": scalar("SELECT COUNT(*) FROM match_runs"),
            "targets_selected": scalar("SELECT COUNT(*) FROM match_runs WHERE status='selected'"),
            "matches": scalar("SELECT COUNT(*) FROM matches"),
            "forecasts_complete": scalar("SELECT COUNT(*) FROM forecast_runs WHERE status='complete'"),
            "forecast_errors": scalar("SELECT COUNT(*) FROM forecast_runs WHERE status='error'"),
            "predictions": scalar("SELECT COUNT(*) FROM forecasts"),
        }
        signature_row = db.execute(
            "SELECT value FROM metadata WHERE key='forecast_signature'"
        ).fetchone()
        summary["forecast_current"] = bool(
            signature_row and str(signature_row[0]) == config.forecast_signature()
        )
    metrics_path = config.output_dir / "metrics.json"
    summary["metrics"] = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if summary["forecast_current"] and metrics_path.is_file()
        else None
    )
    return summary
