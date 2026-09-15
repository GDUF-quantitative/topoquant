from __future__ import annotations

import importlib.metadata
import json
import logging
import os
import platform
import shutil
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .config import PipelineConfig, iter_source_csv
from .data import STOCK_NAME, read_csv_header, required_columns, stock_code_from_path

LOGGER = logging.getLogger(__name__)
# 表头审计在 5000+ CSV 的目录上要跑好几分钟，原实现用裸 print 无条件提示进度。
# 改走 RichHandler（与进度条同源）后，若沿用根 logger 的默认级别，CLI 不带
# --verbose 时（WARNING）这些提示会被整段过滤掉，预检期间屏幕全黑、像卡死。
# 这里只把本模块的级别钉到 INFO，等价于恢复原来的无条件提示，不影响其它模块。
LOGGER.setLevel(logging.INFO)


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
    for distribution in ("numpy", "pandas", "gudhi", "rich"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = None
    return versions


def _read_csv_header(path: Path) -> list[str]:
    return read_csv_header(path)


def _audit_headers(
    config: PipelineConfig, files: list[Path], progress=None
) -> tuple[int, list[str]]:
    required = required_columns(config)
    valid = 0
    issues: list[str] = []
    seen_codes: dict[str, Path] = {}
    total = len(files)
    # 大目录逐个审计时若无任何输出会像卡死，按数量打印进度提示。
    # 同步向 progress 回调上报：总任务量 = 文件数，当前 = 已审计数；
    # 与 LOGGER 节流同频（每 ~2% 一次 + 末次），避免逐文件刷新拖慢审计本身。
    _tick = max(1, total // 50) if total > 50 else 0
    for idx, path in enumerate(files, 1):
        if _tick and idx % _tick == 0:
            if progress is not None:
                # 有进度条时只走 progress 回调，降级为 debug 避免与 rich Live 重绘
                # 抢同一 CONSOLE 造成屏幕重影（既有约定见 build/matching）。
                LOGGER.debug("审计表头 %d/%d ...", idx, total)
                progress("audit", idx, total, {"valid": valid, "issues": len(issues)})
            else:
                LOGGER.info("审计表头 %d/%d ...", idx, total)
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
    if progress is not None and total:
        # 末次补齐：确保进度条精确收敛到 100%，不丢最后一小段。
        progress("audit", total, total, {"valid": valid, "issues": len(issues)})
    # 无进度条（纯日志）模式下，审计结束给一条收尾，明确阶段推进、
    # 避免「审计完 → 计算开始」之间因无任何输出而被误判卡死。
    if progress is None and total:
        LOGGER.info("审计完成：%d/%d 通过，%d 处问题，进入就绪检查", valid, total, len(issues))
    return valid, issues


def check_build_artifacts_exist(config: PipelineConfig) -> tuple[bool, list[str]]:
    """检查 build 阶段产出的 mmap 持久图与索引是否齐全（match/forecast 的前置依赖）。

    返回 (all_present, missing_paths)。缺失文件包括 diagram_counts.npy、
    diagrams_h{dim0}.npy、diagrams_h{dim1}.npy、diagram_index.json。
    """
    dim0, dim1 = config.distance_dimensions
    required = [
        config.work_dir / "diagram_counts.npy",
        config.work_dir / f"diagrams_h{dim0}.npy",
        config.work_dir / f"diagrams_h{dim1}.npy",
        config.work_dir / "diagram_index.json",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    return (not missing), missing


def inspect_environment(  # noqa: PLR0913
    config: PipelineConfig,
    *,
    require_source: bool = True,
    require_database: bool = False,
    require_gudhi: bool = True,
    require_build_artifacts: bool = False,
    strict_build_artifacts: bool = False,
    progress=None,
) -> PreflightReport:
    warnings: list[str] = []
    errors: list[str] = []
    source_files = _check_source_dir(config, require_source, errors)

    database_exists = config.database_path.is_file()
    if require_database and not database_exists:
        errors.append(f"实验库不存在：{config.database_path}；请先执行 build")

    if require_build_artifacts:
        _check_build_artifacts(config, strict_build_artifacts, errors, warnings)

    versions = _dependency_versions()
    _check_dependency_versions(versions, require_gudhi, errors)

    total_memory, available_memory = _memory_status()
    disk_probe = _existing_ancestor(config.work_dir)
    disk_free = shutil.disk_usage(disk_probe).free
    if not os.access(disk_probe, os.W_OK):
        errors.append(f"工作目录的现有上级目录不可写：{disk_probe}")
    source_size = sum(path.stat().st_size for path in source_files)
    if require_source and source_files:
        LOGGER.info("预检：审计 %d 个 CSV 的表头契约...", len(source_files))
    schema_valid, schema_issues = _audit_headers(config, source_files, progress=progress)
    if require_source and schema_issues:
        errors.append(
            f"{len(schema_issues)} 个 CSV 未通过文件名/表头契约；" + "；".join(schema_issues[:3])
        )
    _append_resource_warnings(config, source_size, disk_free, total_memory, warnings)
    warnings.append(
        "同一 work_dir 禁止并发运行 build/match/forecast；跨进程共享可能触发 SQLITE_BUSY "
        "或读到陈旧 mmap 快照（原子替换仅保证不出现半截文件，不保证 mmap==此刻SQLite）。"
    )

    stages = (
        StagePlan(
            "持续同调", "进程/股票", config.resolved_topology_workers, config.topology_workers
        ),
        StagePlan(
            "瓶颈匹配", "进程/目标点云", config.resolved_matching_workers, config.matching_workers
        ),
        StagePlan(
            "行情预测", "线程/目标点云", config.resolved_forecast_workers, config.forecast_workers
        ),
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


def _check_source_dir(
    config: PipelineConfig, require_source: bool, errors: list[str]
) -> list[Path]:
    """inspect_environment 的行情目录检查簇（M4：提取自 inspect_environment，逻辑逐字平移）。"""
    if config.source_dir.is_dir():
        # O-9：统一枚举口径，与 config 指纹 / data.list_stock_files 一致（scandir /
        # 不跟符号链接 / 小写后缀 / 按名排序）。
        source_files = iter_source_csv(config.source_dir)
        if require_source and not source_files:
            errors.append(f"行情目录中没有 CSV：{config.source_dir}")
    elif require_source:
        errors.append(f"行情目录不存在：{config.source_dir}")
        source_files = []
    else:
        source_files = []
    return source_files


def _check_build_artifacts(
    config: PipelineConfig,
    strict_build_artifacts: bool,
    errors: list[str],
    warnings: list[str],
) -> None:
    """inspect_environment 的 build 产物检查簇（M4：提取自 inspect_environment，逻辑逐字平移）。"""
    artifacts_present, missing_artifacts = check_build_artifacts_exist(config)
    if not artifacts_present:
        message = "build 产物缺失（请先执行 build）：" + "；".join(missing_artifacts)
        if strict_build_artifacts:
            errors.append(message)
        else:
            warnings.append(message)


def _check_dependency_versions(
    versions: dict[str, str | None], require_gudhi: bool, errors: list[str]
) -> None:
    """inspect_environment 的依赖版本检查簇（M4：提取自 inspect_environment，逻辑逐字平移）。"""
    for dependency in ("numpy", "pandas", "rich"):
        if versions[dependency] is None:
            errors.append(f"缺少运行依赖 {dependency}")
    if require_gudhi and versions["gudhi"] is None:
        errors.append("缺少运行依赖 gudhi")


def _append_resource_warnings(
    config: PipelineConfig,
    source_size: int,
    disk_free: int,
    total_memory: int | None,
    warnings: list[str],
) -> None:
    """inspect_environment 的资源类警告簇（M4：提取自 inspect_environment，逻辑逐字平移）。"""
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


def ensure_ready(report: PreflightReport) -> None:
    if report.errors:
        raise PreflightError("；".join(report.errors))


def inspect_results(config: PipelineConfig) -> dict[str, object]:
    if not config.database_path.is_file():
        raise PreflightError(f"实验库不存在：{config.database_path}")
    uri = config.database_path.resolve().as_uri() + "?mode=ro"
    # 只读检查实验库内容：刻意不走 storage.connect（其会 create_schema，只读 URI 上会失败）。
    # 此处只读，PRAGMA/WAL 由写入方（storage.connect）负责，无需重复设置。
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
            "forecasts_complete": scalar(
                "SELECT COUNT(*) FROM forecast_runs WHERE status='complete'"
            ),
            "forecast_errors": scalar("SELECT COUNT(*) FROM forecast_runs WHERE status='error'"),
            "predictions": scalar("SELECT COUNT(*) FROM forecasts"),
        }
        forecast_row = db.execute(
            "SELECT status, signature FROM stage_runs WHERE stage='forecast'"
        ).fetchone()
        summary["forecast_current"] = bool(
            forecast_row
            and forecast_row["signature"] == config.forecast_signature()
            and forecast_row["status"] == "completed"
        )
    metrics_path = config.output_dir / "metrics.json"
    summary["metrics"] = (
        json.loads(metrics_path.read_text(encoding="utf-8"))
        if summary["forecast_current"] and metrics_path.is_file()
        else None
    )
    return summary
