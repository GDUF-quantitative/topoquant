from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict

from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from .config import PipelineConfig
from .pipeline import build_topology, forecast, match_clouds, run_all
from .preflight import ensure_ready, inspect_environment, inspect_results
from .reporting import generate_outputs
from .validation import validate_dataset


CONSOLE = Console()
STAGE_LABELS = {
    "topology": "持续同调",
    "matching": "瓶颈匹配",
    "forecast": "行情预测",
    "validation": "数据验证",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="持续同调股票点云实验流水线",
        epilog=(
            "示例：\n"
            "  topoquant --config config.json preflight\n"
            "  topoquant --config config.json validate-data\n"
            "  topoquant --config config.json run\n"
            "  topoquant --config config.json status"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", required=True, help="JSON 配置文件路径")
    parser.add_argument("--verbose", action="store_true", help="显示诊断日志")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON，不显示动态进度")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight", help="检测运行环境并展示实际并发计划")
    subparsers.add_parser("validate-data", help="完整扫描行情日期、数值和可用窗口")
    run_sub = subparsers.add_parser("run", help="预检通过后依次执行全部阶段")
    build_sub = subparsers.add_parser("build", help="预检后生成点云并计算持续同调")
    for sub in (run_sub, build_sub):
        sub.add_argument(
            "--reset", action="store_true",
            help="清空实验库与 mmap 中间产物后从头重算（换/改股票数据时用）",
        )
        sub.add_argument(
            "--force-rebuild", dest="force_rebuild", action="store_true",
            help="忽略“行情/配置已变化”保护，就地清空旧数据并重算（授权混合实验）",
        )
    subparsers.add_parser("match", help="预检后筛选拓扑相似点云")
    subparsers.add_parser("forecast", help="预检后生成并评估未来走势预测")
    subparsers.add_parser("report", help="重新导出报告并显示结果摘要")
    subparsers.add_parser("status", help="只读查看实验进度和已有结果")
    return parser


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "未知"
    size = float(value)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if size < 1024 or unit == "TiB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return str(value)


def render_preflight(report) -> None:
    machine = Table(title="预启动环境检测", show_header=False)
    machine.add_column("项目", style="cyan")
    machine.add_column("检测结果")
    machine.add_row("操作系统", report.platform)
    machine.add_row("Python", report.python_version)
    machine.add_row("逻辑内核", str(report.logical_cpu_count))
    machine.add_row(
        "内存",
        f"可用 {_format_bytes(report.available_memory_bytes)} / 总计 {_format_bytes(report.total_memory_bytes)}",
    )
    machine.add_row("工作盘剩余", _format_bytes(report.disk_free_bytes))
    machine.add_row(
        "行情输入",
        f"{report.source_file_count} 个 CSV，{_format_bytes(report.source_size_bytes)}",
    )
    machine.add_row(
        "文件契约",
        f"通过 {report.schema_valid_file_count}，错误 {report.schema_error_file_count}",
    )
    machine.add_row("实验库", "已存在" if report.database_exists else "尚未创建")
    CONSOLE.print(machine)

    plan = Table(title="本次启动并发计划")
    plan.add_column("阶段")
    plan.add_column("并发模型")
    plan.add_column("配置")
    plan.add_column("实际并发", justify="right", style="green")
    for stage in report.stages:
        plan.add_row(stage.stage, stage.mode, "自动" if stage.requested == 0 else str(stage.requested), str(stage.workers))
    CONSOLE.print(plan)

    dependencies = ", ".join(
        f"{name}={version or '缺失'}" for name, version in report.dependency_versions.items()
    )
    CONSOLE.print(f"[dim]依赖：{dependencies}[/dim]")
    for warning in report.warnings:
        CONSOLE.print(f"[yellow]警告：{warning}[/yellow]")
    for error in report.errors:
        CONSOLE.print(f"[red]错误：{error}[/red]")
    if report.ready:
        CONSOLE.print("[bold green]预启动检查通过[/bold green]")


def render_action_result(result: object) -> None:
    table = Table(title="阶段结果")
    table.add_column("阶段", style="cyan")
    table.add_column("指标")
    if isinstance(result, dict) and result and all(isinstance(value, dict) for value in result.values()):
        for stage, values in result.items():
            table.add_row(str(stage), ", ".join(f"{key}={value}" for key, value in values.items()))
    elif isinstance(result, dict):
        table.add_row("完成", ", ".join(f"{key}={value}" for key, value in result.items()))
    else:
        table.add_row("完成", str(result))
    CONSOLE.print(table)


def render_status(summary: dict[str, object]) -> None:
    stage = Table(title="实验进度与结果")
    stage.add_column("指标", style="cyan")
    stage.add_column("数量", justify="right")
    labels = {
        "clouds": "完成点云",
        "cloud_errors": "点云错误",
        "diagram_pairs": "持续同调数值对",
        "targets_checked": "已检查目标",
        "targets_selected": "已选目标",
        "matches": "Top-N 匹配",
        "forecasts_complete": "完成预测目标",
        "forecast_errors": "预测错误",
        "predictions": "逐日预测",
    }
    for key, label in labels.items():
        stage.add_row(label, str(summary[key]))
    CONSOLE.print(stage)

    metrics = summary.get("metrics")
    if not isinstance(metrics, dict):
        CONSOLE.print("[yellow]尚未生成 metrics.json；可执行 report。[/yellow]")
        return
    accuracy = Table(title="预测准确率")
    accuracy.add_column("跨度")
    accuracy.add_column("正确/总数", justify="right")
    accuracy.add_column("准确率", justify="right")
    for horizon, values in metrics.get("by_horizon", {}).items():
        value = values.get("accuracy")
        accuracy.add_row(
            str(horizon),
            f"{values.get('correct', 0)}/{values.get('total', 0)}",
            "无数据" if value is None else f"{value:.2%}",
        )
    overall = metrics.get("overall", {})
    value = overall.get("accuracy")
    accuracy.add_row(
        "整体",
        f"{overall.get('correct', 0)}/{overall.get('total', 0)}",
        "无数据" if value is None else f"{value:.2%}",
        style="bold",
    )
    CONSOLE.print(accuracy)


def _run_with_progress(
    command: str,
    config: PipelineConfig,
    reset: bool = False,
    force_rebuild: bool = False,
) -> object:
    task_ids: dict[str, int] = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        TextColumn("{task.fields[summary]}"),
        console=CONSOLE,
    ) as display:
        def on_progress(stage: str, current: int, total: int, stats) -> None:
            summary = ", ".join(f"{key}={value}" for key, value in stats.items())
            display_total = max(1, total)
            completed = current if total else 1
            if stage not in task_ids:
                task_ids[stage] = display.add_task(
                    STAGE_LABELS.get(stage, stage), total=display_total, summary=summary
                )
            display.update(
                task_ids[stage], total=display_total, completed=completed, summary=summary
            )

        if command == "run":
            return run_all(config, on_progress, reset=reset, force_rebuild=force_rebuild)
        if command == "build":
            return build_topology(config, on_progress, reset=reset, force_rebuild=force_rebuild)
        if command == "match":
            return match_clouds(config, on_progress)
        if command == "forecast":
            return forecast(config, on_progress)
        if command == "validate-data":
            return validate_dataset(config, on_progress)
        raise ValueError(f"不支持的进度命令：{command}")


def _preflight_options(command: str) -> dict[str, bool]:
    return {
        "require_source": command in {"preflight", "validate-data", "run", "build", "forecast"},
        "require_database": command in {"match", "forecast", "report", "status"},
        "require_topology_backend": command in {"preflight", "run", "build", "match"},
    }


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    try:
        config = PipelineConfig.from_json(args.config)
        report = inspect_environment(config, **_preflight_options(args.command))
        if args.json and args.command == "preflight":
            print(json.dumps(asdict(report), ensure_ascii=False, indent=2))
            if not report.ready:
                raise SystemExit(2)
            return
        if not args.json:
            render_preflight(report)
        ensure_ready(report)
        if args.command == "preflight":
            return

        if args.command in {"run", "build", "match", "forecast", "validate-data"}:
            reset = getattr(args, "reset", False)
            force_rebuild = getattr(args, "force_rebuild", False)
            result = _run_with_progress(
                args.command, config, reset=reset, force_rebuild=force_rebuild
            ) if not args.json else {
                "run": lambda: run_all(config, reset=reset, force_rebuild=force_rebuild),
                "build": lambda: build_topology(config, reset=reset, force_rebuild=force_rebuild),
                "match": lambda: match_clouds(config),
                "forecast": lambda: forecast(config),
                "validate-data": lambda: validate_dataset(config),
            }[args.command]()
        elif args.command == "report":
            result = generate_outputs(config)
        else:
            result = inspect_results(config)

        if args.json:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        elif args.command == "status":
            render_status(result)
        else:
            render_action_result(result)
            if args.command != "validate-data":
                render_status(inspect_results(config))
    except SystemExit:
        raise
    except Exception as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        else:
            CONSOLE.print(Panel(str(exc), title="启动失败", border_style="red"))
            CONSOLE.print("[dim]请先运行 preflight，按错误提示修正配置或环境。[/dim]")
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
