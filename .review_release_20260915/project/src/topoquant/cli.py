from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import os
import sys
from dataclasses import asdict
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Column, Table

from ._logging import configure_logging
from ._progress_log import ProgressWriter, WorkerLogHub
from .config import PipelineConfig
from .pipeline import build_topology, forecast, match_clouds, run_all
from .preflight import PreflightReport, ensure_ready, inspect_environment, inspect_results
from .reporting import generate_outputs
from .validation import validate_dataset

CONSOLE = Console()
# JSON 模式专用：日志走 stderr，避免污染 stdout 上的机器可读 JSON（见 run_pipeline_from_config）。
CONSOLE_ERR = Console(stderr=True)
STAGE_LABELS = {
    "topology": "持续同调",
    "matching": "瓶颈匹配",
    "forecast": "行情预测",
    "validation": "数据验证",
    # 以下四项对应「原先长时间静默、与挂死无法区分」的环节（本次新增可见化）。
    # _run_with_progress.on_progress 对任何未登记的 stage 也会自动建条，
    # 这里登记只是为了给出中文标题，便于用户一眼看懂当前卡在哪一步。
    "export": "导出持久图",  # mmap 持久图导出主循环（数万点云的 BLOB 读写）
    "export_wait": "等待导出锁",  # 跨进程 .mmap_export.lock 排队（最长 30 分钟）
    "pivot": "pivot 距离缓存",  # 候选 × pivot 精确瓶颈距离预计算
    "pivot_wait": "等待 pivot 锁",  # 同一把导出锁，pivot 缓存构建前的排队
    "load_clouds": "加载点云记录",  # 从 SQLite 读取全部点云元数据（静默读取阶段）
    "audit": "审计表头",  # 预检表头审计进度（N/total），单行刷新显示
    "report": "结果输出",  # 预测结果落盘（selected_matches/predictions CSV + metrics/report）
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
    parser.add_argument(
        "--strict-preflight",
        dest="strict_preflight",
        action="store_true",
        help="预检更严格：build 产物缺失等直接报错而非仅警告（P3-11）",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight", help="检测运行环境并展示实际并发计划")
    subparsers.add_parser("validate-data", help="完整扫描行情日期、数值和可用窗口")
    run_sub = subparsers.add_parser("run", help="预检通过后依次执行全部阶段")
    build_sub = subparsers.add_parser("build", help="预检后生成点云并计算持续同调")
    for sub in (run_sub, build_sub):
        sub.add_argument(
            "--reset",
            action="store_true",
            help="清空实验库与 mmap 中间产物后从头重算（换/改股票数据时用）",
        )
        sub.add_argument(
            "--force-rebuild",
            dest="force_rebuild",
            action="store_true",
            help="忽略“行情/配置已变化”保护，就地清空旧数据并重算（授权混合实验）",
        )
        sub.add_argument(
            "--force-content-hash",
            dest="force_content_hash",
            action="store_true",
            help="跳过源签名的两阶段元数据缓存，强制对全部行情文件重算内容哈希（P1-3）",
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
        "可用 "
        + _format_bytes(report.available_memory_bytes)
        + " / 总计 "
        + _format_bytes(report.total_memory_bytes),
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
        plan.add_row(
            stage.stage,
            stage.mode,
            "自动" if stage.requested == 0 else str(stage.requested),
            str(stage.workers),
        )
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
    if (
        isinstance(result, dict)
        and result
        and all(isinstance(value, dict) for value in result.values())
    ):
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
    force_content_hash: bool = False,
    live: bool | None = None,
    progress=None,
    progress_writer: "ProgressWriter | None" = None,
    log_queue=None,
) -> object:
    # live 默认按 TTY 自适应：TTY（真实控制台/独立窗口）用 rich Progress 原地重绘；
    # 非 TTY（GUI capture 模式走 PIPE、或 --json 之外的重定向）改用纯文本逐行进度，
    # 否则 rich Live 在 non-tty 下要么不渲染、要么反复重打完整块，表现为「黑屏无反应」。
    #
    # 关键修复（2026-08-23）：经 launcher 的独立运行框（wt new-tab → cmd /c → wrapper
    # → 子进程）启动时，控制台能力探测不稳定——部分 Windows/WT 组合下 isatty() 返回 True、
    # 部分返回 False。返回 True 时 rich 走 SpinnerColumn 逐帧动画，其 \x1b[2K+\r 原地重绘
    # 在 WT 标签 + mode con 改宽后发生竞态，清行序列丢失导致 spinner 字符与进度文字逐行
    # 向下堆叠（rich issue #2691/#1024/#3182 统称的「抽风/闪屏」）。
    # 统一由 wrapper 注入 TDA_FORCE_PLAIN_PROGRESS=1，强制走纯文本逐行分支，从根上消除
    # 任何光标重绘竞态——独立运行框只需「可滚动审查进度」，不需要动画。
    #
    # 纯文本分支（2026-08-24）：进度一律用 ProgressWriter 回车覆盖式单行刷新（无 ANSI、
    # Windows 控制台兼容）；若调用方已构造 progress（如 run_pipeline_from_config 统一
    # 建立的 writer 回调）则直接复用，审计表头与各阶段进度共用同一单行。
    force_plain = os.environ.get("TDA_FORCE_PLAIN_PROGRESS") == "1"
    if live is None:
        live = sys.stdout.isatty() and not force_plain
    task_ids: dict[str, int] = {}
    if live:
        return _run_with_live(
            command, config, reset, force_rebuild, force_content_hash
        )

    # ── 纯文本进度（GUI 日志框 / 独立运行框实时可见，单行刷新） ──
    if progress is None:
        last_stage: dict[str, int] = {}

        def on_progress(stage: str, current: int, total: int, stats) -> None:
            summary = ", ".join(f"{key}={value}" for key, value in stats.items())
            desc = STAGE_LABELS.get(stage, stage)
            if progress_writer is not None:
                # 回车覆盖式单行刷新：审计表头 N/5303、各阶段进度均覆盖同一行。
                progress_writer.update(desc, current, total, summary)
                return
            # 无 writer 兜底：沿用 rich 纯文本逐行（CLI 子命令等非 GUI 路径）。
            if total:
                pct = current * 100 // total
                if stage not in last_stage or pct - last_stage[stage] >= 5 or current >= total:
                    CONSOLE.print(f"[cyan]{desc}[/] {current}/{total} ({pct}%) {summary}")
                    last_stage[stage] = pct
            else:
                CONSOLE.print(f"[cyan]{desc}[/] {summary}")
    else:
        # 调用方已提供进度回调（如 run_pipeline_from_config 统一构造的 writer 单行刷新回调），
        # 直接复用，避免 on_progress 在 progress is None 分支外未绑定导致 UnboundLocalError。
        on_progress = progress

    if command == "run":
        return run_all(
            config,
            on_progress,
            reset=reset,
            force_rebuild=force_rebuild,
            force_content_hash=force_content_hash,
            log_queue=log_queue,
        )
    if command == "build":
        return build_topology(
            config,
            on_progress,
            reset=reset,
            force_rebuild=force_rebuild,
            force_content_hash=force_content_hash,
            log_queue=log_queue,
        )
    if command == "match":
        return match_clouds(config, on_progress, log_queue=log_queue)
    if command == "forecast":
        return forecast(config, on_progress)
    if command == "validate-data":
        return validate_dataset(config, on_progress)
    raise ValueError(f"不支持的进度命令：{command}")


def _preflight_options(
    command: str, *, strict_preflight: bool = False, config: "PipelineConfig | None" = None
) -> dict[str, bool]:
    # gudhi 只是可选 [gudhi] extra（对拍/兜底），默认后端 ripser_topp 与 native_c_dll
    # 均不依赖它。仅当显式选择 gudhi 后端时才强制要求，避免对非 gudhi 后端误杀
    # （修复之前的硬编码：command in {...} 无条件 require_gudhi，见重构评审 R8-1）。
    need_gudhi = command in {"preflight", "run", "build", "match"} and (
        config is not None and getattr(config, "topology_backend", "ripser_topp") == "gudhi"
    )
    return {
        "require_source": command in {"preflight", "validate-data", "run", "build", "forecast"},
        "require_database": command in {"match", "forecast", "report", "status"},
        "require_gudhi": need_gudhi,
        # match/forecast 依赖 build 产出的 mmap 持久图；strict 时缺失即报错，否则仅警告（P3-11）。
        "require_build_artifacts": command in {"match", "forecast"},
        "strict_build_artifacts": strict_preflight,
    }


def _run_with_live(
    command: str,
    config: PipelineConfig,
    reset: bool = False,
    force_rebuild: bool = False,
    force_content_hash: bool = False,
) -> object:
    """TTY 模式：rich Progress Live 原地重绘进度条（真实控制台/独立窗口用）。"""
    task_ids: dict[str, int] = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeElapsedColumn(),
        # summary 文本随计数位数变长，无宽度限制时会在 80 列终端折行，
        # 导致 Live 的行数估算失准、屏幕撕裂；限宽 + 省略号是零风险修复。
        TextColumn(
            "{task.fields[summary]}",
            table_column=Column(no_wrap=True, overflow="ellipsis", width=48),
        ),
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
            # 阶段跑满后停掉该任务：spinner 不再空转，run_all 三阶段也不会
            # 一直堆在 Live 区里。finished_time 守卫避免重复 stop_task。
            if total and current >= total and display.tasks[task_ids[stage]].finished_time is None:
                display.stop_task(task_ids[stage])

        if command == "run":
            return run_all(
                config,
                on_progress,
                reset=reset,
                force_rebuild=force_rebuild,
                force_content_hash=force_content_hash,
            )
        if command == "build":
            return build_topology(
                config,
                on_progress,
                reset=reset,
                force_rebuild=force_rebuild,
                force_content_hash=force_content_hash,
            )
        if command == "match":
            return match_clouds(config, on_progress)
        if command == "forecast":
            return forecast(config, on_progress)
        if command == "validate-data":
            return validate_dataset(config, on_progress)
        raise ValueError(f"不支持的进度命令：{command}")


def run_pipeline_from_config(  # noqa: PLR0913,PLR0917
    config: str | Path | PipelineConfig,
    reset: bool = False,
    force_rebuild: bool = False,
    json_output: bool = False,
    strict_preflight: bool = False,
    force_content_hash: bool = False,
    preflight_report: PreflightReport | None = None,
) -> None:
    """从配置文件执行完整流水线（``run`` 命令的底层实现，无 argparse 依赖）。

    等价于 CLI ``run`` 子命令的全部行为：加载配置 → 预启动检测 → 确保就绪
    → 带进度条执行全部阶段（持续同调 / 瓶颈匹配 / 行情预测）→ 渲染结果。
    run.py 在交互式收集参数并写入 config.json、完成自身健康检查后委托本函数；
    CLI ``main`` 的 ``run`` 分支也直接调用本函数。

    ``reset`` / ``force_rebuild`` 透传给流水线（对应 CLI ``--reset`` / ``--force-rebuild``）；
    ``json_output`` 为 True 时只打印 JSON 结果（对应 CLI ``--json``）。
    """
    # 从 run.py 交互入口进来时不会经过 main() 的 basicConfig，根 logger 没有任何 handler，
    # 各 stage 的 LOGGER.info/warning（进度、超时告警、看门狗）会被整段丢弃——一旦卡死
    # 就完全看不出卡在哪个阶段。这里补一次配置，并用 handlers 守卫避免与 main() 重复配置。
    # JSON 模式：日志改走 stderr（CONSOLE_ERR），保证 stdout 只含最终 JSON（契约见文档）。
    configure_logging("rich", level=logging.INFO, console=CONSOLE_ERR if json_output else CONSOLE)
    # 多进程日志合并（2026-08-24）：主进程建一个 spawn 上下文的队列 + WorkerLogHub，
    # worker 经 QueueHandler 把 Python 日志上送此处去重打印（stderr），避免 8 个 worker
    # 的并行日志流在窗口里交错/重复；C++ 级噪声仍由 worker 端 fd→devnull 封死。
    # hub 必须在任何 worker 拉起前启动，并在本函数退出前停止（finally）。
    log_queue = mp.get_context("spawn").Queue()
    hub = WorkerLogHub(log_queue)
    hub.start()
    try:
        # 复用调用方已构造的 config 对象（run.py / cli main 已 from_json 过），避免对
        # source_dir 重复做全量内容哈希——该段代价高昂，且两次计算间文件变化会引入
        # work_dir TOCTOU 与实例锁错位（见风险评审）。传入路径则仍走 from_json。
        # 无论来源，validate() 幂等兜底，保证进入流水线前参数一定合法（避免对象来源绕过校验）。
        if not isinstance(config, PipelineConfig):
            config = PipelineConfig.from_json(config)
        config.validate()
        # 早期阶段标记：import 重包 + 拓扑后端（topp C++）加载已完成，进入预检/就绪阶段。
        # 这条日志能在 GUI 日志框（capture 模式）第一时间出现，缩短「启动后黑屏无反应」的空白感知——
        # 否则 inspect_environment 审计数千个 CSV 期间没有任何输出，用户会以为卡死。
        if not json_output:
            CONSOLE.print("[dim]配置已加载，开始预检数据集与就绪检查…[/]")
        # 统一进度回调：审计表头与各阶段进度共用同一个 ProgressWriter 单行刷新。
        writer = ProgressWriter(sys.stdout) if not json_output else None

        def _on_progress(stage: str, current: int, total: int, stats) -> None:
            if writer is None:
                return
            summary = ", ".join(f"{key}={value}" for key, value in stats.items())
            writer.update(STAGE_LABELS.get(stage, stage), current, total, summary)

        # O-8：复用调用方已算好的预检 report（run.py 交互路径在 _run_prelaunch_checks 已算过），
        # 避免对 source_dir 重复审计；CLI run 路径不预计算 → 此处仍自算。
        # 复用 run.py 的 report 还顺带把 gudhi 要求改为后端感知（修复 cli 对非 gudhi 后端
        # 无条件要求 gudhi 的潜在误杀，见重构评审 R8-1）。
        if preflight_report is not None:
            report = preflight_report
        else:
            report = inspect_environment(
                config,
                progress=_on_progress,
                **_preflight_options(
                    "run", strict_preflight=strict_preflight, config=config
                ),
            )
        ensure_ready(report)
        # 审计进度收尾换行，避免与阶段横幅挤在同一行。
        if writer is not None:
            writer.finish()
        # O-5：阶段横幅（仅人看模式；JSON 模式跳过以保 stdout 仅含最终 JSON 契约）。
        # 横幅仅在主进程打印一次（worker 端日志已改走队列，不可能重复）。
        if not json_output:
            CONSOLE.print("[bold cyan]运行流水线：持续同调 → 瓶颈匹配 → 行情预测[/]")
        # json 模式与基线 CLI --json 一致：直接调用 run_all 返回结果（不显示动态进度条）；
        # hub 仍后台排干 worker 日志，避免队列积压导致 worker 阻塞。
        if json_output:
            result = run_all(
                config,
                reset=reset,
                force_rebuild=force_rebuild,
                force_content_hash=force_content_hash,
                log_queue=log_queue,
            )
        else:
            result = _run_with_progress(
                "run",
                config,
                reset=reset,
                force_rebuild=force_rebuild,
                force_content_hash=force_content_hash,
                progress=_on_progress,
                progress_writer=writer,
                log_queue=log_queue,
            )
            writer.finish()
    finally:
        hub.stop()
    if json_output:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        render_action_result(result)
        render_status(inspect_results(config))


# ── 命令派发映射 ───────────────────────────────────────
# 将各子命令的「动作」收敛为统一签名的处理器，消除 main() 中重复的
# if/elif 派发与 json 模式下的内联 lambda 字典（见 A2/B8）。
# 仅包含「产出 result 对象、交由下方统一渲染」的标准命令；
# preflight / run 因输出形态特殊，仍在 main() 顶部单独处理。


def _cmd_build(
    config: PipelineConfig,
    *,
    reset: bool,
    force_rebuild: bool,
    json_output: bool,
    force_content_hash: bool = False,
):
    if json_output:
        return build_topology(
            config,
            reset=reset,
            force_rebuild=force_rebuild,
            force_content_hash=force_content_hash,
        )
    return _run_with_progress(
        "build",
        config,
        reset=reset,
        force_rebuild=force_rebuild,
        force_content_hash=force_content_hash,
    )


def _cmd_match(
    config: PipelineConfig,
    *,
    reset: bool,
    force_rebuild: bool,
    json_output: bool,
    force_content_hash: bool = False,
):
    if json_output:
        return match_clouds(config)
    return _run_with_progress("match", config)


def _cmd_forecast(
    config: PipelineConfig,
    *,
    reset: bool,
    force_rebuild: bool,
    json_output: bool,
    force_content_hash: bool = False,
):
    if json_output:
        return forecast(config)
    return _run_with_progress("forecast", config)


def _cmd_validate_data(
    config: PipelineConfig,
    *,
    reset: bool,
    force_rebuild: bool,
    json_output: bool,
    force_content_hash: bool = False,
):
    if json_output:
        return validate_dataset(config)
    return _run_with_progress("validate-data", config)


def _cmd_report(
    config: PipelineConfig,
    *,
    reset: bool,
    force_rebuild: bool,
    json_output: bool,
    force_content_hash: bool = False,
):
    return generate_outputs(config)


def _cmd_status(
    config: PipelineConfig,
    *,
    reset: bool,
    force_rebuild: bool,
    json_output: bool,
    force_content_hash: bool = False,
):
    return inspect_results(config)


COMMAND_FUNCS: dict[str, object] = {
    "build": _cmd_build,
    "match": _cmd_match,
    "forecast": _cmd_forecast,
    "validate-data": _cmd_validate_data,
    "report": _cmd_report,
    "status": _cmd_status,
}


def main() -> None:
    args = build_parser().parse_args()
    # JSON 模式日志走 stderr，避免污染 stdout 上的 JSON 结果。
    configure_logging(
        "rich",
        level=logging.INFO if args.verbose else logging.WARNING,
        console=CONSOLE_ERR if args.json else CONSOLE,
    )
    try:
        config = PipelineConfig.from_json(args.config)
        report = inspect_environment(
            config, **_preflight_options(
                args.command, strict_preflight=args.strict_preflight, config=config
            )
        )
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

        # run 命令的完整流程收敛到 run_pipeline_from_config（见 A2），避免与 run.py 重复实现。
        if args.command == "run":
            _run_command(config, args)
            return

        # 其余标准命令统一经 COMMAND_FUNCS 派发（preflight / run 已在上方单独处理）。
        action = COMMAND_FUNCS[args.command]
        reset = getattr(args, "reset", False)
        force_rebuild = getattr(args, "force_rebuild", False)
        force_content_hash = getattr(args, "force_content_hash", False)
        result = action(
            config,
            reset=reset,
            force_rebuild=force_rebuild,
            json_output=args.json,
            force_content_hash=force_content_hash,
        )
        _render_result(args, result, config)
    except SystemExit:
        raise
    except Exception as exc:
        if args.json:
            print(json.dumps({"error": str(exc)}, ensure_ascii=False))
        else:
            CONSOLE.print(Panel(str(exc), title="启动失败", border_style="red"))
            CONSOLE.print("[dim]请先运行 preflight，按错误提示修正配置或环境。[/dim]")
        raise SystemExit(2) from exc


def _run_command(config: PipelineConfig, args: argparse.Namespace) -> None:
    """main 的 run 命令分支（M4：提取自 main，逻辑逐字平移，含 strict_preflight 透传）。"""
    reset = getattr(args, "reset", False)
    force_rebuild = getattr(args, "force_rebuild", False)
    force_content_hash = getattr(args, "force_content_hash", False)
    run_pipeline_from_config(
        config,
        reset=reset,
        force_rebuild=force_rebuild,
        json_output=args.json,
        strict_preflight=args.strict_preflight,
        force_content_hash=force_content_hash,
    )


def _render_result(args: argparse.Namespace, result: object, config: PipelineConfig) -> None:
    """main 的结果渲染分支（M4：提取自 main，逻辑逐字平移）。"""
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    elif args.command == "status":
        render_status(result)
    else:
        render_action_result(result)
        if args.command != "validate-data":
            render_status(inspect_results(config))


if __name__ == "__main__":
    main()
