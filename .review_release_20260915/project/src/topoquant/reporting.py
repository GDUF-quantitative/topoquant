from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Callable, Mapping
from contextlib import closing
from datetime import datetime

from .config import PipelineConfig
from .data import DataError
from .storage import connect, get_stage_status
from .tabular import write_table

# 与 build/forecast 同口径的进度回调类型别名（generate_outputs 注解使用）。
ProgressCallback = Callable[[str, int, int, Mapping[str, int]], None]

# 输入参数在 report.txt / metrics.json 中的展示顺序与中文标签。
# 顺序固定（不随 serializable() 字段顺序浮动），便于人眼逐行核对与跨 run 对比。
_PARAM_DISPLAY: list[tuple[str, str]] = [
    ("as_of_date", "实验截止日"),
    ("source_dir", "行情源目录"),
    ("work_dir", "工作目录"),
    ("work_root", "实验根目录"),
    ("window_size", "窗口大小"),
    ("lookback_trading_days", "回看交易日"),
    ("min_windows", "最少窗口数"),
    ("features", "特征"),
    ("max_edge_length", "最大边长"),
    ("distance_dimensions", "距离维度"),
    ("distance_threshold_h0", "距离阈值 H0"),
    ("distance_threshold_h1", "距离阈值 H1"),
    ("top_k", "相似点云保留数"),
    ("forecast_horizon", "预测跨度"),
    ("data_quality_mode", "数据质量模式"),
    ("topology_backend", "拓扑后端"),
    ("matching_pivots", "匹配 pivot 数"),
    ("topology_workers", "拓扑并发(配置)"),
    ("matching_workers", "匹配并发(配置)"),
    ("forecast_workers", "预测并发(配置)"),
    ("max_homology_dimension", "最高同调维度"),
    ("resolved_workers", "并发(解析后)"),
]


def _format_param_value(value: object) -> str:
    """把配置值渲染为单行可读文本（列表/字典展开，其余转 str）。"""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    if isinstance(value, dict):
        return ", ".join(f"{key}={val}" for key, val in value.items())
    return str(value)


def _render_params_block(params: dict) -> list[str]:
    """把全量输入参数渲染为 report.txt 的可读区块（固定顺序，跳过缺失键）。"""
    lines = ["全部输入参数：", "-" * 72]
    for key, label in _PARAM_DISPLAY:
        if key not in params:
            continue
        lines.append(f"  {label} ({key}): {_format_param_value(params[key])}")
    return lines


def generate_outputs(
    config: PipelineConfig, progress: "ProgressCallback | None" = None
) -> dict[str, int]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with closing(connect(config.database_path)) as db:
        forecast_status, stored_forecast_signature = get_stage_status(db, "forecast")
        if (
            stored_forecast_signature != config.forecast_signature()
            or forecast_status != "completed"
        ):
            raise DataError("预测配置已变化或预测阶段尚未完整结束，请先执行 forecast")
        match_rows = db.execute(
            """
            SELECT m.target_id, m.similar_id, m.rank, m.distance_dim0, m.distance_dim1,
                   r.qualified_count
            FROM matches m JOIN match_runs r ON r.target_id=m.target_id
            ORDER BY m.target_id, m.rank
            """
        ).fetchall()
        # 结果表（一）：已选相似匹配。经统一出口写出为 CSV（utf-8-sig），列顺序稳定。
        write_table(
            config.output_dir / "selected_matches.csv",
            header=[
                "target_id",
                "similar_id",
                "rank",
                "distance_dim0",
                "distance_dim1",
                "qualified_count",
            ],
            rows=[tuple(row) for row in match_rows],
        )

        forecast_rows = db.execute(
            """
            SELECT target_id, horizon, target_date, actual_difference, actual_direction,
                   predicted_direction, vote_up, vote_count, correct
            FROM forecasts ORDER BY target_id, horizon
            """
        ).fetchall()
        total_forecast_rows = len(forecast_rows)
        headers = [
            "target_id",
            "horizon",
            "target_date",
            "actual_difference",
            "actual_direction",
            "predicted_direction",
            "vote_up",
            "vote_count",
            "correct",
        ]
        # 结果表（二）：预测明细。O(目标数×预测天数) 单次顺序遍历，一边累计逐日准确率
        # （by_day）、一边收集 CSV 行，并周期性回弹进度——避免预测天数很大时整段同步落盘
        # 无任何进度反馈、GUI 进度条冻结在「行情预测 100%」被误判为卡死/阻塞。
        by_day: dict[int, dict[str, int]] = defaultdict(lambda: {"correct": 0, "total": 0})
        csv_rows: list[tuple[object, ...]] = []
        if progress is not None:
            progress("report", 0, total_forecast_rows or 1, {"status": "读取预测明细"})
        for i, row in enumerate(forecast_rows):
            day = int(row["horizon"])
            by_day[day]["correct"] += int(row["correct"])
            by_day[day]["total"] += 1
            csv_rows.append(tuple(row))
            if progress is not None and (i + 1) % 5000 == 0:
                progress("report", i + 1, total_forecast_rows or 1, {"status": "汇总预测明细"})
        if progress is not None and total_forecast_rows:
            progress("report", total_forecast_rows, total_forecast_rows, {"status": "写入预测明细 CSV"})
        # 结果表（二）：预测明细。经统一出口写出为 CSV（utf-8-sig），列顺序稳定。
        write_table(
            config.output_dir / "predictions.csv",
            header=headers,
            rows=csv_rows,
        )
        if progress is not None and total_forecast_rows:
            progress("report", total_forecast_rows, total_forecast_rows, {"status": "生成统计报告"})
        total_correct = sum(item["correct"] for item in by_day.values())
        total = sum(item["total"] for item in by_day.values())
        # 全量输入参数快照（含路径/窗口/特征/阈值/worker/后端等），供 report 与 metrics 复用。
        params = config.serializable()
        metrics = {
            "top_k": config.top_k,
            "as_of_date": config.as_of_date.isoformat(),
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "target_count": len({str(row["target_id"]) for row in forecast_rows}),
            "params": params,
            "by_horizon": {
                f"d{day}": {
                    **values,
                    "accuracy": values["correct"] / values["total"] if values["total"] else None,
                }
                for day, values in sorted(by_day.items())
            },
            "overall": {
                "correct": total_correct,
                "total": total,
                "accuracy": total_correct / total if total else None,
            },
        }
        (config.output_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        lines = [
            "预测结果统计分析报告",
            "=" * 72,
            f"实验截止日: {config.as_of_date.isoformat()}",
            f"top_k（相似点云保留数）: {config.top_k}",
            f"成功预测目标数: {metrics['target_count']}",
            "",
            *_render_params_block(params),
            "",
            "按预测跨度统计：",
        ]
        for name, item in metrics["by_horizon"].items():
            accuracy = item["accuracy"]
            rendered = "无数据" if accuracy is None else f"{accuracy:.2%}"
            lines.append(f"{name}: {item['correct']}/{item['total']}，准确率 {rendered}")
        overall = metrics["overall"]
        rendered_overall = "无数据" if overall["accuracy"] is None else f"{overall['accuracy']:.2%}"
        lines.extend(
            ["", f"整体: {overall['correct']}/{overall['total']}，准确率 {rendered_overall}"]
        )
        (config.output_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"matches": len(match_rows), "predictions": len(forecast_rows)}
