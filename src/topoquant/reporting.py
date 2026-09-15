from __future__ import annotations

import csv
import json
from collections import defaultdict
from contextlib import closing
from datetime import datetime

from .config import PipelineConfig
from .data import DataError
from .storage import connect, get_metadata


def generate_outputs(config: PipelineConfig) -> dict[str, int]:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    with closing(connect(config.database_path)) as db:
        if get_metadata(db, "forecast_signature") != config.forecast_signature():
            raise DataError("预测配置已变化或预测阶段尚未完整结束，请先执行 forecast")
        match_rows = db.execute(
            """
            SELECT m.target_id, m.similar_id, m.rank, m.distance_dim0, m.distance_dim1,
                   r.qualified_count
            FROM matches m JOIN match_runs r ON r.target_id=m.target_id
            ORDER BY m.target_id, m.rank
            """
        ).fetchall()
        with (config.output_dir / "selected_matches.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["target_id", "similar_id", "rank", "distance_dim0", "distance_dim1", "qualified_count"])
            writer.writerows([tuple(row) for row in match_rows])

        forecast_rows = db.execute(
            """
            SELECT target_id, horizon, target_date, actual_difference, actual_log_return,
                   actual_direction,
                   predicted_direction, vote_up, vote_count, correct
            FROM forecasts ORDER BY target_id, horizon
            """
        ).fetchall()
        headers = [
            "target_id", "horizon", "target_date", "actual_difference", "actual_log_return",
            "strategy_log_return", "actual_direction", "predicted_direction", "vote_up",
            "vote_count", "correct",
        ]
        with (config.output_dir / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerows([
                (
                    row["target_id"],
                    row["horizon"],
                    row["target_date"],
                    row["actual_difference"],
                    row["actual_log_return"],
                    float(row["actual_log_return"])
                    * (1.0 if int(row["predicted_direction"]) == 1 else -1.0),
                    row["actual_direction"],
                    row["predicted_direction"],
                    row["vote_up"],
                    row["vote_count"],
                    row["correct"],
                )
                for row in forecast_rows
            ])

        by_day: dict[int, dict[str, int | float]] = defaultdict(
            lambda: {"correct": 0, "total": 0, "log_return_sum": 0.0}
        )
        for row in forecast_rows:
            day = int(row["horizon"])
            by_day[day]["correct"] += int(row["correct"])
            by_day[day]["total"] += 1
            position = 1.0 if int(row["predicted_direction"]) == 1 else -1.0
            by_day[day]["log_return_sum"] += position * float(row["actual_log_return"])
        total_correct = sum(int(item["correct"]) for item in by_day.values())
        total = sum(int(item["total"]) for item in by_day.values())
        total_log_return = sum(float(item["log_return_sum"]) for item in by_day.values())
        metrics = {
            "as_of_date": config.as_of_date.isoformat(),
            "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "target_count": len({str(row["target_id"]) for row in forecast_rows}),
            "by_horizon": {
                f"d{day}": {
                    "correct": int(values["correct"]),
                    "total": int(values["total"]),
                    "accuracy": values["correct"] / values["total"] if values["total"] else None,
                    "log_return": values["log_return_sum"] / values["total"] if values["total"] else None,
                }
                for day, values in sorted(by_day.items())
            },
            "overall": {
                "correct": total_correct,
                "total": total,
                "accuracy": total_correct / total if total else None,
                "log_return": total_log_return / total if total else None,
            },
        }
        (config.output_dir / "metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        lines = [
            "预测结果统计分析报告",
            "=" * 72,
            f"实验截止日: {config.as_of_date.isoformat()}",
            f"成功预测目标数: {metrics['target_count']}",
            "",
            "按预测跨度统计：",
        ]
        for name, item in metrics["by_horizon"].items():
            accuracy = item["accuracy"]
            rendered = "无数据" if accuracy is None else f"{accuracy:.2%}"
            log_return = item["log_return"]
            rendered_return = "无数据" if log_return is None else f"{log_return:.4%}"
            lines.append(
                f"{name}: {item['correct']}/{item['total']}，准确率 {rendered}，"
                f"平均策略对数收益率 {rendered_return}"
            )
        overall = metrics["overall"]
        rendered_overall = "无数据" if overall["accuracy"] is None else f"{overall['accuracy']:.2%}"
        rendered_overall_return = (
            "无数据" if overall["log_return"] is None else f"{overall['log_return']:.4%}"
        )
        lines.extend([
            "",
            f"整体: {overall['correct']}/{overall['total']}，准确率 {rendered_overall}，"
            f"平均策略对数收益率 {rendered_overall_return}",
        ])
        (config.output_dir / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        return {"matches": len(match_rows), "predictions": len(forecast_rows)}
