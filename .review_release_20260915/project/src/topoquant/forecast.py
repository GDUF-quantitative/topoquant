"""预测阶段编排（Phase 3 从 god-module 抽取）。

机械搬迁（byte-for-byte）：原 ``pipeline.py`` 中的 ``forecast`` 编排函数平移至此；
``pipeline.py`` 以 ``from .forecast import forecast`` 重导出，外部 ``pipeline.forecast``
引用零破坏（cli.py / tests 均经 pipeline 访问）。函数体逐字节一致，无循环依赖
（仅依赖 config/data/db_utils/mmap_io/persistence/policy/signatures/storage/topology
与标准库）。
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import socket
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, suppress
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import PipelineConfig
from .data import DataError, load_future_directions
from .db_utils import _clear_incomplete_marker, _commit_with_retry, _write_incomplete_marker
from .mmap_io import _progress
from .persistence import CommitBatch
from .policy import POLICY
from .signatures import matching_resume_key
from .storage import (
    connect,
    get_stage_status,
    load_cloud_records,
    save_forecast_error,
    save_forecasts,
    upsert_stage_run,
)
from .topology import set_topology_backend

LOGGER = logging.getLogger(__name__)

# 与 pipeline.py 同口径的进度回调类型别名（forecast 签名注解使用）。
ProgressCallback = Callable[[str, int, int, Mapping[str, int]], None]


def forecast(  # noqa: PLR0912,PLR0915
    config: PipelineConfig,
    progress: ProgressCallback | None = None,
) -> dict[str, int]:
    set_topology_backend(config.topology_backend)
    _clear_incomplete_marker(config.work_dir, "forecast")
    with closing(connect(config.database_path)) as db:
        # 守卫键与 match_clouds 的续算键一致（matching_resume_key）：图内容 + 距离参数。
        # 仅改 top_k 时键不变，forecast 可直接复用既有匹配结果（步骤 A）。
        # 状态统一由 stage_runs 承载（get_stage_status 仅读该表；旧库经迁移回填）。
        _, stored_signature = get_stage_status(db, "matching")
        if stored_signature != matching_resume_key(config):
            raise DataError("匹配参数已变化或尚未运行 match，请先重新执行匹配阶段")
        _progress(progress, "load_clouds", 0, 1, {"status": "读取点云元数据"})
        records = load_cloud_records(db)
        _progress(
            progress,
            "load_clouds",
            1,
            1,
            {"status": f"已读 {len(records)} 个点云"},
        )
        # ── 断点续算（与 match_clouds 同构）──
        # forecast_runs(status) 表本就存在且 save_forecasts 落盘时写 status='complete'，
        # 但此前从不读它做续算，导致每次都全量重算。这里补齐：
        #   · 仅当预测参数签名（forecast_signature：horizon/top_k/源数据口径等）不变时，
        #     才跳过既已完成（status='complete'）的目标，只补算缺失项；
        #   · 预测参数变化 → forecast_signature 改变 → 不复用、全量重算，
        #     由 save_forecasts 的 UPSERT 幂等覆盖，不留陈旧行。
        all_target_rows = db.execute(
            "SELECT target_id FROM match_runs WHERE status='selected' ORDER BY target_id"
        ).fetchall()
        all_target_ids = [str(row["target_id"]) for row in all_target_rows]
        _, stored_forecast_signature = get_stage_status(db, "forecast")
        current_forecast_signature = config.forecast_signature()
        resume_forecast = (
            bool(stored_forecast_signature)
            and stored_forecast_signature == current_forecast_signature
        )
        done_forecast_ids: set[str] = set()
        if resume_forecast:
            done_forecast_ids = {
                str(row["target_id"])
                for row in db.execute("SELECT target_id FROM forecast_runs WHERE status='complete'")
            }
            if done_forecast_ids:
                LOGGER.info(
                    "复用既有预测结果，跳过 %d 个已完成目标的重新预测（仅补算缺失项）",
                    len(done_forecast_ids),
                )
        pending_target_ids = [tid for tid in all_target_ids if tid not in done_forecast_ids]

        _forecast_started_at = dt.datetime.now(dt.timezone.utc).isoformat()
        upsert_stage_run(
            db,
            stage="forecast",
            status="running",
            signature=config.forecast_signature(),
            pid=os.getpid(),
            host=socket.gethostname(),
            started_at=_forecast_started_at,
            updated_at=_forecast_started_at,
        )

        worker_count = config.resolved_forecast_workers
        _progress(
            progress,
            "forecast",
            0,
            len(pending_target_ids),
            {"complete": 0, "error": 0, "workers": worker_count},
        )

        tasks: list[tuple[str, list[str]]] = []
        for target_id in pending_target_ids:
            match_rows = db.execute(
                "SELECT similar_id FROM matches WHERE target_id=? ORDER BY rank", (target_id,)
            ).fetchall()
            tasks.append((target_id, [str(row["similar_id"]) for row in match_rows]))

        # 有界缓存：未来行情方向按 (源文件, 截止日) 复用；默认上限 4096 足以覆盖
        # 全部目标×候选组合，同时避免无界缓存长期占用内存（P3-7）。
        @lru_cache(maxsize=4096)
        def future(source_path: Path, cloud_date: object):
            return load_future_directions(source_path, cloud_date, config.forecast_horizon)

        def forecast_one(task: tuple[str, list[str]]):
            target_id, similar_ids = task
            try:
                if len(similar_ids) != config.top_k:
                    raise DataError(f"匹配数为 {len(similar_ids)}，期望 {config.top_k}")
                target_record = records[target_id]
                # 目标自身的未来行情缺失属于硬错误：没有 actual 就无法预测，直接失败。
                target_dates, target_diffs, target_directions = future(
                    target_record.source_path, target_record.cloud_date
                )
                # 逐邻居容错（P1-2）：单个候选邻居的未来行情读取失败，不应拖累整次预测。
                # 收集有效邻居的方向向量，跳过失败的邻居继续投票；最终以 valid_k 个有效票
                # 降级投票（多数决阈值随 valid_k 调整），仅当所有邻居都不可用时才失败。
                valid_directions: list[np.ndarray] = []
                failed_reasons: list[str] = []
                for item in similar_ids:
                    try:
                        _, _, direction = future(
                            records[item].source_path, records[item].cloud_date
                        )
                        valid_directions.append(direction)
                    except Exception as exc:  # 单邻居坏：记录原因并跳过，不中断整体
                        failed_reasons.append(f"{item}: {exc}")
                valid_k = len(valid_directions)
                if valid_k == 0:
                    raise DataError(
                        "所有候选邻居均无可用未来行情，无法投票：" + "; ".join(failed_reasons)
                    )
                if failed_reasons:
                    LOGGER.warning(
                        "目标 %s 有 %d/%d 个候选邻居被跳过，降级为 %d 票投票：%s",
                        target_id,
                        len(failed_reasons),
                        len(similar_ids),
                        valid_k,
                        "; ".join(failed_reasons),
                    )
                analog_directions = np.vstack(valid_directions)
                # 多数决阈值随有效票数浮动，而非固定 top_k：降级后仍是合法多数决（P1-2）。
                majority = valid_k // 2 + 1
                vote_up = analog_directions.sum(axis=0)
                predictions = (vote_up >= majority).astype(np.int8)
                rows: list[tuple[object, ...]] = []
                for horizon in range(config.forecast_horizon):
                    actual = int(target_directions[horizon])
                    predicted = int(predictions[horizon])
                    rows.append(
                        (
                            horizon + 1,
                            target_dates[horizon].isoformat(),
                            float(target_diffs[horizon]),
                            actual,
                            predicted,
                            int(vote_up[horizon]),
                            valid_k,  # 实际有效票数（降级后可能 < top_k），如实记录
                            int(actual == predicted),
                        )
                    )
                return target_id, rows, None
            except Exception as exc:
                return target_id, [], str(exc)

        if worker_count == 1:
            results = map(forecast_one, tasks)
            executor = None
        else:
            executor = ThreadPoolExecutor(max_workers=worker_count)
            results = executor.map(forecast_one, tasks)

        complete = 0
        errors = 0
        with CommitBatch(
            db, every=POLICY.commit_every_forecast, what="预测中间结果"
        ) as _forecast_batch:
            try:
                for index, (target_id, rows, error) in enumerate(results, 1):
                    # 单目标落盘隔离：任一步骤（save / commit）失败都只记为错误并继续，
                    # 不再穿透整个预测阶段（与 match_clouds 的 _fail_target、build 的
                    # 单 worker 隔离同源改造）。compute 阶段的失败已由 forecast_one 内部
                    # 捕获为 error 返回，此处仅处理持久化失败。
                    try:
                        if error is None:
                            save_forecasts(db, target_id, rows)
                            complete += 1
                        else:
                            save_forecast_error(db, target_id, error)
                            errors += 1
                            LOGGER.warning("目标 %s 预测失败：%s", target_id, error)
                    except Exception as exc:
                        errors += 1
                        LOGGER.exception(
                            "目标 %s 预测结果落盘失败：%s，记为错误并继续后续目标", target_id, exc
                        )
                        try:  # noqa: SIM105
                            save_forecast_error(db, target_id, f"结果落盘失败：{exc}")
                        except Exception:  # 落盘错误也写不进，仅记录，不二次抛出  # noqa: S110
                            pass
                    # R2-S3：提交节奏收敛到 CommitBatch（每 commit_every_forecast 项一次 + 退出时尾批补提交），  # noqa: E501
                    # 锁争抢自愈语义不变。
                    _forecast_batch.tick()
                    _progress(
                        progress,
                        "forecast",
                        index,
                        len(pending_target_ids),
                        {"complete": complete, "error": errors, "workers": worker_count},
                    )
            finally:
                if executor is not None:
                    executor.shutdown()
        # 阶段末落盘签名：同样走带重试提交，并包裹异常——避免末尾 commit 撞锁时
        # 丢弃已完成结果（与 match_clouds 的 finally 守卫一致）。签名写不进则下次
        # 运行会因签名未落地而重算全部目标，属可接受的降级而非数据丢失。
        try:
            upsert_stage_run(
                db,
                stage="forecast",
                status="completed",
                signature=config.forecast_signature(),
                pid=os.getpid(),
                host=socket.gethostname(),
                started_at=_forecast_started_at,
                updated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
            )
            _commit_with_retry(db, what="预测阶段完成标记")
        except Exception as exc:
            LOGGER.error(
                "写入预测完成标记失败：%s（已算出的预测结果仍在库中，"
                "下次运行会因签名未落地而重算全部目标）",
                exc,
            )
            # 显式落一个告警标记，使「库不可写」这一严重状态可被监控/运维捕获
            # （仅写文件系统，不依赖 DB；失败则忽略，不影响主流程）。
            with suppress(Exception):
                _write_incomplete_marker(config.work_dir, "forecast", exc)
        return {
            "selected": len(all_target_ids),
            "complete": complete,
            "error": errors,
            "resumed": len(done_forecast_ids),
            "workers": worker_count,
        }
