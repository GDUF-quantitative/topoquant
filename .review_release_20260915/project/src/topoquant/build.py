"""拓扑构建阶段原语（Phase 3 从 god-module 抽取）。

机械搬迁（byte-for-byte）：原 ``pipeline.py`` 中的 build worker 初始化
（_init_build_worker）、单股构建（_build_stock）、拓扑阶段编排（build_topology）
与 --reset 清理（_reset_experiment）平移至此；``pipeline.py`` 以
``from .build import (...)`` 重导出，外部 ``pipeline.build_topology`` 等引用零破坏。
可变全局 ``_BUILD_CONFIG`` / ``_BUILD_COMPLETED`` 随迁（仅 build 簇使用，单一真相）。
函数体逐字节一致，无循环依赖（仅依赖 data/db_utils/domain/mmap_io/_logging/
persistence/policy/pooling/storage/topology 与标准库）。
"""

from __future__ import annotations

import datetime as dt
import logging
import os
import socket
import time
from collections.abc import Callable, Mapping
from contextlib import closing
from pathlib import Path

import numpy as np

from ._logging import configure_logging
from .config import PipelineConfig
from .data import (
    file_content_signature,
    iter_cloud_windows,
    list_stock_files,
    standardized_points,
    stock_quality_skip_reason,
)
from .db_utils import (
    _clear_incomplete_marker,
    _commit_with_retry,
    _write_incomplete_marker,
)
from .domain import CloudRecord, Diagram
from .mmap_io import (
    DIAGRAM_SIGNATURE_FILENAME,
    _diagram_mmap_paths,
    _export_diagrams_to_mmap,
    _progress,
)
from .persistence import CommitBatch
from .policy import POLICY
from .pooling import _run_pool_stage
from .storage import (
    check_or_set_identity,
    connect,
    diagram_complete,
    save_cloud,
    save_cloud_error,
    upsert_stage_run,
)
from .topology import compute_persistence, set_topology_backend

LOGGER = logging.getLogger(__name__)

ProgressCallback = Callable[[str, int, int, Mapping[str, int]], None]

_BUILD_CONFIG: PipelineConfig | None = None
_BUILD_COMPLETED: set[str] = set()


def _init_build_worker(
    config: PipelineConfig,
    completed_cloud_ids: set[str],
    log_queue=None,
) -> None:
    global _BUILD_CONFIG, _BUILD_COMPLETED
    _BUILD_CONFIG = config
    _BUILD_COMPLETED = completed_cloud_ids
    configure_logging("worker-silent", log_queue=log_queue)


def _build_stock(source_path_text: str) -> dict[str, object]:
    if _BUILD_CONFIG is None:
        raise RuntimeError("持续同调工作进程尚未初始化")
    source_path = Path(source_path_text)
    counts = {"complete": 0, "skipped": 0, "error": 0, "stock_error": 0, "data_quality_skipped": 0}
    completed: list[tuple[CloudRecord, Diagram]] = []
    errors: list[tuple[str, object, str, Path, str]] = []
    try:
        # P2-2 / R2：构建前对单文件快扫，命中问题则整支股票跳过（不计为 error，单独计数
        # 以便审计）。与改动前的差异：不再用 `if data_quality_mode == "strict"` 前置守卫，
        # 两种模式都会调用；由 stock_quality_skip_reason 内部分级决定是否返回原因——
        # 「重复交易日」等硬伤始终拦截（否则会在 forecast 阶段硬崩），「非数值行」等软伤
        # 仍只在 strict 下拦截。data_quality_mode 默认值保持 permissive 不变。
        reason = stock_quality_skip_reason(source_path, _BUILD_CONFIG)
        if reason is not None:
            counts["data_quality_skipped"] += 1
            level = (
                "严格数据质量模式" if _BUILD_CONFIG.data_quality_mode == "strict" else "数据硬伤"
            )
            LOGGER.warning("%s跳过 %s：%s", level, source_path.name, reason)
            return {"counts": counts, "completed": completed, "errors": errors, "fatal": None}
        for window in iter_cloud_windows(_BUILD_CONFIG, [source_path]):
            if window.cloud_id in _BUILD_COMPLETED:
                counts["skipped"] += 1
                continue
            try:
                points = standardized_points(window, _BUILD_CONFIG.features)
                if not np.isfinite(points).all():
                    # 云级有限性守卫（方案 B）：点云含 NaN/±inf 会让持久同调进入
                    # 未定义行为并静默污染下游；此处显式跳过，与 :355 硬伤拦截一致。
                    counts["data_quality_skipped"] += 1
                    LOGGER.warning(
                        "数据质量跳过 %s：标准化点云含非有限坐标(NaN/±inf)，"
                        "可能源自整列全 inf 等脏数据，已跳过以避免静默污染持久图",
                        window.cloud_id,
                    )
                    continue
                diagram = compute_persistence(
                    points,
                    _BUILD_CONFIG.max_edge_length,
                    _BUILD_CONFIG.max_homology_dimension,
                )
                completed.append(
                    (
                        CloudRecord(
                            window.cloud_id,
                            window.cloud_date,
                            window.stock_code,
                            window.source_path,
                            len(points),
                        ),
                        diagram,
                    )
                )
                counts["complete"] += 1
            except Exception as exc:
                errors.append(
                    (
                        window.cloud_id,
                        window.cloud_date,
                        window.stock_code,
                        window.source_path,
                        str(exc),
                    )
                )
                counts["error"] += 1
    except Exception as exc:
        counts["stock_error"] += 1
        return {"counts": counts, "completed": completed, "errors": errors, "fatal": str(exc)}
    return {"counts": counts, "completed": completed, "errors": errors, "fatal": None}


def build_topology(  # noqa: PLR0915
    config: PipelineConfig,
    progress: ProgressCallback | None = None,
    reset: bool = False,
    force_rebuild: bool = False,
    force_content_hash: bool = False,
    log_queue=None,
) -> dict[str, int]:
    set_topology_backend(config.topology_backend)
    reset_failed: list[Path] = []
    if reset:
        reset_failed = _reset_experiment(config)
    identity_force = force_rebuild or bool(reset_failed)
    if reset and not reset_failed and force_rebuild:
        LOGGER.debug("--reset 已物理清空实验库，--force-rebuild 本次无需生效")
    files = list_stock_files(config.source_dir)
    config.work_dir.mkdir(parents=True, exist_ok=True)
    # R8：阶段起始 best-effort 清除上一轮可能残留的「完成标记写入失败」告警标记；
    # 若本轮确实未落盘，仍会在失败时再次落 marker，不影响正确性。
    _clear_incomplete_marker(config.work_dir, "topology")
    LOGGER.info("工作目录（按参数派生）: %s", config.work_dir)
    with closing(connect(config.database_path)) as db:
        # 源文件内容签名：冷缓存 / 首次运行 / 参数变更（work_dir 随之变化）时需对全部源
        # CSV 做流式 SHA256，是「工作目录打印后、持续同调进度条出现前」那段静默停顿的主因。
        # 提前打印状态日志消除假死观感；计算本身不变（仅透出耗时）。
        _sig_t0 = time.time()
        LOGGER.info(
            "正在计算源数据内容签名（%d 个文件；源未变更则命中缓存秒回，"
            "否则需全量哈希，请稍候）…", len(files))
        source_sig = file_content_signature(
            files,
            force_content_hash=force_content_hash,
            cache_path=config.work_dir / "source_signature_cache.json",
        )
        LOGGER.info("源数据内容签名完成，耗时 %.1fs", time.time() - _sig_t0)
        try:
            check_or_set_identity(
                db,
                config.topology_signature(),
                source_sig,
                config.serializable(),
                force=identity_force,
            )
        except Exception as exc:
            # R8：完成标记落库失败时在文件系统写告警标记（旁路信号，原样上抛、不吞首因）。
            _write_incomplete_marker(config.work_dir, "topology", exc)
            raise
        expected_dimensions = config.max_homology_dimension + 1
        all_ids = {
            str(row["cloud_id"]) for row in db.execute("SELECT cloud_id FROM diagrams").fetchall()
        }
        completed_cloud_ids = {
            cloud_id for cloud_id in all_ids if diagram_complete(db, cloud_id, expected_dimensions)
        }
        # R9：明确播报持久图缓存命中情况，避免「明明算过却全 completed、skipped 恒为 0」的困惑。
        # skip 的唯一闸门是 diagram_complete：要求该 cloud 的 diagrams 表恰好有 expected_dimensions
        # 行。跳过数为 0 的常见原因：当前 work_dir 数据库无完整持久图（H9 卡死未落库 / 维度不匹配
        # / 改动任一参数派生出新 work_dir），这些股票会被重新计算并记为 completed。
        LOGGER.info(
            "持久图缓存状态：可跳过 %d / 源文件 %d（expected_dimensions=%d）。"
            "跳过数为 0 说明当前 work_dir 无完整持久图（空库/维度不匹配/参数变更派生新 work_dir），"
            "这些将全量重算并记为 completed。",
            len(completed_cloud_ids), len(files), expected_dimensions,
        )
        worker_count = config.resolved_topology_workers
        counts = {"complete": 0, "skipped": 0, "error": 0, "stock_error": 0}
        # 数据硬伤跳过数单独用局部变量累计（不并入 counts，保持返回结构与
        # 进度条 summary 口径不变），循环结束后在父进程汇总打印一条——
        # worker 内的逐支 LOGGER.warning 已被 NullHandler 静默，诊断不能丢。
        data_quality_skipped = 0
        _progress(progress, "topology", 0, len(files), {**counts, "workers": worker_count})

        def _consume_stock(stock_index: int, result: object) -> None:
            """原 ``for stock_index, result in enumerate(results, 1):`` 的整段循环体。

            逐字保留：counts 聚合方式、save_cloud / save_cloud_error 调用、
            fatal 日志、``stock_index % 25`` 的 commit 节奏、_progress 调用点。
            抽成函数只是为了让串行分支与并行分支共用同一份逻辑。
            """
            nonlocal data_quality_skipped
            result_counts = result["counts"]
            for key in counts:
                counts[key] += int(result_counts[key])
            data_quality_skipped += int(result_counts.get("data_quality_skipped", 0))
            for record, diagram in result["completed"]:
                save_cloud(db, record, diagram)
            for cloud_id, cloud_date, stock_code, source_path, error in result["errors"]:
                save_cloud_error(db, cloud_id, cloud_date, stock_code, source_path, error)
                LOGGER.warning("点云 %s 处理失败：%s", cloud_id, error)
            if result["fatal"]:
                LOGGER.warning(
                    "行情文件 %s 处理失败：%s", files[stock_index - 1].name, result["fatal"]
                )
            # R2-S3：提交节奏收敛到 CommitBatch（每 commit_every_topology 支一次 + 退出时尾批补提交），  # noqa: E501
            # 不再散落 ``stock_index % 25`` 魔法数；落库韧性（退避重试）由 CommitBatch 内部沿用。
            _commit_batch.tick()
            # 进度日志仍按原 25 支节奏输出（与提交节奏一致）；已有 rich 进度条时降级 DEBUG，
            # 无进度条（headless / --json 模式）保留 INFO，避免完全看不到进度。
            if stock_index % POLICY.commit_every_topology == 0:
                if progress is None:
                    LOGGER.info("持续同调进度：%d/%d 支股票，%s", stock_index, len(files), counts)
                else:
                    LOGGER.debug("持续同调进度：%d/%d 支股票，%s", stock_index, len(files), counts)
            _progress(
                progress,
                "topology",
                stock_index,
                len(files),
                {**counts, "workers": worker_count},
            )

        def _fail_stock(stock_index: int, label: str, reason: str, timed_out: bool) -> None:
            """超时/异常导致整支股票没有结果时的记账。

            计数口径与 ``_build_stock`` 既有的 fatal 分支保持一致：整支股票失败记
            ``stock_error``；同时按 P0 要求记一次 ``error``，保证「超时不会被静默吞掉」。
            这里同样推进一次 ``_progress``，否则进度条会永远停在失败的那一项上。
            """
            counts["error"] += 1
            counts["stock_error"] += 1
            LOGGER.error("行情文件 %s 未产出结果（%s），已记为错误并继续后续股票", label, reason)
            _progress(
                progress,
                "topology",
                stock_index,
                len(files),
                {**counts, "workers": worker_count},
            )

        _stage_now = dt.datetime.now(dt.timezone.utc).isoformat()
        upsert_stage_run(
            db,
            stage="topology",
            status="running",
            signature=config.topology_signature(),
            pid=os.getpid(),
            host=socket.gethostname(),
            started_at=_stage_now,
            updated_at=_stage_now,
        )
        _init_build_worker(config, completed_cloud_ids)
        with CommitBatch(
            db, every=POLICY.commit_every_topology, what="持续同调中间结果"
        ) as _commit_batch:
            if worker_count == 1:
                results = map(_build_stock, (str(path) for path in files))
                for stock_index, result in enumerate(results, 1):
                    # 与多 worker 路径（_run_pool_stage 的 consume 隔离）保持一致：
                    # 落盘异常不再穿透整个阶段，而是记错并继续后续股票，阶段末仍能
                    # 正常 commit + 导出 mmap，避免单支股票的数据库争抢拖垮整轮 build
                    # （与已修的 match 续算死循环同源风险）。
                    try:
                        _consume_stock(stock_index, result)
                    except Exception as exc:
                        LOGGER.exception(
                            "持续同调结果落盘失败：%s（第 %d/%d 项）：%s，记为错误并继续后续股票",
                            files[stock_index - 1].name,
                            stock_index,
                            len(files),
                            exc,
                        )
                        _fail_stock(
                            stock_index, files[stock_index - 1].name, f"结果落盘失败：{exc}", False
                        )
            else:
                _run_pool_stage(
                    stage_label="持续同调",
                    worker_count=worker_count,
                    initializer=_init_build_worker,
                    initargs=(config, completed_cloud_ids, log_queue),
                    task_func=_build_stock,
                    items=[str(path) for path in files],
                    labels=[path.name for path in files],
                    consume=_consume_stock,
                    on_failure=_fail_stock,
                )
        if data_quality_skipped:
            # 父进程汇总（RichHandler 接管，干净上滚）；逐支原因仍可从
            # counts["data_quality_skipped"] 与 worker 返回的 errors 明细审计。
            LOGGER.warning("数据硬伤跳过 %d 支股票", data_quality_skipped)
        upsert_stage_run(
            db,
            stage="topology",
            status="completed",
            signature=config.topology_signature(),
            pid=os.getpid(),
            host=socket.gethostname(),
            started_at=_stage_now,
            updated_at=dt.datetime.now(dt.timezone.utc).isoformat(),
        )
        _commit_with_retry(db, what="持续同调阶段末提交")
        counts["workers"] = worker_count

    # 在 SQLite 连接关闭之后再导出，避免与写连接争抢文件锁。
    LOGGER.info("导出 mmap 持久图，供匹配阶段零拷贝共享……")
    # 透传 progress：导出数万个点云的持久图是 build 阶段收尾的一段重 IO，
    # 原先全程无反馈，现在以 export 进度条实时可见。
    counts["mmap_diagrams"] = len(_export_diagrams_to_mmap(config, progress))
    return counts


def _reset_experiment(config: PipelineConfig) -> list[Path]:
    """清空实验库与 mmap 中间产物，回到全新状态（``--reset`` 使用）。

    返回删除失败的路径列表；空列表表示全部产物已成功清空。删除动作本身与成功路径
    行为**完全不变**——仅把原先被 ``pass`` 吞掉的失败改为可见（记 ERROR 并收集），
    使「日志说清空了、其实没清」的误导不再发生（仍不抛异常，保持 reset 尽力而为语义）。

    直接删除实验库文件（含 WAL/SHM）与所有 mmap 产物；下次 ``connect`` 会按
    当前 schema 重建空库。与 ``--force-rebuild`` 不同，这里是“物理删除从头来过”。

    注意：``source_signature_cache.json``（R1 源文件指纹侧车）**刻意保留**。它只是
    「文件元数据 → 内容哈希」的纯缓存，与实验结果无关；保留它可让 reset 后的重算
    仍免去一次全量 sha256，且不会让任何旧结果混入新数据。
    """
    failed: list[Path] = []
    db_path = config.database_path
    for path in (db_path, Path(str(db_path) + "-wal"), Path(str(db_path) + "-shm")):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            failed.append(path)
            LOGGER.error("--reset 无法删除 %s：%s", path, exc)  # R6：不再静默 pass
    index_path, count_path, _ = _diagram_mmap_paths(config)
    # 用 glob 而非当前配置的维度列表：改过 distance_dimensions 后，工作目录里可能
    # 残留上一份配置导出的 diagrams_h*.npy，--reset 应当把它们一并清掉（P1-1）。
    stale_dimension_files = sorted(config.work_dir.glob("diagrams_h*.npy"))
    for path in (
        index_path,
        count_path,
        *stale_dimension_files,
        config.work_dir / DIAGRAM_SIGNATURE_FILENAME,
    ):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            failed.append(path)
            LOGGER.error("--reset 无法删除 %s：%s", path, exc)  # R6：不再静默 pass
    if failed:
        LOGGER.warning("--reset：部分产物删除失败（%d 项），将回退为清空数据行", len(failed))
    else:
        LOGGER.warning("--reset：实验库与 mmap 中间产物已清空，将从头重算")
    return failed
