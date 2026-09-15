"""源数据/匹配/预测签名簇（Phase 3 机械搬迁，零逻辑变更）。

从 ``pipeline.py`` 抽出，仅做位置搬迁 + ``pipeline.py`` 重导出（branch-by-abstraction），
函数体逐字节不变。INV-2（签名键）由全量 V2 套件间接覆盖（matching_resume_key 依赖 DB 内容）。
"""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing

from .config import PipelineConfig, _distance_algo_label, _sha256_signature
from .storage import connect


def _exportable_diagram_rows(config: PipelineConfig) -> list[sqlite3.Row]:
    """返回满足「可导出」条件的点云行列表（按 cloud_id 排序）。

    「可导出」定义：cloud 在 clouds 中 status='complete'，且其 diagrams 在 ``config``
    所需的全部维度（``0..max_homology_dimension``）上均存在
    （即 ``COUNT(DISTINCT dimension) == 维度数``）。

    两个消费者共用此查询以避免 SQL 重复：
    - ``_count_exportable_diagrams`` 取其行数；
    - ``_compute_source_signature`` 在返回行上做聚合生成签名。
    """
    dimensions = tuple(range(config.max_homology_dimension + 1))
    placeholders = ",".join("?" for _ in dimensions)
    with closing(connect(config.database_path)) as db:
        return db.execute(
            f"""
            SELECT d.cloud_id, SUM(d.pair_count) AS pc
            FROM diagrams d JOIN clouds c ON c.cloud_id = d.cloud_id
            WHERE c.status = 'complete' AND d.dimension IN ({placeholders})
            GROUP BY d.cloud_id
            HAVING COUNT(DISTINCT d.dimension) = ?
            ORDER BY d.cloud_id
            """,
            (*dimensions, len(dimensions)),
        ).fetchall()


def _count_exportable_diagrams(config: PipelineConfig) -> int:
    """统计数据库中可导出的持久图数量，用于判断 mmap 文件是否过期。"""
    return len(_exportable_diagram_rows(config))


def _compute_source_signature(config: PipelineConfig) -> str:
    """计算源持久图集合的廉价签名，用于判断既有 mmap 文件是否仍然有效。

    仅做 SQL 聚合，不把持久图 BLOB 载入内存，开销为毫秒级。签名由「可导出点云数 +
    持久对总数 + 所有 cloud_id 的排序哈希」组成：只要源数据的成员或内容发生任何变化，
    签名即改变，从而触发重导出；否则可安全复用既有文件，避免冗余重导出。

    注意：**不做记忆化（无 @cache）**——签名直接反映当前数据库内容。函数入参
    ``PipelineConfig`` 虽可哈希（frozen），但其值不随库内持久图变化；若加 @cache，
    同 config 在库变更后再次调用会返回陈旧签名，导致续算键误判（误以为源未变、跳过
    重导出）。此处每次调用都实时查库，确保「库变 → 签名变」严格成立（模式 F）。
    """
    rows = _exportable_diagram_rows(config)
    cloud_count = len(rows)
    total_pairs = sum(int(row["pc"]) for row in rows)
    id_blob = "\n".join(row["cloud_id"] for row in rows).encode("utf-8")
    id_hash = hashlib.sha256(id_blob).hexdigest()[:16]
    return f"{cloud_count}:{total_pairs}:{id_hash}"


def matching_resume_key(config: PipelineConfig) -> str:
    """匹配阶段续算键：仅反映匹配真正消费的输入（图内容 + 距离参数）。

    用于 ``match_clouds`` / ``forecast`` 判定既有匹配结果是否仍有效，替代原先
    把整段 ``topology_signature``（含 ``source_dir`` / ``as_of_date`` 等无关项）
    卷入的 ``matching_signature``（步骤 B）。

    - 图内容：复用既有 ``_compute_source_signature``（mmap 内容寻址），仅当持久图
      字节真正变化时键才变；挪动 source_dir、改不影响匹配输入的拓扑参数（mmap 字节
      一致）都不会让键变化，从而复用既有距离结果。
    - 距离参数：``distance_dimensions`` / ``distance_threshold_h0`` / ``distance_threshold_h1`` /
      ``distance_algo``。
    - **不含** ``top_k``（步骤 A：top_k 只影响保留几条，不改瓶颈距离）；
      **不含** ``source_dir`` / ``as_of_date`` 等不影响匹配输入的拓扑参数。
    - 该键依赖数据库（图内容签名由 diagrams 聚合得出），故仅在 match/forecast 阶段、
      库已就绪时计算；目录派生（``match_segment``）现已剔除 ``source_dir``、与续算键
      彻底解耦——段名仅随距离参数走，内容复用由本续算键（mmap 内容签名 + 距离参数）
      负责。两者各司其职，挪动 source_dir（内容不变）时目录稳定、续算键不变，复用生效。
    """
    fields = {
        "source_signature": _compute_source_signature(config),
        "distance_dimensions": config.distance_dimensions,
        "distance_threshold_h0": config.distance_threshold_h0,
        "distance_threshold_h1": config.distance_threshold_h1,
        "distance_algo": _distance_algo_label(config.topology_backend),
    }
    return _sha256_signature(fields)


def _pivot_signature(
    config: PipelineConfig,
    dimensions: tuple[int, int],
    candidate_ids: tuple[str, ...],
    pivot_count: int,
) -> str:
    candidate_hash = hashlib.sha256("\n".join(candidate_ids).encode("utf-8")).hexdigest()[:16]
    return (
        f"{_compute_source_signature(config)}:pivot-v3:"
        f"dims={dimensions}:candidates={candidate_hash}:count={pivot_count}"
    )


def _stage_signature(config: PipelineConfig, stage: str) -> str | None:
    """返回某阶段用于幂等判定的签名；须与 stage_runs.signature 写入时保持一致。"""
    if stage == "topology":
        return config.topology_signature()
    if stage == "matching":
        return matching_resume_key(config)
    if stage == "forecast":
        return config.forecast_signature()
    return None
