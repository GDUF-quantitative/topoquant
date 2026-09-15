"""实验库持久化层（SQLite / WAL）。

提交契约（调用方必读）
----------------------
本模块各写入函数的 commit 行为分两类，调用前务必确认由谁负责落盘，
漏一次 commit 会静默丢结果：

* 自行 commit（函数内已 ``db.commit()``，调用方 **不要** 再提交）：
    - ``clear_experiment_data``
    - ``check_or_set_identity``
* 不 commit（仅执行语句，依赖调用方在合适时机提交）：
    - ``save_cloud``、``save_cloud_error``、``save_matches``、
      ``save_match_candidates``、``save_forecast_error``、
      ``save_forecasts``、``set_metadata``

上述「不 commit」类函数首行 docstring 均标注
「本函数不 commit，由调用方负责」；「自行 commit」类函数
首行标注「本函数自行 commit」。
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from .domain import CloudRecord, Diagram

LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = "3"

_MIGRATIONS: dict[str, Callable[[sqlite3.Connection], None]] = {}
# 键为「起始版本」，值为把库从该版本升到下一版的函数（每步只做加法迁移：
# CREATE TABLE IF NOT EXISTS / ALTER TABLE ADD COLUMN）。


class StorageError(RuntimeError):
    """实验库状态与当前运行不兼容。"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    # 跨进程写自动退避 30s，避免 SQLITE_BUSY 立即抛错。
    # 决策登记册 #1：指令原建议 5000，此处**有意偏离**取 30000——
    # 本库为 WAL 单写派生缓存库（无并发写竞争），30s 等待不损吞吐；
    # 而 5000 在写入高峰仍可能偶发 BUSY 导致部分写，与指令 §3.3 自身将
    # SQLITE_BUSY 列为风险一致，故取生产推荐区间(5000–30000ms)上沿。
    # 如需对齐指令：将 30000 改回 5000 即可（低风险）。
    connection.execute("PRAGMA busy_timeout = 30000")
    create_schema(connection)
    return connection


@contextmanager
def _savepoint(db: sqlite3.Connection, name: str) -> Iterator[None]:
    """在 db 当前事务内开一个命名 SAVEPOINT；函数体异常则回滚到该点，否则释放（模式 C-2）。

    用于「不自行 commit」的写函数：保证函数体内多条语句要么全成功、要么全撤销，
    避免中途异常留下部分写。若调用方已有外层事务，SAVEPOINT 嵌套其中，回滚只影响
    本函数区间内的语句，不影响调用方已写入的其它数据（与既有「调用方负责提交」契约兼容）。
    """
    db.execute(f"SAVEPOINT {name}")
    try:
        yield
    except Exception:
        db.execute(f"ROLLBACK TO {name}")
        db.execute(f"RELEASE {name}")
        raise
    else:
        db.execute(f"RELEASE {name}")


class Storage:
    """SQLite 连接的事务性包装（上下文管理器，模式 C-1）。

    进入时持有连接，退出时若发生异常自动 ``rollback()``；正常退出**不**自动提交
    （仍由调用方显式 ``commit()``，保持既有「不自行 commit」写入契约）。

    本类为**增量**引入：不替换既有自由函数式写入接口，仅供需要「退出即回滚」语义的
    调用方按需选用；自由函数式 ``save_*`` 仍由调用方负责提交。
    """

    def __init__(self, db: sqlite3.Connection) -> None:
        self._db = db

    @classmethod
    def open(cls, path: Path) -> Storage:
        return cls(connect(path))

    def __enter__(self) -> sqlite3.Connection:
        return self._db

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is not None:
            self._db.rollback()


def create_schema(db: sqlite3.Connection) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS clouds (
            cloud_id TEXT PRIMARY KEY,
            cloud_date TEXT NOT NULL,
            stock_code TEXT NOT NULL,
            source_path TEXT NOT NULL,
            point_count INTEGER NOT NULL,
            status TEXT NOT NULL,
            error TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_cloud_date ON clouds(cloud_date);
        CREATE TABLE IF NOT EXISTS diagrams (
            cloud_id TEXT NOT NULL REFERENCES clouds(cloud_id) ON DELETE CASCADE,
            dimension INTEGER NOT NULL,
            pair_count INTEGER NOT NULL,
            pairs BLOB NOT NULL,
            PRIMARY KEY (cloud_id, dimension)
        );
        CREATE TABLE IF NOT EXISTS match_runs (
            target_id TEXT PRIMARY KEY REFERENCES clouds(cloud_id) ON DELETE CASCADE,
            -- 剪枝后实际计算距离的候选数（仅供审计，不参与任何计算；字段名保持不变，见 P3-9）。
            candidate_count INTEGER NOT NULL,
            qualified_count INTEGER NOT NULL,
            status TEXT NOT NULL,
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS matches (
            target_id TEXT NOT NULL REFERENCES clouds(cloud_id) ON DELETE CASCADE,
            similar_id TEXT NOT NULL REFERENCES clouds(cloud_id) ON DELETE CASCADE,
            rank INTEGER NOT NULL,
            distance_dim0 REAL NOT NULL,
            distance_dim1 REAL NOT NULL,
            PRIMARY KEY (target_id, rank),
            UNIQUE (target_id, similar_id)
        );
        CREATE TABLE IF NOT EXISTS match_candidates (
            target_id TEXT NOT NULL REFERENCES clouds(cloud_id) ON DELETE CASCADE,
            similar_id TEXT NOT NULL REFERENCES clouds(cloud_id) ON DELETE CASCADE,
            rank INTEGER NOT NULL,
            distance_dim0 REAL NOT NULL,
            distance_dim1 REAL NOT NULL,
            PRIMARY KEY (target_id, rank),
            UNIQUE (target_id, similar_id)
        );
        CREATE TABLE IF NOT EXISTS forecast_runs (
            target_id TEXT PRIMARY KEY REFERENCES clouds(cloud_id) ON DELETE CASCADE,
            status TEXT NOT NULL,
            error TEXT
        );
        CREATE TABLE IF NOT EXISTS forecasts (
            target_id TEXT NOT NULL REFERENCES clouds(cloud_id) ON DELETE CASCADE,
            horizon INTEGER NOT NULL,
            target_date TEXT NOT NULL,
            actual_difference REAL NOT NULL,
            actual_direction INTEGER NOT NULL,
            predicted_direction INTEGER NOT NULL,
            vote_up INTEGER NOT NULL,
            vote_count INTEGER NOT NULL,
            correct INTEGER NOT NULL,
            PRIMARY KEY (target_id, horizon)
        );
        CREATE TABLE IF NOT EXISTS stage_runs (
            stage      TEXT PRIMARY KEY,
            status     TEXT NOT NULL,
            signature  TEXT NOT NULL,
            pid        INTEGER,
            host       TEXT,
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            error      TEXT
        );
        """
    )
    existing = get_metadata(db, "schema_version")
    if existing is not None and existing != SCHEMA_VERSION:
        # 版本不符时不立即硬失败，先尝试前向迁移；无迁移路径才上抛 StorageError。
        _migrate_schema(db, existing)
    set_metadata(db, "schema_version", SCHEMA_VERSION)
    db.commit()


def _migrate_schema(db: sqlite3.Connection, existing: str) -> None:
    """把既有库就地前向升级到 ``SCHEMA_VERSION``；不支持降级。

    每个迁移步骤内部只做加法（``CREATE TABLE IF NOT EXISTS`` / ``ALTER TABLE ADD COLUMN``），
    因此中途崩溃留下的半升级库可被重入式重跑。
    """
    cursor = existing
    while cursor != SCHEMA_VERSION:
        step = _MIGRATIONS.get(cursor)
        if step is None:
            raise StorageError(
                f"实验库版本 {existing} 无法升级到 {SCHEMA_VERSION}"
                f"（缺少 {cursor} → 下一版的迁移步骤）；请使用新的 work_dir"
            )
        step(db)
        cursor = get_metadata(db, "schema_version")


def _upgrade_v1_to_v2(db: sqlite3.Connection) -> None:
    """加法迁移：新增 ``stage_runs`` 状态表（仅 ``CREATE TABLE IF NOT EXISTS``）。"""
    db.execute(
        """
        CREATE TABLE IF NOT EXISTS stage_runs (
            stage      TEXT PRIMARY KEY,
            status     TEXT NOT NULL,
            signature  TEXT NOT NULL,
            pid        INTEGER,
            host       TEXT,
            started_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            error      TEXT
        )
        """
    )
    set_metadata(db, "schema_version", "2")


_MIGRATIONS["1"] = _upgrade_v1_to_v2


def _upgrade_v2_to_v3(db: sqlite3.Connection) -> None:
    """加法迁移：把 R1c 之前仅靠 legacy 元数据键（``matching_signature`` /
    ``forecast_signature``）承载的续算状态，收敛回填到权威表 ``stage_runs``。

    仅 ``INSERT`` 缺失行、绝不覆盖既有 ``stage_runs`` 行，因此可重入（中途崩溃的
    半升级库重跑本步骤安全）。旧库经此迁移后，``get_stage_status`` 不再依赖 legacy
    元数据键，为删除 matching 回退分支与 legacy 双写扫清障碍。
    """
    now = datetime.now(timezone.utc).isoformat()
    # matching：优先保留既有 stage_runs 行；无则按 legacy matching_signature 回填。
    # 运行中硬杀会残留 ``in_progress:`` 前缀，对应到 stage_runs 的 "running" 状态。
    if db.execute("SELECT 1 FROM stage_runs WHERE stage='matching'").fetchone() is None:
        legacy = get_metadata(db, "matching_signature")
        if legacy:
            prefix = "in_progress:"
            if legacy.startswith(prefix):
                status, signature = "running", legacy[len(prefix) :]
            else:
                status, signature = "completed", legacy
            db.execute(
                "INSERT INTO stage_runs(stage, status, signature, started_at, updated_at) "
                "VALUES ('matching', ?, ?, ?, ?)",
                (status, signature, now, now),
            )
    # forecast：同理，从 legacy forecast_signature 回填（forecast 无 in_progress 前缀）。
    if db.execute("SELECT 1 FROM stage_runs WHERE stage='forecast'").fetchone() is None:
        legacy_f = get_metadata(db, "forecast_signature")
        if legacy_f:
            db.execute(
                "INSERT INTO stage_runs(stage, status, signature, started_at, updated_at) "
                "VALUES ('forecast', 'completed', ?, ?, ?)",
                (legacy_f, now, now),
            )
    set_metadata(db, "schema_version", "3")


_MIGRATIONS["2"] = _upgrade_v2_to_v3


def get_metadata(db: sqlite3.Connection, key: str) -> str | None:
    row = db.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row["value"])


def set_metadata(db: sqlite3.Connection, key: str, value: str) -> None:
    """本函数不 commit，由调用方负责。写入一条 metadata 键值。"""
    db.execute(
        "INSERT INTO metadata(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def upsert_stage_run(
    db: sqlite3.Connection,
    *,
    stage: str,
    status: str,
    signature: str,
    pid: int | None = None,
    host: str | None = None,
    started_at: str,
    updated_at: str,
    error: str | None = None,
) -> None:
    """本函数不 commit，由调用方负责。写入/更新一条 ``stage_runs`` 阶段状态（双写过渡的一部分）。"""
    db.execute(
        "INSERT INTO stage_runs(stage, status, signature, pid, host, "
        "started_at, updated_at, error) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
        "ON CONFLICT(stage) DO UPDATE SET "
        "status=excluded.status, signature=excluded.signature, pid=excluded.pid, "
        "host=excluded.host, started_at=excluded.started_at, updated_at=excluded.updated_at, "
        "error=excluded.error",
        (stage, status, signature, pid, host, started_at, updated_at, error),
    )


def get_stage_status(db: sqlite3.Connection, stage: str) -> tuple[str | None, str | None]:
    """返回 ``(status, signature)``，唯一权威来源为 ``stage_runs``。

    旧库（R1c 之前仅以 legacy 元数据键承载续算状态）经 ``_upgrade_v2_to_v3`` 在
    ``connect`` 时就地回填 ``stage_runs``，故此处不再直接读 legacy 元数据键。
    """
    row = db.execute(
        "SELECT status, signature FROM stage_runs WHERE stage = ?", (stage,)
    ).fetchone()
    if row is not None:
        return row["status"], row["signature"]
    return None, None


def clear_experiment_data(db: sqlite3.Connection) -> None:
    """清空全部实验数据行（保留 metadata），用于强制重算。

    本函数**自行 commit**（函数内已 ``db.commit()``），调用方无需再提交。
    """
    for table in (
        "matches",
        "match_runs",
        "match_candidates",
        "forecasts",
        "forecast_runs",
        "diagrams",
        "clouds",
    ):
        db.execute(f"DELETE FROM {table}")
    db.commit()


def check_or_set_identity(
    db: sqlite3.Connection,
    topology_signature: str,
    source_signature: str,
    config_json: dict[str, object],
    force: bool = False,
) -> None:
    """校验并写入实验身份签名；本函数**自行 commit**（内部已 ``db.commit()``）。

    配置变更需 ``force=True`` 方清空旧数据；调用方无需再提交。
    """
    stored_topology = get_metadata(db, "topology_signature")
    stored_source = get_metadata(db, "source_signature")
    diagram_count = db.execute("SELECT COUNT(*) FROM diagrams").fetchone()[0]
    if diagram_count and stored_topology != topology_signature:
        if not force:
            raise StorageError("当前配置会改变点云或持续同调结果；请使用新的 work_dir")
        clear_experiment_data(db)
        LOGGER.warning("配置变更且已授权 --force-rebuild，已清空旧实验数据")
    if diagram_count and stored_source != source_signature:
        if not force:
            raise StorageError("原始行情文件已变化；为避免混合实验，请使用新的 work_dir")
        clear_experiment_data(db)
        LOGGER.warning("行情文件变更且已授权 --force-rebuild，已清空旧实验数据")
    set_metadata(db, "topology_signature", topology_signature)
    set_metadata(db, "source_signature", source_signature)
    set_metadata(db, "config", json.dumps(config_json, ensure_ascii=False, sort_keys=True))
    set_metadata(db, "updated_at", datetime.now(timezone.utc).isoformat())
    db.commit()


def diagram_complete(db: sqlite3.Connection, cloud_id: str, expected_dimensions: int) -> bool:
    row = db.execute(
        "SELECT COUNT(*) AS count FROM diagrams WHERE cloud_id = ?", (cloud_id,)
    ).fetchone()
    return int(row["count"]) == expected_dimensions


def save_cloud(db: sqlite3.Connection, record: CloudRecord, diagram: Diagram) -> None:
    """本函数不 commit，由调用方负责。写入单个点云及其持久图。"""
    with _savepoint(db, "sp_save_cloud"):
        db.execute(
            """
            INSERT INTO clouds(
                cloud_id, cloud_date, stock_code, source_path, point_count,
                status, error
            )
            VALUES (?, ?, ?, ?, ?, 'complete', NULL)
            ON CONFLICT(cloud_id) DO UPDATE SET
                cloud_date=excluded.cloud_date,
                stock_code=excluded.stock_code,
                source_path=excluded.source_path,
                point_count=excluded.point_count,
                status='complete', error=NULL
            """,
            (
                record.cloud_id,
                record.cloud_date.isoformat(),
                record.stock_code,
                str(record.source_path),
                record.point_count,
            ),
        )
        db.execute("DELETE FROM diagrams WHERE cloud_id = ?", (record.cloud_id,))
        for dimension, pairs in sorted(diagram.items()):
            contiguous = np.ascontiguousarray(pairs, dtype="<f8").reshape(-1, 2)
            db.execute(
                "INSERT INTO diagrams(cloud_id, dimension, pair_count, pairs) VALUES (?, ?, ?, ?)",
                (record.cloud_id, dimension, len(contiguous), contiguous.tobytes()),
            )


def save_cloud_error(
    db: sqlite3.Connection,
    cloud_id: str,
    cloud_date: date,
    stock_code: str,
    source_path: Path,
    error: str,
) -> None:
    """本函数不 commit，由调用方负责。记录点云处理失败。"""
    with _savepoint(db, "sp_save_cloud_error"):
        db.execute(
            """
            INSERT INTO clouds(
                cloud_id, cloud_date, stock_code, source_path, point_count,
                status, error
            )
            VALUES (?, ?, ?, ?, 0, 'error', ?)
            ON CONFLICT(cloud_id) DO UPDATE SET status='error', error=excluded.error
            """,
            (cloud_id, cloud_date.isoformat(), stock_code, str(source_path), error),
        )


def load_cloud_records(db: sqlite3.Connection) -> dict[str, CloudRecord]:
    rows = db.execute(
        "SELECT cloud_id, cloud_date, stock_code, source_path, point_count "
        "FROM clouds WHERE status='complete' ORDER BY cloud_id"
    )
    return {
        str(row["cloud_id"]): CloudRecord(
            cloud_id=str(row["cloud_id"]),
            cloud_date=date.fromisoformat(row["cloud_date"]),
            stock_code=str(row["stock_code"]),
            source_path=Path(row["source_path"]),
            point_count=int(row["point_count"]),
        )
        for row in rows
    }


def save_matches(
    db: sqlite3.Connection,
    target_id: str,
    candidate_count: int,
    qualified_count: int,
    matches: list[tuple[str, float, float]],
    selected: bool,
) -> None:
    """本函数不 commit，由调用方负责。

    落盘单个目标的匹配结果（截断后的 top_k 邻居）。

    ``selected`` 显式传入：选中判定（合格候选数 >= top_k）原本在 worker 内按 top_k
    截断时一并决定，步骤 A 把 top_k 移出距离计算后，该判定被延迟到落盘期、由调用方
    按 ``config.top_k`` 给出，从而改 top_k 不影响距离计算/缓存键。
    ``matches`` 仅含 top_k 条（与改动前语义一致），保证下游 forecast/report 消费口径不变。
    """
    status = "selected" if selected else "not_selected"
    with _savepoint(db, "sp_save_matches"):
        # 注意（隐藏耦合）：此处重写某目标的 matches 时，会**级联清空**该目标既有的
        # forecasts / forecast_runs。因此 match 阶段的续算（_reselect_completed_match_targets）
        # 一旦因 top_k 变化而重写 matches，该目标的预测结果即失效、必须由 forecast 阶段
        # 按新邻居集重算覆盖。调用方改写 matches 前应明确此副作用，避免误以为 match 续算
        # 对 forecast 完全透明。SAVEPOINT 保证「级联删除 + 重写」原子化：中途异常回滚整段，
        # 不留「删了 forecasts 却没写回 matches」的半截状态。
        db.execute("DELETE FROM forecasts WHERE target_id = ?", (target_id,))
        db.execute("DELETE FROM forecast_runs WHERE target_id = ?", (target_id,))
        db.execute("DELETE FROM matches WHERE target_id = ?", (target_id,))
        db.execute(
            """
            INSERT INTO match_runs(target_id, candidate_count, qualified_count, status, error)
            VALUES (?, ?, ?, ?, NULL)
            ON CONFLICT(target_id) DO UPDATE SET
                candidate_count=excluded.candidate_count,
                qualified_count=excluded.qualified_count,
                status=excluded.status, error=NULL
            """,
            (target_id, candidate_count, qualified_count, status),
        )
        db.executemany(
            "INSERT INTO matches(target_id, similar_id, rank, distance_dim0, distance_dim1) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (target_id, similar_id, rank, d0, d1)
                for rank, (similar_id, d0, d1) in enumerate(matches, 1)
            ],
        )


def save_match_candidates(
    db: sqlite3.Connection,
    target_id: str,
    candidates: list[tuple[str, float, float]],
) -> None:
    """本函数不 commit，由调用方负责。

    落盘单个目标的完整候选排序（按距离升序，至安全上限）。

    与 ``matches``（仅 top_k 截断）解耦：本表保留全量排序，供改 top_k 或挪动
    source_dir 后在 match 阶段廉价地「重新截断」复用既有距离结果，而无需重算瓶颈距离。
    属于新增表，forecast/report 不读取，故不破坏既有消费方（步骤 A/B 红线）。
    """
    with _savepoint(db, "sp_save_match_candidates"):
        db.execute("DELETE FROM match_candidates WHERE target_id = ?", (target_id,))
        db.executemany(
            "INSERT INTO match_candidates(target_id, similar_id, rank, "
            "distance_dim0, distance_dim1) "
            "VALUES (?, ?, ?, ?, ?)",
            [
                (target_id, similar_id, rank, d0, d1)
                for rank, (similar_id, d0, d1) in enumerate(candidates, 1)
            ],
        )


def save_forecast_error(db: sqlite3.Connection, target_id: str, error: str) -> None:
    """本函数不 commit，由调用方负责。记录目标预测失败。"""
    with _savepoint(db, "sp_save_forecast_error"):
        db.execute("DELETE FROM forecasts WHERE target_id = ?", (target_id,))
        db.execute(
            """
            INSERT INTO forecast_runs(target_id, status, error) VALUES (?, 'error', ?)
            ON CONFLICT(target_id) DO UPDATE SET status='error', error=excluded.error
            """,
            (target_id, error),
        )


def save_forecasts(db: sqlite3.Connection, target_id: str, rows: list[tuple[object, ...]]) -> None:
    """本函数不 commit，由调用方负责。写入单个目标的预测结果。"""
    with _savepoint(db, "sp_save_forecasts"):
        db.execute("DELETE FROM forecasts WHERE target_id = ?", (target_id,))
        db.execute(
            """
            INSERT INTO forecast_runs(target_id, status, error) VALUES (?, 'complete', NULL)
            ON CONFLICT(target_id) DO UPDATE SET status='complete', error=NULL
            """,
            (target_id,),
        )
        db.executemany(
            """
            INSERT INTO forecasts(
                target_id, horizon, target_date, actual_difference, actual_direction,
                predicted_direction, vote_up, vote_count, correct
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [(target_id, *row) for row in rows],
        )
