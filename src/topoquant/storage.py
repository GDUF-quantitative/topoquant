from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Iterable
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np

from .domain import CloudRecord, Diagram


LOGGER = logging.getLogger(__name__)

SCHEMA_VERSION = "1"


class StorageError(RuntimeError):
    """实验库状态与当前运行不兼容。"""


def connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    connection.execute("PRAGMA synchronous = NORMAL")
    create_schema(connection)
    return connection


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
        """
    )
    existing = get_metadata(db, "schema_version")
    if existing is not None and existing != SCHEMA_VERSION:
        raise StorageError(f"实验库版本为 {existing}，当前程序需要 {SCHEMA_VERSION}")
    set_metadata(db, "schema_version", SCHEMA_VERSION)
    db.commit()


def get_metadata(db: sqlite3.Connection, key: str) -> str | None:
    row = db.execute("SELECT value FROM metadata WHERE key = ?", (key,)).fetchone()
    return None if row is None else str(row["value"])


def set_metadata(db: sqlite3.Connection, key: str, value: str) -> None:
    db.execute(
        "INSERT INTO metadata(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def clear_experiment_data(db: sqlite3.Connection) -> None:
    """清空全部实验数据行（保留 metadata），用于强制重算。"""
    for table in ("matches", "match_runs", "forecasts", "forecast_runs", "diagrams", "clouds"):
        db.execute(f"DELETE FROM {table}")
    db.commit()


def check_or_set_identity(
    db: sqlite3.Connection,
    topology_signature: str,
    source_signature: str,
    config_json: dict[str, object],
    force: bool = False,
) -> None:
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
    db.execute(
        """
        INSERT INTO clouds(cloud_id, cloud_date, stock_code, source_path, point_count, status, error)
        VALUES (?, ?, ?, ?, ?, 'complete', NULL)
        ON CONFLICT(cloud_id) DO UPDATE SET
            cloud_date=excluded.cloud_date,
            stock_code=excluded.stock_code,
            source_path=excluded.source_path,
            point_count=excluded.point_count,
            status='complete', error=NULL
        """,
        (record.cloud_id, record.cloud_date.isoformat(), record.stock_code, str(record.source_path), record.point_count),
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
    db.execute(
        """
        INSERT INTO clouds(cloud_id, cloud_date, stock_code, source_path, point_count, status, error)
        VALUES (?, ?, ?, ?, 0, 'error', ?)
        ON CONFLICT(cloud_id) DO UPDATE SET status='error', error=excluded.error
        """,
        (cloud_id, cloud_date.isoformat(), stock_code, str(source_path), error),
    )


def load_diagrams(db: sqlite3.Connection, dimensions: Iterable[int]) -> dict[str, Diagram]:
    requested = tuple(sorted(set(dimensions)))
    placeholders = ",".join("?" for _ in requested)
    rows = db.execute(
        f"""
        SELECT d.cloud_id, d.dimension, d.pair_count, d.pairs
        FROM diagrams d JOIN clouds c ON c.cloud_id = d.cloud_id
        WHERE c.status = 'complete' AND d.dimension IN ({placeholders})
        ORDER BY d.cloud_id, d.dimension
        """,
        requested,
    )
    result: dict[str, Diagram] = {}
    for row in rows:
        pairs = np.frombuffer(row["pairs"], dtype="<f8").copy().reshape(int(row["pair_count"]), 2)
        result.setdefault(str(row["cloud_id"]), {})[int(row["dimension"])] = pairs
    return {cloud_id: diagram for cloud_id, diagram in result.items() if all(dim in diagram for dim in requested)}


def load_cloud_records(db: sqlite3.Connection) -> dict[str, CloudRecord]:
    rows = db.execute(
        "SELECT cloud_id, cloud_date, stock_code, source_path, point_count FROM clouds WHERE status='complete'"
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
) -> None:
    status = "selected" if matches else "not_selected"
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
        "INSERT INTO matches(target_id, similar_id, rank, distance_dim0, distance_dim1) VALUES (?, ?, ?, ?, ?)",
        [(target_id, similar_id, rank, d0, d1) for rank, (similar_id, d0, d1) in enumerate(matches, 1)],
    )


def save_forecast_error(db: sqlite3.Connection, target_id: str, error: str) -> None:
    db.execute("DELETE FROM forecasts WHERE target_id = ?", (target_id,))
    db.execute(
        """
        INSERT INTO forecast_runs(target_id, status, error) VALUES (?, 'error', ?)
        ON CONFLICT(target_id) DO UPDATE SET status='error', error=excluded.error
        """,
        (target_id, error),
    )


def save_forecasts(db: sqlite3.Connection, target_id: str, rows: list[tuple[object, ...]]) -> None:
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
