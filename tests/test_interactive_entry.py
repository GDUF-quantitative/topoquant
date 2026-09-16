from __future__ import annotations

import json
from pathlib import Path

import pytest

import run


def test_saved_protocol_takes_priority_over_example(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "config.example.json").write_text(
        json.dumps({"source_dir": "example", "work_dir": "example-work"}),
        encoding="utf-8",
    )
    (tmp_path / "config.json").write_text(
        json.dumps({"source_dir": "saved", "work_dir": "saved-work"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(run, "_PROJECT_ROOT", tmp_path)

    defaults = run.load_defaults()

    assert defaults["source_dir"] == "saved"
    assert defaults["work_dir"] == "saved-work"


def test_invalid_saved_protocol_is_not_silently_replaced(monkeypatch, tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("not-json", encoding="utf-8")
    (tmp_path / "config.example.json").write_text(
        json.dumps({"source_dir": "example", "work_dir": "example-work"}),
        encoding="utf-8",
    )
    monkeypatch.setattr(run, "_PROJECT_ROOT", tmp_path)

    with pytest.raises(RuntimeError, match="实验协议文件无效"):
        run.load_defaults()


def test_next_incremental_work_dir_uses_next_sibling(tmp_path: Path) -> None:
    (tmp_path / "runs" / "20240628").mkdir(parents=True)
    (tmp_path / "runs" / "20240628_2").mkdir()
    (tmp_path / "runs" / "20240628_4").mkdir()

    selected = run.next_incremental_work_dir("runs/20240628", tmp_path)

    assert selected == str(Path("runs") / "20240628_5")
    assert (tmp_path / selected).is_dir()


def test_next_incremental_work_dir_continues_from_generated_path(tmp_path: Path) -> None:
    (tmp_path / "runs" / "experiment").mkdir(parents=True)
    (tmp_path / "runs" / "experiment_2").mkdir()

    selected = run.next_incremental_work_dir("runs/experiment_2", tmp_path)

    assert selected == str(Path("runs") / "experiment_3")
    assert (tmp_path / selected).is_dir()
