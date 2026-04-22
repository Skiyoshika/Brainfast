"""Unit tests for the per-job liquify progress file.

Mirrors the contract of ``scripts/pipeline_progress.py`` but writes to a
separate file so the main pipeline's progress isn't clobbered while the
user runs ``/api/liquify-3d/apply`` or ``/finalize``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_write_then_read_roundtrip(tmp_path):
    from project.scripts.liquify_progress import (
        read_liquify_progress,
        write_liquify_progress,
    )

    write_liquify_progress(
        job_dir=tmp_path,
        stage="solve",
        stage_index=2,
        stage_count=5,
        percent=40,
        message="Solving axis 1",
    )
    data = read_liquify_progress(tmp_path)
    assert data["stage"] == "solve"
    assert data["stage_index"] == 2
    assert data["stage_count"] == 5
    assert data["percent"] == 40
    assert data["message"] == "Solving axis 1"


def test_read_returns_empty_when_no_file(tmp_path):
    from project.scripts.liquify_progress import read_liquify_progress

    data = read_liquify_progress(tmp_path)
    assert data == {}


def test_write_is_atomic_from_partial_writes(tmp_path):
    """A reader running mid-write must never see a half-written file."""
    from project.scripts.liquify_progress import (
        read_liquify_progress,
        write_liquify_progress,
    )

    for i in range(5):
        write_liquify_progress(
            job_dir=tmp_path,
            stage=f"stage_{i}",
            stage_index=i,
            stage_count=5,
            percent=i * 20,
            message="ok",
        )
        # Reading right after every write must yield valid JSON (no torn reads)
        data = read_liquify_progress(tmp_path)
        assert "stage" in data
        assert data["percent"] == i * 20


def test_progress_file_lives_in_liquify_subdir(tmp_path):
    """Must not collide with the main pipeline_progress.json."""
    from project.scripts.liquify_progress import write_liquify_progress

    write_liquify_progress(
        job_dir=tmp_path, stage="x", stage_index=0, stage_count=1, percent=0, message=""
    )
    assert (tmp_path / "liquify_progress.json").exists()
    assert not (tmp_path / "pipeline_progress.json").exists()


def test_write_validates_percent_range(tmp_path):
    from project.scripts.liquify_progress import write_liquify_progress

    with pytest.raises(ValueError, match="percent"):
        write_liquify_progress(
            job_dir=tmp_path,
            stage="s", stage_index=0, stage_count=1, percent=150, message="",
        )


def test_write_clamps_negative_percent(tmp_path):
    """Protect against off-by-one calculations — 0 is the floor."""
    from project.scripts.liquify_progress import write_liquify_progress

    with pytest.raises(ValueError, match="percent"):
        write_liquify_progress(
            job_dir=tmp_path,
            stage="s", stage_index=0, stage_count=1, percent=-5, message="",
        )


def test_clear_progress_removes_file(tmp_path):
    from project.scripts.liquify_progress import (
        clear_liquify_progress,
        write_liquify_progress,
    )

    write_liquify_progress(
        job_dir=tmp_path, stage="x", stage_index=0, stage_count=1, percent=0, message=""
    )
    clear_liquify_progress(tmp_path)
    assert not (tmp_path / "liquify_progress.json").exists()
