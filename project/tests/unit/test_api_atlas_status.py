"""Tests for GET /api/atlas/status — surfaces atlas asset readiness to the UI.

The endpoint backs the "atlas missing" banner: frontend calls it on page load
and, if `allRequiredReady` is false, shows a banner explaining how to obtain
the atlas.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import project.frontend.server_context as ctx  # noqa: E402
from project.frontend.server import app  # noqa: E402

pytestmark = pytest.mark.unit


def test_atlas_status_reports_ready_when_files_present(tmp_path, monkeypatch):
    monkeypatch.setattr(ctx, "PROJECT_ROOT", tmp_path)
    (tmp_path / "annotation_25.nii.gz").write_bytes(b"stub")
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "allen_mouse_structure_graph.csv").write_text("id,name\n1,root\n")

    with app.test_client() as client:
        resp = client.get("/api/atlas/status")
        assert resp.status_code == 200
        data = resp.get_json()

    assert data["ok"] is True
    assert data["annotationReady"] is True
    assert data["structureReady"] is True
    assert data["allRequiredReady"] is True
    assert "projectRoot" in data


def test_atlas_status_reports_missing_when_files_absent(tmp_path, monkeypatch):
    monkeypatch.setattr(ctx, "PROJECT_ROOT", tmp_path)

    with app.test_client() as client:
        resp = client.get("/api/atlas/status")
        assert resp.status_code == 200
        data = resp.get_json()

    assert data["ok"] is True
    assert data["annotationReady"] is False
    assert data["allRequiredReady"] is False
    assert "downloadHint" in data
    assert "download_atlas" in data["downloadHint"].lower()


def test_atlas_status_distinguishes_missing_structure_from_missing_annotation(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(ctx, "PROJECT_ROOT", tmp_path)
    (tmp_path / "annotation_25.nii.gz").write_bytes(b"stub")

    with app.test_client() as client:
        resp = client.get("/api/atlas/status")
        assert resp.status_code == 200
        data = resp.get_json()

    assert data["annotationReady"] is True
    assert data["structureReady"] is False
    assert data["allRequiredReady"] is False
