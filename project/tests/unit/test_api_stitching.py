"""Tests for the stitching blueprint.

The stitching core is a heavy port from Xu Lab with opencv-python +
colorama + joblib as optional runtime deps. Tests here exercise only the
lightweight Flask plumbing (probe endpoint, validation, graceful failure)
plus a smoke import of the vendored package once the ``stitching`` extras
are installed — we don't try to run the actual stitch without real tile
data because that requires TissueCyte Mosaic files.
"""

from __future__ import annotations

import json

import pytest

from project.frontend.server import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_stitching_available_probe_shape(client):
    resp = client.get("/api/stitching/available")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert "available" in body
    assert isinstance(body["available"], bool)
    assert body["install"] == 'pip install -e ".[stitching]"'


def test_stitching_start_rejects_missing_fields(client):
    resp = client.post(
        "/api/stitching/start",
        data=json.dumps({}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["ok"] is False


def test_stitching_start_rejects_nonexistent_input(client, tmp_path):
    resp = client.post(
        "/api/stitching/start",
        data=json.dumps(
            {"inputDir": str(tmp_path / "no_such_dir"),
             "outputDir": str(tmp_path / "out")}
        ),
        content_type="application/json",
    )
    assert resp.status_code == 404


def test_stitching_start_requires_bezier_path_when_default_asset_missing(
    client, tmp_path, monkeypatch
):
    from project.frontend.blueprints import api_stitching

    class _NoopThread:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

    input_dir = tmp_path / "tiles"
    input_dir.mkdir()
    monkeypatch.setattr(api_stitching, "_deps_available", lambda: (True, None))
    monkeypatch.setattr(api_stitching.threading, "Thread", _NoopThread)
    monkeypatch.setattr(
        api_stitching,
        "_default_bezier_path",
        lambda: tmp_path / "missing" / "bezier16x.pkl",
        raising=False,
    )

    resp = client.post(
        "/api/stitching/start",
        data=json.dumps({"inputDir": str(input_dir), "outputDir": str(tmp_path / "out")}),
        content_type="application/json",
    )

    assert resp.status_code == 400
    body = resp.get_json()
    assert body["ok"] is False
    assert "bezierPath" in body["error"]


def test_stitching_status_rejects_unknown_job(client):
    resp = client.get("/api/stitching/status?jobId=does-not-exist")
    assert resp.status_code == 404


def test_stitching_status_rejects_missing_job_id(client):
    resp = client.get("/api/stitching/status")
    assert resp.status_code == 400


def test_stitching_package_metadata_importable():
    """The package init + launch stub must be importable even without the
    optional runtime deps (which opencv+joblib+colorama satisfy when the
    ``stitching`` extras are installed)."""
    import project.scripts.stitching as mod

    assert mod.__doc__ is not None
    assert "Xu Lab" in mod.__doc__ or "UCI-XuLab" in mod.__doc__


def test_stitching_launch_stub_raises_not_implemented():
    """The vendored launch.py is a stub — calling it should tell the user
    where to go (web UI / CLI) rather than silently succeed."""
    from project.scripts.stitching.launch import main

    with pytest.raises(NotImplementedError):
        main()
