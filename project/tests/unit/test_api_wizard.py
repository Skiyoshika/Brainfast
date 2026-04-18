"""REST API tests for the onboarding wizard.

The wizard helps a brand-new user get from "I have raw TIFFs" to "the
pipeline is running" without leaving the UI. Currently the only way to
start is hand-editing a config JSON and invoking ``python scripts/main.py``
from the CLI — these endpoints close that gap.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from tifffile import imwrite

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import project.frontend.server_context as ctx  # noqa: E402
from project.frontend.server import app  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(ctx, "OUTPUT_DIR", tmp_path / "outputs")
    ctx.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ctx.run_state["outputDir"] = ""
    ctx.run_state["runName"] = ""
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# /api/wizard/inspect-source
# ---------------------------------------------------------------------------


def test_inspect_source_missing_path_returns_400(client):
    resp = client.post(
        "/api/wizard/inspect-source",
        data=json.dumps({}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False


def test_inspect_source_nonexistent_path_returns_404(client, tmp_path):
    resp = client.post(
        "/api/wizard/inspect-source",
        data=json.dumps({"sourcePath": str(tmp_path / "no_such_file.tif")}),
        content_type="application/json",
    )
    assert resp.status_code == 404


def test_inspect_source_directory_lists_tiff_files(client, tmp_path):
    src = tmp_path / "slices"
    src.mkdir()
    for z in (50, 55, 60):
        imwrite(str(src / f"z{z:04d}.tif"), np.full((10, 10), 100, dtype=np.uint16))

    resp = client.post(
        "/api/wizard/inspect-source",
        data=json.dumps({"sourcePath": str(src)}),
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert data["kind"] == "directory"
    assert data["n_files"] == 3
    assert data["sample_shape"] == [10, 10]
    # Sample id should auto-derive from directory name
    assert data["suggested_sample_id"] == "slices"
    # Without ImageJ metadata, we still hand back sane defaults
    assert data["suggested_pixel_um_xy"] == 5.0
    assert data["suggested_z_spacing_um"] == 25.0


def test_inspect_source_multipage_tiff_reads_imagej_spacing(client, tmp_path):
    # Build a 4-page TIFF with ImageJ-style spacing metadata
    src = tmp_path / "stack.tif"
    pages = np.full((4, 8, 8), 50, dtype=np.uint16)
    imwrite(
        str(src),
        pages,
        imagej=True,
        resolution=(1.0 / 5.0, 1.0 / 5.0),  # 5 um pixel
        metadata={"spacing": 24.765, "unit": "micron"},
    )

    resp = client.post(
        "/api/wizard/inspect-source",
        data=json.dumps({"sourcePath": str(src)}),
        content_type="application/json",
    )
    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    assert data["kind"] == "multipage_tiff"
    assert data["n_pages"] == 4
    assert data["sample_shape"] == [8, 8]
    assert data["suggested_z_spacing_um"] == pytest.approx(24.765, rel=1e-3)
    # Pixel size derived from XResolution (5 µm)
    assert data["suggested_pixel_um_xy"] == pytest.approx(5.0, rel=1e-2)
    # Multi-page TIFFs need extraction before pipeline can run
    assert data["needs_extraction"] is True


# ---------------------------------------------------------------------------
# /api/wizard/launch
# ---------------------------------------------------------------------------


def test_launch_missing_required_fields_returns_400(client):
    resp = client.post(
        "/api/wizard/launch",
        data=json.dumps({"sampleId": "sample1"}),  # missing inputDir + spacings
        content_type="application/json",
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["ok"] is False
    assert "inputDir" in body["error"] or "missing" in body["error"]


def test_launch_with_valid_payload_invokes_run_pipeline(client, tmp_path, monkeypatch):
    """Wizard launch should compose a payload and POST it to /api/run.

    We monkey-patch the pipeline runner so the test does not actually launch
    a Python subprocess, only verifies that the wizard reaches /api/run with
    a sensible payload.
    """
    src_dir = tmp_path / "slices"
    src_dir.mkdir()
    imwrite(str(src_dir / "z0050.tif"), np.full((4, 4), 100, dtype=np.uint16))

    captured: dict = {}

    def _fake_runner(config, input_dir, channels, params, *, job_id=None):
        captured["config"] = str(config)
        captured["input_dir"] = str(input_dir)
        captured["channels"] = list(channels)
        captured["job_id"] = job_id

    monkeypatch.setattr(ctx, "_runner", _fake_runner)

    resp = client.post(
        "/api/wizard/launch",
        data=json.dumps(
            {
                "sampleId": "sample42",
                "inputDir": str(src_dir),
                "pixelSizeUm": 5.0,
                "zSpacingUm": 24.765,
                "channels": ["red"],
                "atlasHemisphere": "right_flipped",
            }
        ),
        content_type="application/json",
    )
    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    assert data["ok"] is True
    assert data["jobId"]  # auto-assigned or echoed
    # The runner must have been called via /api/run plumbing
    assert "input_dir" in captured
    assert captured["input_dir"] == str(src_dir)
    assert "red" in captured["channels"]
    # Generated config file should exist on disk and contain our spacings
    cfg_path = Path(captured["config"])
    assert cfg_path.exists()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    assert cfg["input"]["pixel_size_um_xy"] == pytest.approx(5.0)
    assert cfg["input"]["slice_spacing_um"] == pytest.approx(24.765)


def test_launch_rejects_nonexistent_input_dir(client, tmp_path):
    resp = client.post(
        "/api/wizard/launch",
        data=json.dumps(
            {
                "sampleId": "x",
                "inputDir": str(tmp_path / "no_such_dir"),
                "pixelSizeUm": 5.0,
                "zSpacingUm": 25.0,
            }
        ),
        content_type="application/json",
    )
    assert resp.status_code == 404
