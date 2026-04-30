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


# ---------------------------------------------------------------------------
# /api/wizard/launch — dual-channel (Phase 5)
# ---------------------------------------------------------------------------


def test_launch_accepts_inputDirs_dict_for_dual_channel(client, tmp_path, monkeypatch):
    """Dual-channel payload sends inputDirs={red:..., farred:...} + channels=[...].
    Runner should receive the dict so per-channel slice dirs are respected.
    """
    dir_red = tmp_path / "c0_slices"
    dir_far = tmp_path / "c1_slices"
    dir_red.mkdir()
    dir_far.mkdir()
    imwrite(str(dir_red / "z0000.tif"), np.full((4, 4), 100, dtype=np.uint16))
    imwrite(str(dir_far / "z0000.tif"), np.full((4, 4), 200, dtype=np.uint16))

    captured: dict = {}

    def _fake_runner(config, input_dir, channels, params, *, job_id=None):
        captured["config"] = str(config)
        captured["input_dir"] = input_dir
        captured["channels"] = list(channels)

    monkeypatch.setattr(ctx, "_runner", _fake_runner)

    resp = client.post(
        "/api/wizard/launch",
        data=json.dumps(
            {
                "sampleId": "dual42",
                "inputDirs": {"red": str(dir_red), "farred": str(dir_far)},
                "pixelSizeUm": 5.0,
                "zSpacingUm": 24.765,
                "channels": ["red", "farred"],
                "atlasHemisphere": "right_flipped",
            }
        ),
        content_type="application/json",
    )
    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    assert data["ok"] is True
    assert set(data["channels"]) == {"red", "farred"}
    assert set(data["input_dirs"].keys()) == {"red", "farred"}

    # Runner received the dict form so per-channel dirs are preserved
    assert isinstance(captured["input_dir"], dict)
    assert captured["input_dir"]["red"] == str(dir_red)
    assert captured["input_dir"]["farred"] == str(dir_far)
    assert captured["channels"] == ["red", "farred"]


def test_launch_applies_xulab_parity_overrides_to_generated_config(client, tmp_path, monkeypatch):
    """Advanced wizard fields must flow into the registration block."""
    src_dir = tmp_path / "slices"
    src_dir.mkdir()
    imwrite(str(src_dir / "z0050.tif"), np.full((4, 4), 100, dtype=np.uint16))

    captured: dict = {}

    def _fake_runner(config, input_dir, channels, params, *, job_id=None):
        captured["config"] = str(config)

    monkeypatch.setattr(ctx, "_runner", _fake_runner)

    resp = client.post(
        "/api/wizard/launch",
        data=json.dumps(
            {
                "sampleId": "xulab42",
                "inputDir": str(src_dir),
                "pixelSizeUm": 5.0,
                "zSpacingUm": 25.0,
                "channels": ["red"],
                "atlasHemisphere": "right_flipped",
                "antsTransform": "Affine",
                "axisAlignmentEnabled": True,
                "useCellToCcfMapping": True,
                "fixedMaxDim": 256,
            }
        ),
        content_type="application/json",
    )
    assert resp.status_code == 200, resp.get_json()
    cfg = json.loads(Path(captured["config"]).read_text(encoding="utf-8"))
    reg = cfg["registration"]
    assert reg["ants_transform"] == "Affine"
    assert reg["axis_alignment_enabled"] is True
    assert reg["use_cell_to_ccf_mapping"] is True
    assert reg["fixed_max_dim"] == 256


def test_launch_without_xulab_fields_keeps_safe_defaults(client, tmp_path, monkeypatch):
    """Old payloads without the new fields default to SyNRA + flags off + no fixed_max_dim."""
    src_dir = tmp_path / "slices"
    src_dir.mkdir()
    imwrite(str(src_dir / "z0050.tif"), np.full((4, 4), 100, dtype=np.uint16))

    captured: dict = {}

    def _fake_runner(config, input_dir, channels, params, *, job_id=None):
        captured["config"] = str(config)

    monkeypatch.setattr(ctx, "_runner", _fake_runner)

    resp = client.post(
        "/api/wizard/launch",
        data=json.dumps(
            {
                "sampleId": "legacy42",
                "inputDir": str(src_dir),
                "pixelSizeUm": 5.0,
                "zSpacingUm": 25.0,
                "channels": ["red"],
                "atlasHemisphere": "right_flipped",
            }
        ),
        content_type="application/json",
    )
    assert resp.status_code == 200
    cfg = json.loads(Path(captured["config"]).read_text(encoding="utf-8"))
    reg = cfg["registration"]
    assert reg["ants_transform"] == "SyNRA"
    assert reg["axis_alignment_enabled"] is False
    assert reg["use_cell_to_ccf_mapping"] is False
    assert "fixed_max_dim" not in reg


def test_launch_dual_channel_rejects_missing_channel_dir(client, tmp_path, monkeypatch):
    """If inputDirs references a directory that doesn't exist, fail fast with
    404 + channel name so the user knows which path is broken.
    """
    dir_red = tmp_path / "c0_slices"
    dir_red.mkdir()
    imwrite(str(dir_red / "z0000.tif"), np.full((4, 4), 100, dtype=np.uint16))

    # Runner must NOT be invoked when validation fails
    called = False

    def _fake_runner(*args, **kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(ctx, "_runner", _fake_runner)

    resp = client.post(
        "/api/wizard/launch",
        data=json.dumps(
            {
                "sampleId": "dual_bad",
                "inputDirs": {
                    "red": str(dir_red),
                    "farred": str(tmp_path / "does_not_exist"),
                },
                "pixelSizeUm": 5.0,
                "zSpacingUm": 25.0,
                "channels": ["red", "farred"],
            }
        ),
        content_type="application/json",
    )
    assert resp.status_code == 404
    body = resp.get_json()
    assert body["ok"] is False
    assert "farred" in body["error"]
    assert not called


def test_launch_inputDirs_alone_synthesizes_inputDir_for_legacy_validation(client, tmp_path, monkeypatch):
    """Callers may send only inputDirs (no inputDir) — backend should still
    pass the legacy required-field check by picking the first channel's dir.
    """
    dir_red = tmp_path / "c0_slices"
    dir_red.mkdir()
    imwrite(str(dir_red / "z0000.tif"), np.full((4, 4), 100, dtype=np.uint16))

    monkeypatch.setattr(ctx, "_runner", lambda *a, **kw: None)

    resp = client.post(
        "/api/wizard/launch",
        data=json.dumps(
            {
                "sampleId": "dual_no_legacy_field",
                # No inputDir at all — only inputDirs
                "inputDirs": {"red": str(dir_red)},
                "pixelSizeUm": 5.0,
                "zSpacingUm": 25.0,
                "channels": ["red"],
            }
        ),
        content_type="application/json",
    )
    assert resp.status_code == 200, resp.get_json()


# ---------------------------------------------------------------------------
# /api/wizard/extract-multipage-tiff
# ---------------------------------------------------------------------------


def test_extract_multipage_tiff_rejects_missing_fields(client):
    resp = client.post(
        "/api/wizard/extract-multipage-tiff",
        data=json.dumps({}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False


def test_extract_multipage_tiff_rejects_nonexistent_src(client, tmp_path):
    resp = client.post(
        "/api/wizard/extract-multipage-tiff",
        data=json.dumps(
            {
                "src": str(tmp_path / "nope.tif"),
                "outDir": str(tmp_path / "out"),
            }
        ),
        content_type="application/json",
    )
    assert resp.status_code == 404


def test_extract_multipage_tiff_splits_pages_to_directory(client, tmp_path):
    """Real round-trip: write a 3-page TIFF, hit the endpoint, verify the
    output directory ends up with 3 single-page slice files."""
    src = tmp_path / "multipage.tif"
    out_dir = tmp_path / "slices"
    pages = np.stack(
        [
            np.full((4, 4), 100, dtype=np.uint16),
            np.full((4, 4), 200, dtype=np.uint16),
            np.full((4, 4), 300, dtype=np.uint16),
        ]
    )
    imwrite(str(src), pages)

    resp = client.post(
        "/api/wizard/extract-multipage-tiff",
        data=json.dumps({"src": str(src), "outDir": str(out_dir)}),
        content_type="application/json",
    )
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    assert body["ok"] is True
    assert body["writtenCount"] == 3
    assert out_dir.exists()
    # extract_zstack writes z*.tif slices
    written = sorted(p.name for p in out_dir.glob("*.tif"))
    assert len(written) == 3
