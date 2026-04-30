"""Tests for the Neuroglancer blueprint + ng_converter points module.

The ``neuroglancer`` runtime is an optional extra — tests here only exercise
paths that don't require importing it (deps-available probe + points I/O +
graceful 501 when the extras are missing).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from project.frontend.server import create_app


@pytest.fixture
def client():
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_neuroglancer_available_probe_returns_boolean(client):
    resp = client.get("/api/neuroglancer/available")
    assert resp.status_code == 200
    body = resp.get_json()
    assert body["ok"] is True
    assert "available" in body
    assert isinstance(body["available"], bool)
    assert body["install"] == 'pip install -e ".[neuroglancer]"'


def test_ng_viewer_package_importable_without_server_sys_path():
    repo_root = Path(__file__).resolve().parents[3]
    result = subprocess.run(
        [sys.executable, "-c", "import project.scripts.ng_viewer"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr


def test_neuroglancer_convert_rejects_missing_fields(client):
    resp = client.post(
        "/api/neuroglancer/convert",
        data=json.dumps({}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    body = resp.get_json()
    assert body["ok"] is False
    assert "inputPath" in body["error"] or "outputDir" in body["error"]


def test_neuroglancer_convert_rejects_nonexistent_input(client, tmp_path):
    resp = client.post(
        "/api/neuroglancer/convert",
        data=json.dumps(
            {"inputPath": str(tmp_path / "does_not_exist.nii"),
             "outputDir": str(tmp_path / "out")}
        ),
        content_type="application/json",
    )
    assert resp.status_code == 404


def test_neuroglancer_stop_rejects_unknown_session(client):
    resp = client.post(
        "/api/neuroglancer/stop",
        data=json.dumps({"sessionId": "does-not-exist"}),
        content_type="application/json",
    )
    assert resp.status_code == 404


def test_points_roundtrip_legacy_format(tmp_path):
    """Legacy inputpoints.txt: two header lines + 'z y x' per line."""
    from project.scripts.ng_converter.points import load_legacy_points, load_points

    fp = tmp_path / "inputpoints.txt"
    fp.write_text("index\n3\n10 20 30\n11 21 31\n12 22 32\n", encoding="utf-8")
    pts = load_points(str(fp))
    assert pts.shape == (3, 3)
    np.testing.assert_allclose(pts[0], [10, 20, 30])

    pts_explicit = load_legacy_points(str(fp))
    np.testing.assert_array_equal(pts, pts_explicit)


def test_points_roundtrip_napari_csv(tmp_path):
    """Napari CSV: index,axis-0,axis-1,axis-2."""
    from project.scripts.ng_converter.points import load_points

    fp = tmp_path / "napari.csv"
    fp.write_text("index,axis-0,axis-1,axis-2\n0,5.0,6.0,7.0\n1,8.0,9.0,10.0\n", encoding="utf-8")
    pts = load_points(str(fp))
    assert pts.shape == (2, 3)
    np.testing.assert_allclose(pts[0], [5.0, 6.0, 7.0])


def test_write_precomputed_annotations_creates_info_and_spatial(tmp_path):
    from project.scripts.ng_converter.points import write_precomputed_annotations

    pts = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    out = write_precomputed_annotations(pts, tmp_path, spacing_um=(50, 1.25, 1.25))

    root = Path(out)
    assert (root / "info").exists()
    info = json.loads((root / "info").read_text())
    assert info["annotation_type"] == "POINT"
    assert info["dimensions"]["z"] == [50, "um"]

    spatial = root / "spatial0" / "0_0_0"
    assert spatial.exists()
    # File should contain a uint64 count + 2*(3 floats + 1 uint64) payload
    assert spatial.stat().st_size >= 8 + 2 * (12 + 8)


def test_ng_converter_get_spacing_from_nii(tmp_path):
    """Header zooms > 1 treated as already µm-encoded (Allen CCF style)."""
    import nibabel as nib

    from project.scripts.ng_converter import get_spacing_from_nii

    data = np.zeros((2, 3, 4), dtype=np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    # Set zooms > 1 to mimic µm-encoded atlas files
    img.header.set_zooms((25.0, 25.0, 25.0))
    out = tmp_path / "atlas.nii.gz"
    nib.save(img, str(out))

    spacing = get_spacing_from_nii(str(out))
    assert spacing == (25.0, 25.0, 25.0)


def test_ng_converter_get_spacing_mm_converts_to_um(tmp_path):
    """Header zooms < 1 treated as mm; converted to µm (×1000)."""
    import nibabel as nib

    from project.scripts.ng_converter import get_spacing_from_nii

    data = np.zeros((2, 3, 4), dtype=np.float32)
    img = nib.Nifti1Image(data, np.eye(4))
    img.header.set_zooms((0.025, 0.025, 0.025))
    out = tmp_path / "mm_encoded.nii.gz"
    nib.save(img, str(out))

    spacing = get_spacing_from_nii(str(out))
    np.testing.assert_allclose(spacing, (25.0, 25.0, 25.0))
