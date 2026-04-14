"""Tests for mask-related endpoints (detect preview masks + save training sample)."""

from __future__ import annotations

import io
import json
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from tifffile import imwrite


@pytest.fixture()
def app(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "outputs").mkdir()

    import project.frontend.server_context as ctx

    ctx.ROOT = project_root / "frontend"
    ctx.PROJECT_ROOT = project_root
    ctx.OUTPUT_DIR = project_root / "outputs"

    from flask import Flask

    from project.frontend.blueprints.api_cellpose import bp as cellpose_bp
    from project.frontend.blueprints.api_detect_preview import bp as detect_bp

    flask_app = Flask(__name__)
    flask_app.register_blueprint(detect_bp)
    flask_app.register_blueprint(cellpose_bp)
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def sample_slice(tmp_path):
    slice_path = tmp_path / "test_slice.tif"
    img = np.random.randint(0, 65535, (64, 64), dtype=np.uint16)
    imwrite(str(slice_path), img)
    return slice_path


class TestMasksEndpoint:
    def test_returns_masks_after_detection(self, client, sample_slice):
        """After a detection run, /masks returns the raw mask array."""
        import pandas as pd

        fake_masks = np.array([[0, 1, 1], [0, 2, 2], [0, 0, 0]], dtype=np.int32)
        fake_df = pd.DataFrame({
            "x": [1.5, 1.5], "y": [0.0, 1.0],
            "detector": ["cellpose_cyto3"] * 2,
            "cell_id": [1, 2], "score": [1.0, 1.0], "area_px": [2.0, 2.0],
        })

        def mock_run_detection_with_masks(slice_path, cfg):
            return fake_df, fake_masks

        with patch(
            "project.frontend.blueprints.api_detect_preview._run_detection_with_masks",
            side_effect=mock_run_detection_with_masks,
        ):
            # First run detection
            resp = client.post(
                "/api/detect/preview",
                data=json.dumps({
                    "slicePath": str(sample_slice),
                    "jobId": "mask_test",
                    "returnMasks": True,
                }),
                content_type="application/json",
            )
            assert resp.status_code == 200

            # Then fetch masks
            resp2 = client.get("/api/detect/preview/masks?jobId=mask_test")
            assert resp2.status_code == 200
            data = resp2.get_json()
            assert data["ok"] is True
            assert data["width"] == 3
            assert data["height"] == 3
            assert data["cellCount"] == 2

            # Decompress and verify mask data
            mask_bytes = zlib.decompress(bytes.fromhex(data["maskHex"]))
            mask_arr = np.frombuffer(mask_bytes, dtype=np.int32).reshape(
                data["height"], data["width"]
            )
            assert mask_arr.shape == (3, 3)
            assert int(mask_arr.max()) == 2

    def test_masks_not_available_returns_404(self, client):
        resp = client.get("/api/detect/preview/masks?jobId=nonexistent")
        assert resp.status_code == 404
