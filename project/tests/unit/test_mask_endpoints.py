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


class TestSaveTrainingSample:
    def test_saves_image_and_mask_files(self, app, client, tmp_path, sample_slice):
        """Saves image + mask in Cellpose convention."""
        import project.frontend.server_context as ctx
        from tifffile import imread

        training_dir = ctx.PROJECT_ROOT / "cellpose_training"

        mask_data = np.array([[0, 1], [2, 0]], dtype=np.int32)
        compressed = zlib.compress(mask_data.tobytes())

        resp = client.post(
            "/api/cellpose/save-training-sample",
            data=json.dumps({
                "imagePath": str(sample_slice),
                "maskHex": compressed.hex(),
                "width": 2,
                "height": 2,
            }),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["trainingSetStats"]["totalImages"] >= 1

        # Verify files exist with Cellpose naming convention
        saved_name = Path(data["savedAs"]).stem  # e.g. "test_slice"
        img_path = training_dir / f"{saved_name}.tif"
        mask_path = training_dir / f"{saved_name}_masks.tif"
        assert img_path.exists()
        assert mask_path.exists()

        # Verify mask content
        saved_mask = imread(str(mask_path))
        assert saved_mask.shape == (2, 2)
        assert int(saved_mask.max()) == 2

    def test_rejects_missing_image(self, client):
        resp = client.post(
            "/api/cellpose/save-training-sample",
            data=json.dumps({
                "imagePath": "/nonexistent/path.tif",
                "maskHex": "deadbeef",
                "width": 2,
                "height": 2,
            }),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_training_stats_accumulate(self, app, client, tmp_path):
        """Stats count increases as more samples are saved."""
        import project.frontend.server_context as ctx

        for i in range(3):
            slice_path = tmp_path / f"slice_{i}.tif"
            img = np.random.randint(0, 65535, (8, 8), dtype=np.uint16)
            imwrite(str(slice_path), img)

            mask_data = np.ones((8, 8), dtype=np.int32) * (i + 1)
            compressed = zlib.compress(mask_data.tobytes())

            resp = client.post(
                "/api/cellpose/save-training-sample",
                data=json.dumps({
                    "imagePath": str(slice_path),
                    "maskHex": compressed.hex(),
                    "width": 8,
                    "height": 8,
                }),
                content_type="application/json",
            )
            assert resp.status_code == 200

        data = resp.get_json()
        assert data["trainingSetStats"]["totalImages"] == 3


class TestTrainingEndpoints:
    def test_train_status_idle(self, client):
        """Status is idle when no training has been started."""
        resp = client.get("/api/cellpose/train-status")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["status"] in ("idle", "completed", "cancelled", "failed")

    def test_train_rejects_insufficient_data(self, client, app):
        """Cannot start training with fewer than 2 images."""
        resp = client.post(
            "/api/cellpose/train",
            data=json.dumps({"baseModel": "cyto3", "modelName": "test"}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        assert "at least 2" in resp.get_json()["error"]

    def test_train_cancel(self, client):
        resp = client.post("/api/cellpose/train-cancel")
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True


class TestTrainingSetEndpoints:
    def test_training_set_info_empty(self, client):
        resp = client.get("/api/cellpose/training-set")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["stats"]["totalImages"] == 0
        assert data["ready"] is False

    def test_training_set_info_with_samples(self, client, tmp_path):
        """After saving samples, training-set returns them."""
        import project.frontend.server_context as ctx

        # Create training data directly
        td = ctx.PROJECT_ROOT / "cellpose_training"
        td.mkdir(parents=True, exist_ok=True)
        for i in range(3):
            imwrite(str(td / f"s{i}.tif"), np.zeros((8, 8), dtype=np.uint16))
            mask = np.ones((8, 8), dtype=np.uint16) * (i + 1)
            imwrite(str(td / f"s{i}_masks.tif"), mask)

        resp = client.get("/api/cellpose/training-set")
        data = resp.get_json()
        assert data["stats"]["totalImages"] == 3
        assert data["ready"] is True
        assert len(data["samples"]) == 3

    def test_delete_training_sample(self, client, tmp_path):
        import project.frontend.server_context as ctx

        td = ctx.PROJECT_ROOT / "cellpose_training"
        td.mkdir(parents=True, exist_ok=True)
        imwrite(str(td / "test.tif"), np.zeros((4, 4), dtype=np.uint16))
        imwrite(str(td / "test_masks.tif"), np.ones((4, 4), dtype=np.uint16))

        resp = client.delete("/api/cellpose/training-set/test")
        assert resp.status_code == 200
        assert resp.get_json()["ok"] is True
        assert not (td / "test.tif").exists()
        assert not (td / "test_masks.tif").exists()

    def test_delete_nonexistent_sample(self, client):
        resp = client.delete("/api/cellpose/training-set/nonexistent")
        assert resp.status_code == 404


class TestApplyModel:
    def test_apply_model(self, client):
        with patch("project.scripts.cellpose_trainer._update_config_primary_model") as mock_update, \
             patch("project.scripts.cellpose_trainer._clear_model_cache") as mock_clear:
            resp = client.post(
                "/api/cellpose/apply-model",
                data=json.dumps({"modelName": "my_model"}),
                content_type="application/json",
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["ok"] is True
            assert data["appliedModel"] == "my_model"
            mock_update.assert_called_once_with("my_model")
            mock_clear.assert_called_once()

    def test_apply_model_missing_name(self, client):
        resp = client.post(
            "/api/cellpose/apply-model",
            data=json.dumps({}),
            content_type="application/json",
        )
        assert resp.status_code == 400
