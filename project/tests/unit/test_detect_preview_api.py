"""Unit tests for the api_detect_preview blueprint."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from tifffile import imwrite


@pytest.fixture()
def app(tmp_path):
    """Create a minimal Flask app with the detect preview blueprint."""
    # Set up project paths BEFORE importing the blueprint
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "outputs").mkdir()

    # Patch server_context before importing
    import project.frontend.server_context as ctx

    ctx.ROOT = project_root / "frontend"
    ctx.PROJECT_ROOT = project_root
    ctx.OUTPUT_DIR = project_root / "outputs"

    from flask import Flask

    from project.frontend.blueprints.api_detect_preview import bp

    flask_app = Flask(__name__)
    flask_app.register_blueprint(bp)
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def sample_slice(tmp_path):
    """Create a tiny test TIFF slice."""
    slice_path = tmp_path / "test_slice.tif"
    img = np.random.randint(0, 65535, (64, 64), dtype=np.uint16)
    imwrite(str(slice_path), img)
    return slice_path


class TestDetectPreviewEndpoint:
    def test_missing_slice_path(self, client):
        resp = client.post(
            "/api/detect/preview",
            data=json.dumps({"jobId": "test"}),
            content_type="application/json",
        )
        assert resp.status_code == 400
        data = resp.get_json()
        assert data["ok"] is False
        assert "slicePath" in data["error"]

    def test_nonexistent_slice(self, client, tmp_path):
        resp = client.post(
            "/api/detect/preview",
            data=json.dumps({
                "slicePath": str(tmp_path / "nonexistent.tif"),
                "jobId": "test",
            }),
            content_type="application/json",
        )
        assert resp.status_code == 404
        data = resp.get_json()
        assert data["ok"] is False
        assert "not found" in data["error"]

    def test_detection_success(self, client, sample_slice):
        """Mock detect_cells to return fake results and verify the endpoint."""
        import pandas as pd

        fake_df = pd.DataFrame({
            "x": [10.0, 20.0, 30.0],
            "y": [15.0, 25.0, 35.0],
            "detector": ["mock_cellpose"] * 3,
        })

        with patch(
            "project.frontend.blueprints.api_detect_preview._run_detection",
            return_value=fake_df,
        ):
            resp = client.post(
                "/api/detect/preview",
                data=json.dumps({
                    "slicePath": str(sample_slice),
                    "jobId": "test_detect",
                }),
                content_type="application/json",
            )
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["ok"] is True
            assert data["cellCount"] == 3
            assert data["detector"] == "mock_cellpose"
            assert "overlayUrl" in data
            assert "csvUrl" in data

    def test_detection_error_returns_500(self, client, sample_slice):
        """Verify that detection errors are reported as 500."""
        with patch(
            "project.frontend.blueprints.api_detect_preview._run_detection",
            side_effect=RuntimeError("Cellpose crashed"),
        ):
            resp = client.post(
                "/api/detect/preview",
                data=json.dumps({
                    "slicePath": str(sample_slice),
                    "jobId": "test_err",
                }),
                content_type="application/json",
            )
            assert resp.status_code == 500
            data = resp.get_json()
            assert data["ok"] is False
            assert "Cellpose crashed" in data["error"]

    def test_overlay_endpoint_missing(self, client):
        resp = client.get("/api/detect/preview/overlay?jobId=nonexistent")
        assert resp.status_code == 404

    def test_csv_endpoint_missing(self, client):
        resp = client.get("/api/detect/preview/csv?jobId=nonexistent")
        assert resp.status_code == 404

    def test_status_endpoint_no_result(self, client):
        resp = client.get("/api/detect/preview/status?jobId=nonexistent")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is False
        assert data["available"] is False

    def test_config_loaded_from_config_path(self, app, tmp_path, sample_slice):
        """Verify detector preview reads config from run_state['config_path'],
        not from a nonexistent run_state['config']."""
        import json

        import pandas as pd

        import project.frontend.server_context as ctx

        # Write a real config file with custom detector settings
        custom_cfg = {
            "detection": {
                "primary_model": "cyto3",
                "cellpose_diameter_um": 12.0,
                "fallback_model": "threshold",
            },
            "input": {"pixel_size_um_xy": 2.5},
            "compute": {"device": "cpu"},
        }
        cfg_file = tmp_path / "custom_config.json"
        cfg_file.write_text(json.dumps(custom_cfg))
        ctx.run_state["config_path"] = str(cfg_file)

        fake_df = pd.DataFrame({
            "x": [10.0],
            "y": [15.0],
            "detector": ["cyto3"],
        })

        captured_cfg = {}

        def mock_detect(slice_path, cfg):
            captured_cfg.update(cfg)
            return fake_df

        with patch(
            "project.frontend.blueprints.api_detect_preview._run_detection",
            side_effect=mock_detect,
        ):
            client = app.test_client()
            resp = client.post(
                "/api/detect/preview",
                data=json.dumps({
                    "slicePath": str(sample_slice),
                    "jobId": "test_cfg",
                }),
                content_type="application/json",
            )
            assert resp.status_code == 200
            # Config passed to detector must come from the file, not a hard-coded fallback
            assert captured_cfg.get("detection", {}).get("primary_model") == "cyto3"
            assert captured_cfg.get("input", {}).get("pixel_size_um_xy") == 2.5

    def test_no_prior_run_uses_template_config(self, app, tmp_path, sample_slice):
        """When no prior run exists, detector preview uses template config,
        not a hard-coded mini-config."""
        import json

        import pandas as pd

        import project.frontend.server_context as ctx

        # Clear any prior run config
        ctx.run_state["config_path"] = None

        # Write a template config
        template_dir = ctx.PROJECT_ROOT / "configs"
        template_dir.mkdir(parents=True, exist_ok=True)
        template_cfg = {
            "detection": {"primary_model": "cpsam"},
            "input": {"pixel_size_um_xy": 5.0},
            "compute": {"device": "auto"},
        }
        (template_dir / "run_config.template.json").write_text(json.dumps(template_cfg))

        fake_df = pd.DataFrame({
            "x": [10.0],
            "y": [15.0],
            "detector": ["cpsam"],
        })

        captured_cfg = {}

        def mock_detect(slice_path, cfg):
            captured_cfg.update(cfg)
            return fake_df

        with patch(
            "project.frontend.blueprints.api_detect_preview._run_detection",
            side_effect=mock_detect,
        ):
            client = app.test_client()
            resp = client.post(
                "/api/detect/preview",
                data=json.dumps({
                    "slicePath": str(sample_slice),
                    "jobId": "test_template",
                }),
                content_type="application/json",
            )
            assert resp.status_code == 200
            # Must use template config, not empty dict
            assert captured_cfg.get("detection", {}).get("primary_model") == "cpsam"

    def test_param_overrides_applied_to_config(self, app, tmp_path, sample_slice):
        """When request includes 'params', they override config values."""
        import json

        import pandas as pd

        import project.frontend.server_context as ctx

        base_cfg = {
            "detection": {
                "primary_model": "cpsam",
                "cellpose_diameter_um": 12.0,
                "cellpose_flow_threshold": 0.4,
                "cellpose_cellprob_threshold": 0.0,
                "cellpose_min_size_px": 8,
            },
            "input": {"pixel_size_um_xy": 5.0},
            "compute": {"device": "cpu"},
        }
        cfg_file = tmp_path / "base_config.json"
        cfg_file.write_text(json.dumps(base_cfg))
        ctx.run_state["config_path"] = str(cfg_file)

        fake_df = pd.DataFrame({
            "x": [10.0], "y": [15.0], "detector": ["cyto3"],
        })

        captured_cfg = {}

        def mock_detect(slice_path, cfg):
            captured_cfg.update(cfg)
            return fake_df

        from unittest.mock import patch
        with patch(
            "project.frontend.blueprints.api_detect_preview._run_detection",
            side_effect=mock_detect,
        ):
            client = app.test_client()
            resp = client.post(
                "/api/detect/preview",
                data=json.dumps({
                    "slicePath": str(sample_slice),
                    "jobId": "test_params",
                    "params": {
                        "model": "cyto3",
                        "diameter_um": 20.0,
                        "flow_threshold": 0.6,
                        "cellprob_threshold": -2.0,
                        "min_size_px": 15,
                    },
                }),
                content_type="application/json",
            )
            assert resp.status_code == 200
            det = captured_cfg["detection"]
            assert det["primary_model"] == "cyto3"
            assert det["cellpose_diameter_um"] == 20.0
            assert det["cellpose_flow_threshold"] == 0.6
            assert det["cellpose_cellprob_threshold"] == -2.0
            assert det["cellpose_min_size_px"] == 15

    def test_param_overrides_without_params_key_unchanged(self, app, tmp_path, sample_slice):
        """When request has no 'params', config is used as-is (backward compat)."""
        import json

        import pandas as pd

        import project.frontend.server_context as ctx

        base_cfg = {
            "detection": {"primary_model": "cpsam", "cellpose_diameter_um": 12.0},
            "input": {"pixel_size_um_xy": 5.0},
            "compute": {"device": "cpu"},
        }
        cfg_file = tmp_path / "compat_config.json"
        cfg_file.write_text(json.dumps(base_cfg))
        ctx.run_state["config_path"] = str(cfg_file)

        fake_df = pd.DataFrame({
            "x": [10.0], "y": [15.0], "detector": ["cpsam"],
        })
        captured_cfg = {}

        def mock_detect(slice_path, cfg):
            captured_cfg.update(cfg)
            return fake_df

        from unittest.mock import patch
        with patch(
            "project.frontend.blueprints.api_detect_preview._run_detection",
            side_effect=mock_detect,
        ):
            client = app.test_client()
            resp = client.post(
                "/api/detect/preview",
                data=json.dumps({
                    "slicePath": str(sample_slice),
                    "jobId": "test_compat",
                }),
                content_type="application/json",
            )
            assert resp.status_code == 200
            assert captured_cfg["detection"]["primary_model"] == "cpsam"
            assert captured_cfg["detection"]["cellpose_diameter_um"] == 12.0

    def test_param_override_model_reaches_detector(self, app, tmp_path, sample_slice):
        """Full chain: param override 'model' changes which model string
        reaches the actual detect_cells function."""
        import json

        import pandas as pd

        import project.frontend.server_context as ctx

        # Config says cpsam, but param override says cyto3
        base_cfg = {
            "detection": {"primary_model": "cpsam"},
            "input": {"pixel_size_um_xy": 5.0},
            "compute": {"device": "cpu"},
        }
        cfg_file = tmp_path / "override_config.json"
        cfg_file.write_text(json.dumps(base_cfg))
        ctx.run_state["config_path"] = str(cfg_file)

        fake_df = pd.DataFrame({
            "x": [10.0], "y": [15.0], "detector": ["cellpose_cyto3"],
        })
        captured_cfg = {}

        def mock_detect(slice_path, cfg):
            captured_cfg.update(cfg)
            return fake_df

        from unittest.mock import patch
        with patch(
            "project.frontend.blueprints.api_detect_preview._run_detection",
            side_effect=mock_detect,
        ):
            client = app.test_client()
            resp = client.post(
                "/api/detect/preview",
                data=json.dumps({
                    "slicePath": str(sample_slice),
                    "jobId": "test_override_model",
                    "params": {"model": "cyto3"},
                }),
                content_type="application/json",
            )
            assert resp.status_code == 200
            # Config passed to detector must have the overridden model
            assert captured_cfg["detection"]["primary_model"] == "cyto3"
