"""Unit tests for the api_cellpose blueprint."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture()
def app(tmp_path):
    """Create a minimal Flask app with the cellpose blueprint."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "outputs").mkdir()

    import project.frontend.server_context as ctx

    ctx.ROOT = project_root / "frontend"
    ctx.PROJECT_ROOT = project_root
    ctx.OUTPUT_DIR = project_root / "outputs"

    from flask import Flask

    from project.frontend.blueprints.api_cellpose import bp

    flask_app = Flask(__name__)
    flask_app.register_blueprint(bp)
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


class TestListModels:
    def test_returns_builtin_models(self, client):
        """Endpoint lists at least the built-in model names."""
        with patch(
            "project.frontend.blueprints.api_cellpose._get_user_models",
            return_value=[],
        ):
            resp = client.get("/api/cellpose/models")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["ok"] is True
            names = [m["name"] for m in data["models"]]
            assert "cpsam" in names
            assert "cyto3" in names
            assert "cyto2" in names
            assert "nuclei" in names
            assert all(m["type"] == "builtin" for m in data["models"] if m["name"] in names)

    def test_includes_user_models(self, client):
        """User-trained models appear with type='custom'."""
        with patch(
            "project.frontend.blueprints.api_cellpose._get_user_models",
            return_value=["brainfast_v1_20260413", "my_custom"],
        ):
            resp = client.get("/api/cellpose/models")
            data = resp.get_json()
            names = [m["name"] for m in data["models"]]
            assert "brainfast_v1_20260413" in names
            assert "my_custom" in names
            custom_models = [m for m in data["models"] if m["type"] == "custom"]
            assert len(custom_models) == 2

    def test_handles_cellpose_import_error(self, client):
        """Endpoint returns empty list when cellpose is not installed."""
        with patch(
            "project.frontend.blueprints.api_cellpose._get_user_models",
            side_effect=ImportError("No module named 'cellpose'"),
        ):
            resp = client.get("/api/cellpose/models")
            data = resp.get_json()
            assert data["ok"] is True
            # Still returns builtins even if cellpose is not installed
            assert len(data["models"]) >= 4
