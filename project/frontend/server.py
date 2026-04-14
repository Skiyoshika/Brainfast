"""server.py — Thin orchestrator: sets up paths, imports blueprints, creates Flask app.

All route logic lives in project/frontend/blueprints/*.
All shared state and helpers live in project/frontend/server_context.py.
"""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

from flask import Flask, jsonify, request
from werkzeug.exceptions import RequestEntityTooLarge

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    ROOT = Path(sys._MEIPASS)
    _proj = Path(sys.executable).resolve().parent
    PROJECT_ROOT = _proj.parent if str(_proj).endswith("frontend") else _proj
else:
    ROOT = Path(__file__).resolve().parent
    PROJECT_ROOT = ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Ensure the parent of PROJECT_ROOT is on sys.path so that
# `import project.frontend.server_context` resolves correctly from blueprints.
_repo_root = str(PROJECT_ROOT.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from project.frontend.app_metadata import resolve_max_content_length
from project.scripts.paths import ensure_runtime_cache_dirs

ensure_runtime_cache_dirs(PROJECT_ROOT)

# Patch context paths before any blueprint imports so helpers resolve correctly.
import project.frontend.server_context as ctx
from project.scripts.asset_bootstrap import default_structure_source

ctx.ROOT = ROOT
ctx.PROJECT_ROOT = PROJECT_ROOT
ctx.OUTPUT_DIR = PROJECT_ROOT / "outputs"
ctx.DEFAULT_STRUCTURE_SOURCE = default_structure_source(PROJECT_ROOT) or (
    PROJECT_ROOT / "configs" / "allen_structure_tree.json"
)

from project.frontend.blueprints.api_alignment import bp as alignment_bp
from project.frontend.blueprints.api_atlas import bp as atlas_bp
from project.frontend.blueprints.api_batch import bp as batch_bp
from project.frontend.blueprints.api_browse import bp as browse_bp
from project.frontend.blueprints.api_cellpose import bp as cellpose_bp
from project.frontend.blueprints.api_compare import bp as compare_bp
from project.frontend.blueprints.api_demo import bp as demo_bp
from project.frontend.blueprints.api_detect_preview import bp as detect_preview_bp
from project.frontend.blueprints.api_docs import bp as docs_bp
from project.frontend.blueprints.api_outputs import bp as outputs_bp
from project.frontend.blueprints.api_overlay import bp as overlay_bp
from project.frontend.blueprints.api_pipeline import bp as pipeline_bp
from project.frontend.blueprints.api_projects import bp as projects_bp
from project.frontend.blueprints.api_training import bp as training_bp


def create_app() -> Flask:
    app = Flask(__name__, static_folder=str(ROOT), static_url_path="")
    app.config["MAX_CONTENT_LENGTH"] = resolve_max_content_length()
    app.register_blueprint(pipeline_bp)
    app.register_blueprint(atlas_bp)
    app.register_blueprint(overlay_bp)
    app.register_blueprint(alignment_bp)
    app.register_blueprint(outputs_bp)
    app.register_blueprint(training_bp)
    app.register_blueprint(demo_bp)
    app.register_blueprint(browse_bp)
    app.register_blueprint(cellpose_bp)
    app.register_blueprint(compare_bp)
    app.register_blueprint(batch_bp)
    app.register_blueprint(detect_preview_bp)
    app.register_blueprint(docs_bp)
    app.register_blueprint(projects_bp)

    @app.errorhandler(RequestEntityTooLarge)
    def _handle_request_too_large(_exc):
        limit = int(app.config.get("MAX_CONTENT_LENGTH") or 0)
        if request.path.startswith("/api/"):
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": "request body too large",
                        "limitBytes": limit,
                    }
                ),
                413,
            )
        return (f"Request body too large (limit: {limit} bytes)", 413)

    @app.before_request
    def _normalize_path_params():
        """Normalize Windows paths in query params and JSON bodies to prevent
        double-backslash or mixed-separator issues."""
        # Normalize query-string 'path' parameter (used by /slice/info, /thumbnail, etc.)
        if "path" in request.args:
            raw = request.args.get("path", "")
            if raw:
                normed = os.path.normpath(raw)
                if normed != raw:
                    # Replace the immutable args dict with normalized version
                    from werkzeug.datastructures import ImmutableMultiDict

                    args = request.args.to_dict(flat=False)
                    args["path"] = [normed]
                    request.args = ImmutableMultiDict(args)

    @app.get("/favicon.ico")
    def _favicon():
        return ("", 204)

    return app


app = create_app()


def ensure_port_available(port: int, host: str = "127.0.0.1") -> None:
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind((host, int(port)))
    except OSError as exc:
        raise RuntimeError(
            f"Port {port} is already in use on {host}. "
            "Close the existing Brainfast server before starting another instance."
        ) from exc
    finally:
        probe.close()


def main():
    port = int(os.environ.get("BRAINFAST_PORT", "8787"))
    ensure_port_available(port)
    app.run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
