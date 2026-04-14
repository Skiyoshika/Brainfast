"""api_detect_preview.py — Detector preview route for single-slice cell detection.

Lets the user inspect Cellpose/LoG detection output on the current preview
slice before committing to a full batch run.  Returns an overlay image and
a machine-readable CSV of detected cells.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request, send_file
from PIL import Image
from tifffile import imread

import project.frontend.server_context as ctx

bp = Blueprint("api_detect_preview", __name__, url_prefix="/api")

# ---------------------------------------------------------------------------
# In-memory cache for the last detection result per job
# ---------------------------------------------------------------------------
_detect_results: dict[str, dict] = {}
_detect_lock = threading.Lock()


def _load_active_config() -> dict:
    """Load the active run config from the config file on disk.

    Priority:
      1. ``run_state["config_path"]`` — set when the user last ran the pipeline.
      2. ``run_config.template.json`` — explicit fallback when no prior run exists.

    Never falls back to a hard-coded mini-config; always reads from a real file
    so that all detector knobs (primary_model, cellpose_channels, diameter,
    thresholds, pixel_size, etc.) are preserved.
    """
    import json

    config_path_str = ctx.run_state.get("config_path")
    candidates = []
    if config_path_str:
        candidates.append(Path(config_path_str))
    candidates.append(ctx.PROJECT_ROOT / "configs" / "run_config.template.json")

    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text(encoding="utf-8-sig"))
            except Exception:
                continue
    # Should not happen — template always ships with the repo
    return {}


def _apply_param_overrides(cfg: dict, params: dict) -> dict:
    """Merge user-supplied parameter overrides into the detection config.

    Maps frontend param names to config keys:
        model          → detection.primary_model
        diameter_um    → detection.cellpose_diameter_um
        flow_threshold → detection.cellpose_flow_threshold
        cellprob_threshold → detection.cellpose_cellprob_threshold
        min_size_px    → detection.cellpose_min_size_px
        gpu            → detection.cellpose_gpu
    """
    if not params:
        return cfg

    det = dict(cfg.get("detection", {}))

    _PARAM_MAP = {
        "model": "primary_model",
        "diameter_um": "cellpose_diameter_um",
        "flow_threshold": "cellpose_flow_threshold",
        "cellprob_threshold": "cellpose_cellprob_threshold",
        "min_size_px": "cellpose_min_size_px",
        "gpu": "cellpose_gpu",
    }

    for param_key, config_key in _PARAM_MAP.items():
        if param_key in params:
            det[config_key] = params[param_key]

    cfg = dict(cfg)
    cfg["detection"] = det
    return cfg


def _run_detection(slice_path: Path, cfg: dict) -> pd.DataFrame:
    """Run cell detection on a single slice using the configured detector."""
    try:
        from project.scripts.detect import detect_cells
    except ImportError:
        from scripts.detect import detect_cells
    return detect_cells(slice_path, cfg)


def _make_overlay(img: np.ndarray, cells_df: pd.DataFrame) -> Image.Image:
    """Draw detected cell centroids on the image as an RGBA overlay."""
    from project.scripts.image_utils import norm_u8_robust

    gray = norm_u8_robust(img)
    rgba = np.stack([gray, gray, gray, np.full_like(gray, 255)], axis=-1)

    # Draw circles at each detected centroid
    for _, row in cells_df.iterrows():
        cx, cy = int(round(row["x"])), int(round(row["y"]))
        r = 4
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    py, px = cy + dy, cx + dx
                    if 0 <= py < rgba.shape[0] and 0 <= px < rgba.shape[1]:
                        # Green dot with slight transparency
                        rgba[py, px] = [0, 255, 0, 200]

    return Image.fromarray(rgba, "RGBA")


@bp.post("/detect/preview")
def detect_preview():
    """Run detector on a single slice and return results.

    Request JSON:
        slicePath: path to the TIFF slice
        jobId: (optional) job ID for isolation

    Response JSON:
        ok: bool
        cellCount: int
        detector: str (e.g. "cellpose_cpsam", "fallback_log")
        overlayUrl: str (URL to fetch the overlay image)
        csvUrl: str (URL to fetch the CSV)
    """
    payload = request.get_json(force=True)
    job_id = ctx._payload_job_id(payload)
    slice_path_str = payload.get("slicePath", "")

    if not slice_path_str:
        return jsonify({"ok": False, "error": "slicePath is required"}), 400

    slice_path = Path(slice_path_str)
    if not slice_path.is_absolute():
        slice_path = ctx.PROJECT_ROOT / slice_path
    if not slice_path.exists():
        return jsonify({"ok": False, "error": f"slice not found: {slice_path}"}), 404

    # Build detection config from the active config file.
    # run_state stores "config_path" (a file path string), NOT a parsed config.
    cfg = _load_active_config()

    # Apply user parameter overrides from the request
    params = payload.get("params")
    if params:
        cfg = _apply_param_overrides(cfg, params)

    try:
        cells_df = _run_detection(slice_path, cfg)
    except Exception as exc:
        return jsonify(
            {
                "ok": False,
                "error": str(exc),
                "runtimeAvailable": False,
            }
        ), 500

    cell_count = len(cells_df)
    detector = cells_df["detector"].iloc[0] if cell_count > 0 else "none"

    # Generate overlay image
    try:
        img = imread(str(slice_path))
        if img.ndim == 3:
            img = img[..., 0]
        overlay = _make_overlay(img, cells_df)
    except Exception:
        overlay = None

    # Save results
    out_dir = ctx._job_output_dir(job_id)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / "detect_preview.csv"
    cells_df.to_csv(csv_path, index=False)

    overlay_path = out_dir / "detect_preview.png"
    if overlay is not None:
        overlay.save(str(overlay_path))

    with _detect_lock:
        _detect_results[job_id] = {
            "cellCount": cell_count,
            "detector": detector,
            "csvPath": str(csv_path),
            "overlayPath": str(overlay_path) if overlay is not None else None,
        }

    return jsonify(
        {
            "ok": True,
            "cellCount": cell_count,
            "detector": detector,
            "overlayUrl": f"/api/detect/preview/overlay?jobId={job_id}",
            "csvUrl": f"/api/detect/preview/csv?jobId={job_id}",
        }
    )


@bp.get("/detect/preview/overlay")
def detect_preview_overlay():
    """Return the detection overlay image."""
    job_id = ctx._query_job_id()
    path = ctx._job_file(job_id, "detect_preview.png")
    if not path.exists():
        return jsonify({"ok": False, "error": "no detection preview available"}), 404
    return send_file(str(path), mimetype="image/png")


@bp.get("/detect/preview/csv")
def detect_preview_csv():
    """Return the detection results as CSV."""
    job_id = ctx._query_job_id()
    path = ctx._job_file(job_id, "detect_preview.csv")
    if not path.exists():
        return jsonify({"ok": False, "error": "no detection preview available"}), 404
    return send_file(str(path), mimetype="text/csv", as_attachment=True, download_name="detect_preview.csv")


@bp.get("/detect/preview/status")
def detect_preview_status():
    """Return the last detection preview result for this job."""
    job_id = ctx._query_job_id()
    with _detect_lock:
        result = _detect_results.get(job_id)
    if result is None:
        return jsonify({"ok": False, "available": False})
    return jsonify({"ok": True, "available": True, **result})
