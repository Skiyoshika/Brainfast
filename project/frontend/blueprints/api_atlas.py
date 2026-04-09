"""api_atlas.py — Atlas autopick routes."""

from __future__ import annotations

import os
import threading
import uuid as _uuid_mod
from pathlib import Path

from flask import Blueprint, jsonify, request

import project.frontend.server_context as ctx

bp = Blueprint("api_atlas", __name__, url_prefix="/api/atlas")


def _normalize_path(p: str) -> str:
    """Normalize Windows paths: resolve double-backslashes, mixed separators."""
    return os.path.normpath(p) if p else p


@bp.post("/autopick-z")
def atlas_autopick_z():
    payload = request.get_json(force=True)
    real_path = Path(_normalize_path(payload.get("realPath", "")))
    annotation_path = Path(_normalize_path(payload.get("annotationPath", "")))
    z_step = int(payload.get("zStep", 1))
    raw_real_z = payload.get("realZIndex", None)
    real_z_index = None if raw_real_z in (None, "", "null") else int(raw_real_z)
    pixel_size_um = float(payload.get("pixelSizeUm", 0.65))
    slicing_plane = str(payload.get("slicingPlane", "coronal"))
    roi_mode = str(payload.get("roiMode", "auto"))
    if not real_path.exists():
        return jsonify({"ok": False, "error": f"Real image not found: {real_path}"}), 400
    if not annotation_path.exists():
        return jsonify(
            {"ok": False, "error": f"Atlas annotation not found: {annotation_path}"}
        ), 400

    job_id = ctx._payload_job_id(payload)
    out_label = ctx._job_file(job_id, "auto_label_slice.tif")
    token = _uuid_mod.uuid4().hex

    ctx._autopick_tasks[token] = {
        "status": "pending",
        "progress": 0,
        "step": 0,
        "total": 100,
        "message": "Queued...",
        "result": None,
        "error": None,
        "jobId": job_id,
    }

    ap_method = str(payload.get("apMethod", "auto"))
    raw_ap_start = payload.get("apRangeStart", None)
    raw_ap_end = payload.get("apRangeEnd", None)
    ap_range = None
    if raw_ap_start is not None and raw_ap_end is not None:
        ap_range = (int(raw_ap_start), int(raw_ap_end))
    kwargs = dict(
        z_step=z_step,
        pixel_size_um=pixel_size_um,
        slicing_plane=slicing_plane,
        roi_mode=roi_mode,
        real_z_index=real_z_index,
        ap_method=ap_method,
        ap_range=ap_range,
    )
    t = threading.Thread(
        target=ctx._run_autopick_worker,
        args=(token, real_path, annotation_path, out_label, kwargs),
        daemon=True,
    )
    t.start()
    return jsonify({"ok": True, "token": token, "jobId": job_id, "started": True})


@bp.get("/autopick-z/status")
def atlas_autopick_z_status():
    token = request.args.get("token", "")
    if not token or token not in ctx._autopick_tasks:
        return jsonify({"ok": False, "error": "unknown token"}), 404
    task = ctx._autopick_tasks[token]
    resp = {
        "ok": True,
        "token": token,
        "status": task["status"],
        "progress": task["progress"],
        "step": task["step"],
        "total": task["total"],
        "message": task["message"],
        "jobId": task.get("jobId", ""),
    }
    if task["status"] == "done":
        resp["result"] = task["result"]
    if task["status"] == "error":
        resp["error"] = task["error"]
    return jsonify(resp)


@bp.get("/region-ap-ranges")
def region_ap_ranges():
    """Return navigable brain regions with their AP slice ranges."""
    import json as _json

    json_path = Path(__file__).resolve().parent.parent.parent / "configs" / "region_ap_ranges.json"
    if not json_path.exists():
        return jsonify({"ok": False, "error": "region_ap_ranges.json not found"}), 404
    with open(json_path) as f:
        regions = _json.load(f)
    return jsonify({"ok": True, "regions": regions})
