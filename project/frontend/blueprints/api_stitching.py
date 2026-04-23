"""Stitching pipeline endpoints.

Exposes the ported Xu Lab TissueCyte stitching pipeline to the Brainfast
UI / CLI. The heavy deps (``opencv-python``, ``colorama``, ``joblib``) are
optional so we probe their availability and lazy-import inside the
endpoint handlers.

- ``GET  /api/stitching/available`` — dependency probe for the UI.
- ``POST /api/stitching/start``     — kick off a stitch job in a background
  thread (non-blocking; status is tracked in ``_JOBS``).
- ``GET  /api/stitching/status``    — poll status/progress for a job id.

Runs as fire-and-forget from the browser's perspective; the stitching core
writes console output via ``_Tee`` to ``console.log`` inside the output
directory, which the UI can tail.
"""

from __future__ import annotations

import threading
import time
import traceback
import uuid
from pathlib import Path

from flask import Blueprint, jsonify, request

from project.frontend.api_errors import ERR_INVALID_INPUT, ERR_NOT_FOUND

bp = Blueprint("api_stitching", __name__, url_prefix="/api/stitching")


_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()


def _deps_available() -> tuple[bool, str | None]:
    try:
        import cv2  # noqa: F401
        import colorama  # noqa: F401
        import joblib  # noqa: F401
    except ImportError as exc:
        return False, str(exc)
    return True, None


@bp.get("/available")
def stitching_available():
    ok, err = _deps_available()
    return jsonify({
        "ok": True,
        "available": ok,
        "missing": (err.split("No module named ")[-1].strip("'\"") if err else None),
        "install": "pip install -e \".[stitching]\"",
    })


def _run_stitch_job(job_id: str, input_dir: str, output_dir: str, opts: dict) -> None:
    """Worker: run stitch_pipeline in a background thread; update _JOBS in-place."""
    with _JOBS_LOCK:
        _JOBS[job_id]["status"] = "running"
        _JOBS[job_id]["started_at"] = time.time()
    try:
        from scripts.stitching.pipeline import stitch_pipeline

        stitch_pipeline(
            input_dir=input_dir,
            output_dir=output_dir,
            section_num=int(opts.get("section_num", -1)),
            channel=opts.get("channel"),
            bezier_path=opts.get("bezier_path"),
            n_threads=int(opts.get("n_threads", -3)),
            save_undistorted=bool(opts.get("save_undistorted", False)),
            vignetting_correction=bool(opts.get("vignetting_correction", True)),
            verbose=True,
        )
        with _JOBS_LOCK:
            _JOBS[job_id]["status"] = "done"
            _JOBS[job_id]["finished_at"] = time.time()
    except Exception as exc:  # noqa: BLE001
        with _JOBS_LOCK:
            _JOBS[job_id]["status"] = "error"
            _JOBS[job_id]["error"] = str(exc)
            _JOBS[job_id]["traceback"] = traceback.format_exc()
            _JOBS[job_id]["finished_at"] = time.time()


@bp.post("/start")
def stitching_start():
    """Kick off a stitch job. Returns jobId immediately; status via /status."""
    payload = request.get_json(silent=True) or {}
    input_dir = (payload.get("inputDir") or "").strip()
    output_dir = (payload.get("outputDir") or "").strip()
    if not input_dir or not output_dir:
        return jsonify({"ok": False, "error": "missing inputDir/outputDir",
                        "error_code": ERR_INVALID_INPUT}), 400

    inp = Path(input_dir).expanduser()
    if not inp.exists() or not inp.is_dir():
        return jsonify({"ok": False, "error": f"inputDir not found or not a directory: {inp}",
                        "error_code": ERR_NOT_FOUND}), 404

    ok, err = _deps_available()
    if not ok:
        return jsonify({
            "ok": False,
            "error": "Stitching deps missing. Install with: pip install -e \".[stitching]\"",
            "missing_module": (err.split("No module named ")[-1].strip("'\"") if err else None),
        }), 501

    opts = {
        "section_num": int(payload.get("sectionNum", -1)),
        "channel": payload.get("channel"),
        "bezier_path": payload.get("bezierPath"),
        "n_threads": int(payload.get("nThreads", -3)),
        "save_undistorted": bool(payload.get("saveUndistorted", False)),
        "vignetting_correction": bool(payload.get("vignettingCorrection", True)),
    }

    job_id = f"stitch_{uuid.uuid4().hex[:8]}"
    with _JOBS_LOCK:
        _JOBS[job_id] = {
            "job_id": job_id,
            "input_dir": str(inp),
            "output_dir": str(Path(output_dir).expanduser()),
            "status": "queued",
            "opts": opts,
            "created_at": time.time(),
        }

    t = threading.Thread(
        target=_run_stitch_job,
        args=(job_id, str(inp), str(Path(output_dir).expanduser()), opts),
        daemon=True,
    )
    t.start()

    return jsonify({"ok": True, "jobId": job_id, "status": "queued"})


@bp.get("/status")
def stitching_status():
    job_id = request.args.get("jobId", "").strip()
    if not job_id:
        return jsonify({"ok": False, "error": "missing jobId"}), 400
    with _JOBS_LOCK:
        job = _JOBS.get(job_id)
    if not job:
        return jsonify({"ok": False, "error": "unknown jobId"}), 404
    return jsonify({"ok": True, **job})
