"""Neuroglancer integration endpoints.

Exposes the ported Xu Lab Neuroglancer viewer + OME-Zarr converter to the
browser UI:

- ``POST /api/neuroglancer/convert`` — convert a NIfTI/TIFF-dir/TIFF file
  to OME-Zarr for viewing.
- ``POST /api/neuroglancer/launch`` — start a self-hosted Neuroglancer
  server for a set of image / segmentation / point inputs and return the
  viewer URL the browser can open.
- ``GET  /api/neuroglancer/available`` — lightweight probe returning
  whether the ``neuroglancer``/``ome-zarr``/``zarr`` deps are installed,
  so the UI can hide the button cleanly when they aren't.

The Neuroglancer Python package runs its own tornado server on a separate
port (auto-picked). We keep the launched viewer object alive in a
module-level dict so the server doesn't GC it mid-session.
"""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, request

from project.frontend.api_errors import ERR_INVALID_INPUT, ERR_NOT_FOUND

bp = Blueprint("api_neuroglancer", __name__, url_prefix="/api/neuroglancer")


_LIVE_VIEWERS: dict[str, object] = {}


def _deps_available() -> tuple[bool, str | None]:
    try:
        import neuroglancer  # noqa: F401
        import ome_zarr  # noqa: F401
        import zarr  # noqa: F401
    except ImportError as exc:
        return False, str(exc)
    return True, None


@bp.get("/available")
def ng_available():
    ok, err = _deps_available()
    return jsonify(
        {
            "ok": True,
            "available": ok,
            "missing": (err.split("No module named ")[-1].strip("'\"") if err else None),
            "install": 'pip install -e ".[neuroglancer]"',
        }
    )


@bp.post("/convert")
def ng_convert():
    """Convert a NIfTI or TIFF input to OME-Zarr for Neuroglancer viewing."""
    payload = request.get_json(silent=True) or {}
    raw_input = (payload.get("inputPath") or "").strip()
    raw_output = (payload.get("outputDir") or "").strip()
    if not raw_input or not raw_output:
        return jsonify(
            {"ok": False, "error": "missing inputPath/outputDir", "error_code": ERR_INVALID_INPUT}
        ), 400

    inp = Path(raw_input).expanduser()
    out = Path(raw_output).expanduser()
    if not inp.exists():
        return jsonify(
            {"ok": False, "error": f"input not found: {inp}", "error_code": ERR_NOT_FOUND}
        ), 404

    ok, err = _deps_available()
    if not ok:
        return jsonify(
            {
                "ok": False,
                "error": 'Neuroglancer deps missing. Install with: pip install -e ".[neuroglancer]"',
                "missing_module": (err.split("No module named ")[-1].strip("'\"") if err else None),
            }
        ), 501

    from project.scripts.ng_converter import convert_auto

    spacing_um = payload.get("spacingUm")
    if spacing_um is not None:
        spacing_um = tuple(float(v) for v in spacing_um)

    try:
        zarr_path = convert_auto(str(inp), str(out), spacing_um=spacing_um)
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"convert failed: {exc}"}), 500

    return jsonify({"ok": True, "zarrPath": zarr_path})


@bp.post("/launch")
def ng_launch():
    """Launch a Neuroglancer viewer for given inputs; return the viewer URL."""
    payload = request.get_json(silent=True) or {}

    ok, err = _deps_available()
    if not ok:
        return jsonify(
            {
                "ok": False,
                "error": 'Neuroglancer deps missing. Install with: pip install -e ".[neuroglancer]"',
                "missing_module": (err.split("No module named ")[-1].strip("'\"") if err else None),
            }
        ), 501

    image_inputs = payload.get("imageInputs") or []
    points_inputs = payload.get("pointsInputs") or []
    segmentation_path = payload.get("segmentationPath") or None
    spacing_um = payload.get("spacingUm")
    if spacing_um is not None:
        spacing_um = tuple(float(v) for v in spacing_um)
    bind_address = str(payload.get("bindAddress") or "127.0.0.1")
    port = int(payload.get("port") or 0)
    session_id = str(payload.get("sessionId") or f"ng_{len(_LIVE_VIEWERS) + 1}")

    # Never open browser from the server side; the client does that with
    # the returned URL (we're a headless Flask worker).
    from project.scripts.ng_viewer import launch_viewer

    try:
        viewer = launch_viewer(
            image_inputs=image_inputs,
            points_inputs=points_inputs,
            segmentation_path=segmentation_path,
            spacing_um=spacing_um,
            bind_address=bind_address,
            port=port,
            open_browser=False,
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"viewer launch failed: {exc}"}), 500

    _LIVE_VIEWERS[session_id] = viewer
    url = viewer.get_viewer_url()
    return jsonify({"ok": True, "sessionId": session_id, "url": url})


@bp.post("/stop")
def ng_stop():
    """Drop a live viewer reference so its tornado server can be GC'd."""
    payload = request.get_json(silent=True) or {}
    session_id = str(payload.get("sessionId") or "")
    if not session_id or session_id not in _LIVE_VIEWERS:
        return jsonify({"ok": False, "error": "unknown sessionId"}), 404
    _LIVE_VIEWERS.pop(session_id, None)
    return jsonify({"ok": True, "sessionId": session_id})
