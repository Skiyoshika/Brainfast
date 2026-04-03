"""api_outputs.py — Read-only GET routes for CSV / file serving."""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, Response, jsonify, send_from_directory

import project.frontend.server_context as ctx

bp = Blueprint("api_outputs", __name__, url_prefix="/api/outputs")


@bp.get("/leaf")
def outputs_leaf():
    fp = ctx.active_output_dir() / "cell_counts_leaf.csv"
    if not fp.exists():
        return jsonify({"ok": False, "error": "output not found"}), 404
    return send_from_directory(fp.parent, fp.name)


@bp.get("/leaf/<channel>")
def outputs_leaf_channel(channel: str):
    fp = ctx.active_output_dir() / f"cell_counts_leaf_{channel}.csv"
    if not fp.exists():
        return Response("", mimetype="text/csv")
    return send_from_directory(fp.parent, fp.name)


@bp.get("/hierarchy")
def outputs_hierarchy():
    fp = ctx.active_output_dir() / "cell_counts_hierarchy.csv"
    if not fp.exists():
        return jsonify({"ok": False, "error": "hierarchy output not found"}), 404
    return send_from_directory(fp.parent, fp.name)


@bp.get("/registration-qc")
def outputs_registration_qc():
    fp = ctx.active_output_dir() / "slice_registration_qc.csv"
    if not fp.exists():
        return jsonify({"ok": False, "error": "registration QC not found"}), 404
    return send_from_directory(fp.parent, fp.name)


@bp.get("/volume-reg-stats")
def outputs_volume_reg_stats():
    fp = ctx.active_output_dir() / "volume_registration_qc.csv"
    if not fp.exists():
        return jsonify({"ok": False, "error": "volume QC not found"}), 404
    return send_from_directory(str(fp.parent), fp.name)


@bp.get("/reg-slice-list")
def outputs_reg_slice_list():
    reg_dir = ctx.active_output_dir() / "registered_slices"
    if not reg_dir.exists():
        return jsonify({"ok": True, "files": [], "count": 0})
    files = sorted(reg_dir.glob("slice_*_overlay.png"))
    return jsonify({"ok": True, "files": [f.name for f in files], "count": len(files)})


@bp.get("/reg-slice/<filename>")
def outputs_reg_slice_file(filename: str):
    reg_dir = ctx.active_output_dir() / "registered_slices"
    safe = Path(filename).name
    fp = reg_dir / safe
    if not fp.exists() or not safe.endswith(".png"):
        return jsonify({"ok": False, "error": "file not found"}), 404
    return send_from_directory(str(reg_dir), safe)


@bp.get("/file-list")
def outputs_file_list():
    out_dir = ctx.active_output_dir()
    if not out_dir.exists():
        return jsonify({"ok": True, "files": [], "dir": str(out_dir)})
    files = []
    for f in sorted(out_dir.iterdir()):
        if f.is_file():
            files.append({"name": f.name, "size": f.stat().st_size, "ext": f.suffix.lower()})
        elif f.is_dir():
            # Include first-level subdir files (e.g. paper_report/)
            for sf in sorted(f.iterdir()):
                if sf.is_file():
                    files.append({
                        "name": f"{f.name}/{sf.name}",
                        "size": sf.stat().st_size,
                        "ext": sf.suffix.lower(),
                    })
    return jsonify({"ok": True, "files": files, "dir": str(out_dir)})


@bp.get("/named/<path:filename>")
def outputs_named(filename: str):
    out_dir = ctx.active_output_dir()
    resolved_root = out_dir.resolve()
    fp = (resolved_root / Path(filename)).resolve()
    if not fp.is_relative_to(resolved_root) or not fp.exists() or not fp.is_file():
        return jsonify({"ok": False, "error": "file not found"}), 404
    return send_from_directory(str(fp.parent), fp.name)


@bp.get("/qc-list")
def outputs_qc_list():
    qc_dir = ctx.active_output_dir() / "qc_overlays"
    if not qc_dir.exists():
        return jsonify({"ok": True, "files": [], "count": 0})
    files = sorted(qc_dir.glob("overlay_*.png"))
    return jsonify({"ok": True, "files": [f.name for f in files], "count": len(files)})


@bp.get("/qc-file/<filename>")
def outputs_qc_file(filename: str):
    qc_dir = ctx.active_output_dir() / "qc_overlays"
    safe = Path(filename).name
    return send_from_directory(str(qc_dir), safe)


@bp.post("/open-folder")
def outputs_open_folder():
    out_dir = ctx.active_output_dir()
    if not out_dir.exists():
        return jsonify({"ok": False, "error": "output folder not found", "path": str(out_dir)}), 404
    try:
        ctx.open_folder_in_shell(out_dir)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "path": str(out_dir)}), 500
    return jsonify({"ok": True, "path": str(out_dir)})
