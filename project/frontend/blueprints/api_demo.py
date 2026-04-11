"""api_demo.py - Demo/chart routes."""

from __future__ import annotations

import csv
import subprocess
import sys
import threading
from pathlib import Path

from flask import Blueprint, jsonify, send_from_directory

import project.frontend.server_context as ctx
from project.frontend.services.demo_service import generate_cell_chart, generate_demo_comparison

bp = Blueprint("api_demo", __name__, url_prefix="/api")


def _active_demo_paths() -> tuple[Path, Path | None]:
    out_dir = ctx.active_output_dir()
    return out_dir, ctx.latest_input_dir(out_dir)


@bp.get("/outputs/demo-best-slice")
def outputs_demo_best_slice():
    """Serve the pre-generated best-slice comparison image."""
    out_dir, _raw_dir = _active_demo_paths()
    fp = out_dir / "demo_best_slice.jpg"
    if not fp.exists():
        return jsonify({"ok": False, "error": "Best-slice image not generated yet"}), 404
    return send_from_directory(str(out_dir), fp.name)


@bp.get("/outputs/demo-annotated-slice")
def outputs_demo_annotated_slice():
    """Serve the annotated single-slice with region labels."""
    out_dir, _raw_dir = _active_demo_paths()
    fp = out_dir / "demo_annotated_slice.jpg"
    if not fp.exists():
        return jsonify(
            {"ok": False, "error": "Annotated slice not generated yet. Run refresh_demo.py first."}
        ), 404
    return send_from_directory(str(out_dir), fp.name)


@bp.get("/outputs/cell-chart")
def outputs_cell_chart():
    """Generate and serve the cell-count bar+pie chart."""
    out_dir, _raw_dir = _active_demo_paths()
    chart_path = out_dir / "cell_count_chart.png"
    hier_path = out_dir / "cell_counts_hierarchy.csv"
    if not hier_path.exists():
        return jsonify({"ok": False, "error": "No hierarchy CSV yet"}), 404
    if not chart_path.exists() or hier_path.stat().st_mtime > chart_path.stat().st_mtime:
        try:
            generate_cell_chart(hier_path, chart_path, ctx.PROJECT_ROOT)
        except Exception as exc:
            return jsonify({"ok": False, "error": f"Chart generation failed: {exc}"}), 500
    return send_from_directory(str(out_dir), chart_path.name)


@bp.get("/outputs/demo-comparison/<int:slice_idx>")
def outputs_demo_comparison(slice_idx: int):
    """Generate and serve a side-by-side raw vs atlas comparison for a given slice index."""
    out_dir, raw_dir = _active_demo_paths()
    reg_dir = out_dir / "registered_slices"
    ov_path = reg_dir / f"slice_{slice_idx:04d}_overlay.png"
    if not ov_path.exists():
        return jsonify({"ok": False, "error": f"slice {slice_idx} not found"}), 404

    out_path = out_dir / f"compare_{slice_idx:04d}.jpg"
    try:
        generate_demo_comparison(slice_idx, reg_dir, raw_dir, out_path)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
    return send_from_directory(str(out_dir), out_path.name)


@bp.get("/outputs/demo-panel")
def outputs_demo_panel():
    """Serve the demo panel image, generating it on-demand if needed."""
    out_dir, raw_dir = _active_demo_paths()
    panel_path = out_dir / "demo_panel.jpg"
    reg_dir = out_dir / "registered_slices"
    needs_refresh = not panel_path.exists() or (
        reg_dir.exists()
        and any(
            p.stat().st_mtime > panel_path.stat().st_mtime
            for p in reg_dir.glob("slice_*_overlay.png")
        )
    )
    if needs_refresh:
        try:
            script = ctx.PROJECT_ROOT / "scripts" / "refresh_demo.py"
            cmd = [sys.executable, str(script), "--outputs-dir", str(out_dir)]
            if raw_dir is not None:
                cmd.extend(["--raw-dir", str(raw_dir)])
            subprocess.run(
                cmd,
                cwd=str(ctx.PROJECT_ROOT),
                timeout=180,
                check=True,
                capture_output=True,
                text=True,
            )
        except Exception as exc:
            return jsonify({"ok": False, "error": f"Panel generation failed: {exc}"}), 500
    if not panel_path.exists():
        return jsonify({"ok": False, "error": "Panel not found"}), 404
    return send_from_directory(str(out_dir), panel_path.name)


@bp.post("/outputs/refresh-demo")
def outputs_refresh_demo():
    """Run refresh_demo.py to regenerate all demo visuals for the active output dir."""
    script = ctx.PROJECT_ROOT / "scripts" / "refresh_demo.py"
    if not script.exists():
        return jsonify({"ok": False, "error": "refresh_demo.py not found"}), 404

    out_dir, raw_dir = _active_demo_paths()

    def _run():
        try:
            cmd = [sys.executable, str(script), "--outputs-dir", str(out_dir)]
            if raw_dir is not None:
                cmd.extend(["--raw-dir", str(raw_dir)])
            result = subprocess.run(
                cmd,
                cwd=str(ctx.PROJECT_ROOT),
                timeout=240,
                capture_output=True,
                text=True,
            )
            ctx._append_log(f"[refresh_demo] {result.stdout.strip()}")
            if result.returncode != 0:
                ctx._append_log(f"[refresh_demo] ERROR: {result.stderr.strip()}")
        except Exception as exc:
            ctx._append_log(f"[refresh_demo] exception: {exc}")

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True, "message": "refresh_demo.py started in background"})


@bp.get("/outputs/reg-stats")
def outputs_reg_stats():
    """Return registration quality summary for display."""
    out_dir, _raw_dir = _active_demo_paths()
    qc_path = out_dir / "slice_registration_qc.csv"
    if not qc_path.exists():
        return jsonify({"ok": False, "error": "No registration QC data yet"})
    try:
        with open(qc_path, newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        scores = [float(row["best_score"]) for row in rows if row.get("best_score")]
        ok_count = sum(1 for row in rows if row.get("registration_ok", "").lower() == "true")
        return jsonify(
            {
                "ok": True,
                "total": len(rows),
                "ok_count": ok_count,
                "mean_score": round(sum(scores) / len(scores), 3) if scores else 0,
                "min_score": round(min(scores), 3) if scores else 0,
                "max_score": round(max(scores), 3) if scores else 0,
            }
        )
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 500
