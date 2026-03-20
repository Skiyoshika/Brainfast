"""api_outputs.py - Output routes for CSV/file serving and report management."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, send_from_directory

import project.frontend.server_context as ctx

bp = Blueprint("api_outputs", __name__, url_prefix="/api/outputs")
RUN_STATE_FILE = ".registration_runs_state.json"
RUN_ARCHIVE_DIR = "archive/registration_runs"


def _outputs_root() -> Path:
    return ctx._job_output_dir(ctx._query_job_id())


def _run_state_path(outputs_root: Path | None = None) -> Path:
    root = outputs_root or _outputs_root()
    return root / RUN_STATE_FILE


def _load_run_state(outputs_root: Path | None = None) -> dict[str, object]:
    path = _run_state_path(outputs_root)
    if not path.exists():
        return {"pinned": []}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"pinned": []}
    pinned = data.get("pinned", [])
    if not isinstance(pinned, list):
        pinned = []
    pinned = [Path(str(name)).name for name in pinned if str(name).strip()]
    return {"pinned": pinned}


def _save_run_state(state: dict[str, object], outputs_root: Path | None = None) -> None:
    root = outputs_root or _outputs_root()
    root.mkdir(parents=True, exist_ok=True)
    pinned = state.get("pinned", [])
    if not isinstance(pinned, list):
        pinned = []
    data = {"pinned": [Path(str(name)).name for name in pinned if str(name).strip()]}
    _run_state_path(root).write_text(json.dumps(data, indent=2), encoding="utf-8")


def _pin_run(run_name: str, outputs_root: Path | None = None) -> list[str]:
    state = _load_run_state(outputs_root)
    pinned = [name for name in state.get("pinned", []) if name != run_name]
    pinned.insert(0, run_name)
    state["pinned"] = pinned
    _save_run_state(state, outputs_root)
    return pinned


def _unpin_run(run_name: str, outputs_root: Path | None = None) -> list[str]:
    state = _load_run_state(outputs_root)
    pinned = [name for name in state.get("pinned", []) if name != run_name]
    state["pinned"] = pinned
    _save_run_state(state, outputs_root)
    return pinned


def _load_metrics_csv(path: Path) -> dict[str, float]:
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader, None)
        out: dict[str, float] = {}
        for row in reader:
            if len(row) < 2:
                continue
            try:
                out[str(row[0])] = float(row[1])
            except Exception:
                continue
        return out


def _verdict(metrics: dict[str, float], pre_metrics: dict[str, float]) -> tuple[str, str, str]:
    if not pre_metrics:
        return (
            "Needs visual check",
            "Only final metrics are available. Check the final overview image directly.",
            "neutral",
        )

    ncc_up = metrics.get("NCC", 0.0) >= pre_metrics.get("NCC", 0.0)
    ssim_up = metrics.get("SSIM", 0.0) >= pre_metrics.get("SSIM", 0.0)
    dice_ok = metrics.get("Dice", 0.0) >= (pre_metrics.get("Dice", 0.0) - 0.005)
    psnr_up = metrics.get("PSNR", 0.0) >= pre_metrics.get("PSNR", 0.0)
    mse_down = metrics.get("MSE", 1e9) <= pre_metrics.get("MSE", 1e9)

    good = sum([ncc_up, ssim_up, dice_ok, psnr_up, mse_down])
    if good >= 4:
        return (
            "Looks improved",
            "Most quality metrics moved in the right direction. Check the final overview first.",
            "good",
        )
    if good >= 3:
        return (
            "Probably usable",
            "Metric changes are mostly positive, but you should still inspect the final overview.",
            "warn",
        )
    return (
        "Needs review",
        "Metric changes are mixed. Use the before/final comparison before accepting the run.",
        "bad",
    )


def _registration_run_dirs(outputs_root: Path | None = None) -> list[Path]:
    root = outputs_root or _outputs_root()
    if not root.exists():
        return []

    runs: list[Path] = []
    for child in root.iterdir():
        if not child.is_dir():
            continue
        if (child / "registration_metadata.json").exists() and (child / "registration_metrics.csv").exists():
            runs.append(child)
    state = _load_run_state(root)
    pinned_order = [name for name in state.get("pinned", []) if isinstance(name, str)]
    runs_by_name = {run.name: run for run in runs}
    pinned_runs = [runs_by_name[name] for name in pinned_order if name in runs_by_name]
    other_runs = [run for run in runs if run.name not in pinned_order]
    other_runs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return pinned_runs + other_runs


def _safe_registration_run_dir(run_name: str) -> Path | None:
    root = _outputs_root()
    safe = Path(run_name).name
    run_dir = root / safe
    if safe != run_name:
        return None
    if not run_dir.exists() or not run_dir.is_dir():
        return None
    if not (run_dir / "registration_metadata.json").exists():
        return None
    return run_dir


def _run_file_url(run_name: str, filename: str) -> str:
    return f"/api/outputs/registration-run/{run_name}/{filename}"


def _existing_run_url(run_dir: Path, filename: str) -> str:
    path = run_dir / filename
    if path.exists() and path.is_file():
        return _run_file_url(run_dir.name, filename)
    return ""


def _serialize_registration_run(run_dir: Path) -> dict[str, object]:
    meta_path = run_dir / "registration_metadata.json"
    metrics_path = run_dir / "registration_metrics.csv"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    metrics = _load_metrics_csv(metrics_path)
    pre_metrics = meta.get("metrics_before_laplacian", {}) or {}

    staining = meta.get("staining_stats", {}) or {}
    if not staining:
        staining_path = run_dir / "staining_stats.json"
        if staining_path.exists():
            try:
                staining = json.loads(staining_path.read_text(encoding="utf-8"))
            except Exception:
                staining = {}

    verdict_title, verdict_body, verdict_tone = _verdict(metrics, pre_metrics)
    updated_at = datetime.fromtimestamp(run_dir.stat().st_mtime).isoformat(timespec="seconds")
    input_source = str(meta.get("input_source", ""))
    input_name = Path(input_source).name if input_source else run_dir.name
    backend = str(meta.get("backend", "")).strip().upper()
    pipeline_label = backend or "UNKNOWN"
    if meta.get("laplacian_enabled"):
        pipeline_label = f"{pipeline_label} + Laplacian"
    state = _load_run_state(run_dir.parent)
    pinned = run_dir.name in state.get("pinned", [])

    return {
        "name": run_dir.name,
        "input_name": input_name,
        "input_source": input_source,
        "backend": str(meta.get("backend", "")),
        "pipeline_label": pipeline_label,
        "pinned": pinned,
        "laplacian_enabled": bool(meta.get("laplacian_enabled")),
        "hemisphere": str(meta.get("hemisphere", "")),
        "target_um": meta.get("target_um"),
        "updated_at": updated_at,
        "verdict_title": verdict_title,
        "verdict_body": verdict_body,
        "verdict_tone": verdict_tone,
        "metrics": metrics,
        "pre_metrics": pre_metrics,
        "staining_stats": staining,
        "artifacts": {
            "overview": _existing_run_url(run_dir, "overview.png"),
            "overview_before": _existing_run_url(run_dir, "overview_before.png"),
            "summary": _existing_run_url(run_dir, "registration_summary.txt"),
            "metadata": _existing_run_url(run_dir, "registration_metadata.json"),
            "metrics_csv": _existing_run_url(run_dir, "registration_metrics.csv"),
            "staining_json": _existing_run_url(run_dir, "staining_stats.json"),
            "report": _existing_run_url(run_dir, "report.html"),
        },
    }


def _archive_run_dir(run_dir: Path) -> Path:
    archive_root = run_dir.parent / RUN_ARCHIVE_DIR
    archive_root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archived = archive_root / f"{stamp}_{run_dir.name}"
    counter = 1
    while archived.exists():
        archived = archive_root / f"{stamp}_{run_dir.name}_{counter}"
        counter += 1
    shutil.move(str(run_dir), str(archived))
    return archived


@bp.get("/leaf")
def outputs_leaf():
    fp = _outputs_root() / "cell_counts_leaf.csv"
    if not fp.exists():
        return jsonify({"ok": False, "error": "output not found"}), 404
    return send_from_directory(fp.parent, fp.name)


@bp.get("/leaf/<channel>")
def outputs_leaf_channel(channel: str):
    fp = _outputs_root() / f"cell_counts_leaf_{channel}.csv"
    if not fp.exists():
        return ("", 204, {"Content-Type": "text/csv; charset=utf-8"})
    return send_from_directory(fp.parent, fp.name)


@bp.get("/hierarchy")
def outputs_hierarchy():
    fp = _outputs_root() / "cell_counts_hierarchy.csv"
    if not fp.exists():
        return jsonify({"ok": False, "error": "hierarchy output not found"}), 404
    return send_from_directory(fp.parent, fp.name)


@bp.get("/registration-qc")
def outputs_registration_qc():
    fp = _outputs_root() / "slice_registration_qc.csv"
    if not fp.exists():
        return jsonify({"ok": False, "error": "registration QC not found"}), 404
    return send_from_directory(fp.parent, fp.name)


@bp.get("/reg-slice-list")
def outputs_reg_slice_list():
    reg_dir = _outputs_root() / "registered_slices"
    if not reg_dir.exists():
        return jsonify({"ok": True, "files": [], "count": 0})
    files = sorted(reg_dir.glob("slice_*_overlay.png"))
    return jsonify({"ok": True, "files": [f.name for f in files], "count": len(files)})


@bp.get("/reg-slice/<filename>")
def outputs_reg_slice_file(filename: str):
    reg_dir = _outputs_root() / "registered_slices"
    safe = Path(filename).name
    fp = reg_dir / safe
    if not fp.exists() or not safe.endswith(".png"):
        return jsonify({"ok": False, "error": "file not found"}), 404
    return send_from_directory(str(reg_dir), safe)


@bp.get("/file-list")
def outputs_file_list():
    outputs_root = _outputs_root()
    if not outputs_root.exists():
        return jsonify({"ok": True, "files": []})
    files = []
    for f in sorted(outputs_root.iterdir()):
        if f.is_file():
            files.append({"name": f.name, "size": f.stat().st_size, "ext": f.suffix.lower()})
    return jsonify({"ok": True, "files": files, "dir": str(outputs_root)})


@bp.get("/registration-runs")
def outputs_registration_runs():
    runs: list[dict[str, object]] = []
    for run_dir in _registration_run_dirs(_outputs_root()):
        try:
            runs.append(_serialize_registration_run(run_dir))
        except Exception:
            continue
    return jsonify({"ok": True, "runs": runs, "count": len(runs)})


@bp.post("/registration-run/<run_name>/pin")
def outputs_registration_run_pin(run_name: str):
    run_dir = _safe_registration_run_dir(run_name)
    if run_dir is None:
        return jsonify({"ok": False, "error": "run not found"}), 404
    pinned = _pin_run(run_dir.name, run_dir.parent)
    return jsonify({"ok": True, "run": run_dir.name, "pinned": pinned})


@bp.post("/registration-run/<run_name>/delete-bad")
def outputs_registration_run_delete_bad(run_name: str):
    run_dir = _safe_registration_run_dir(run_name)
    if run_dir is None:
        return jsonify({"ok": False, "error": "run not found"}), 404
    archived_dir = _archive_run_dir(run_dir)
    _unpin_run(run_dir.name, archived_dir.parent.parent)
    return jsonify(
        {
            "ok": True,
            "run": run_dir.name,
            "archived_to": str(archived_dir),
        }
    )


@bp.get("/registration-run/<run_name>/<filename>")
def outputs_registration_run_file(run_name: str, filename: str):
    run_dir = _safe_registration_run_dir(run_name)
    if run_dir is None:
        return jsonify({"ok": False, "error": "run not found"}), 404

    safe = Path(filename).name
    fp = run_dir / safe
    if not fp.exists() or not fp.is_file():
        return jsonify({"ok": False, "error": "file not found"}), 404
    return send_from_directory(str(run_dir), safe)


@bp.get("/named/<filename>")
def outputs_named(filename: str):
    safe = Path(filename).name
    outputs_root = _outputs_root()
    fp = outputs_root / safe
    if not fp.exists():
        return jsonify({"ok": False, "error": "file not found"}), 404
    return send_from_directory(str(outputs_root), safe)


@bp.get("/qc-list")
def outputs_qc_list():
    qc_dir = _outputs_root() / "qc_overlays"
    if not qc_dir.exists():
        return jsonify({"ok": True, "files": [], "count": 0})
    files = sorted(qc_dir.glob("overlay_*.png"))
    return jsonify({"ok": True, "files": [f.name for f in files], "count": len(files)})


@bp.get("/qc-file/<filename>")
def outputs_qc_file(filename: str):
    qc_dir = _outputs_root() / "qc_overlays"
    safe = Path(filename).name
    return send_from_directory(str(qc_dir), safe)
