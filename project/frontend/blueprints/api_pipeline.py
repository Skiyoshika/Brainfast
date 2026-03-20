"""api_pipeline.py — Pipeline run/status/cancel/logs/history routes."""

from __future__ import annotations

import json
import threading
from pathlib import Path

from flask import Blueprint, jsonify, request, send_from_directory

import project.frontend.server_context as ctx

bp = Blueprint("api_pipeline", __name__)


@bp.get("/")
def index():
    return send_from_directory(ctx.ROOT, "index.html")


@bp.get("/api/info")
def info():
    default_atlas = ctx.PROJECT_ROOT / "annotation_25.nii.gz"
    default_struct = ctx.DEFAULT_STRUCTURE_SOURCE
    return jsonify(
        {
            "app": "BrainfastUI",
            "version": "0.3.0-desktop",
            "frontend": str(ctx.ROOT),
            "project": str(ctx.PROJECT_ROOT),
            "outputs": str(ctx.OUTPUT_DIR),
            "defaults": {
                "atlasPath": str(default_atlas),
                "structPath": str(default_struct) if default_struct.exists() else "",
                "sampleLimit": int(ctx.MAX_CALIB_SAMPLES),
            },
        }
    )


@bp.get("/api/validate")
def validate():
    input_dir = request.args.get("inputDir", "")
    atlas = request.args.get("atlasPath", "")
    struct = request.args.get("structPath", "")

    issues = []
    if not input_dir or not Path(input_dir).exists():
        issues.append("Input TIFF folder missing or not found")
    if not atlas or not Path(atlas).exists():
        issues.append("Atlas annotation file missing or not found")
    if not struct or not Path(struct).exists():
        issues.append("Structure mapping file missing or not found")
    elif Path(struct).suffix.lower() not in {".csv", ".json"}:
        issues.append("Structure mapping file must be .csv or .json")

    return jsonify({"ok": len(issues) == 0, "issues": issues})


@bp.post("/api/run")
def run_pipeline():
    with ctx._run_state_lock:
        if ctx.run_state["running"]:
            return jsonify({"ok": False, "error": "pipeline already running"}), 409

    payload = request.get_json(force=True)
    config = payload.get("configPath") or str(
        ctx.PROJECT_ROOT / "configs" / "run_config.template.json"
    )
    input_dir = payload.get("inputDir", "")
    channels = payload.get("channels", ["red"])
    if isinstance(channels, str):
        channels = [channels]

    run_params = payload.get("params", {})
    with ctx._run_state_lock:
        ctx.run_state["config_path"] = config
    t = threading.Thread(
        target=ctx._runner, args=(config, input_dir, channels, run_params), daemon=True
    )
    t.start()
    return jsonify({"ok": True, "started": True})


@bp.get("/api/status")
def status():
    # Count how many slices have been registered (overlay files present)
    reg_dir = ctx.OUTPUT_DIR / "registered_slices"
    slices_done = len(list(reg_dir.glob("slice_*_overlay.png"))) if reg_dir.exists() else 0
    # Count total input slices using whichever processing directory is active.
    merged_dir = ctx.OUTPUT_DIR / "tmp_merged"
    channel_dir = ctx.OUTPUT_DIR / "tmp_channel"
    merged_total = len(list(merged_dir.glob("*.tif"))) if merged_dir.exists() else 0
    channel_total = len(list(channel_dir.glob("*.tif"))) if channel_dir.exists() else 0
    sampling_mode = ""
    config_path = ctx.run_state.get("config_path")
    if config_path:
        try:
            cfg = json.loads(Path(str(config_path)).read_text(encoding="utf-8-sig"))
            sampling_mode = str(cfg.get("input", {}).get("sampling_mode", "")).lower().strip()
        except Exception:
            sampling_mode = ""
    if sampling_mode in {"single", "native", "raw", "original"}:
        slices_total = channel_total
    elif sampling_mode in {"merge", "merged", "merge5", "merge_n"}:
        slices_total = merged_total
    else:
        slices_total = merged_total if merged_total > 0 else channel_total
    return jsonify(
        {
            "running": ctx.run_state["running"],
            "done": ctx.run_state["done"],
            "error": ctx.run_state["error"],
            "channels": ctx.run_state["channels"],
            "currentChannel": ctx.run_state["current_channel"],
            "logCount": len(ctx.run_state["logs"]),
            "slicesDone": slices_done,
            "slicesTotal": slices_total,
            "startEpoch": ctx.run_state.get("startEpoch"),
        }
    )


@bp.post("/api/cancel")
def cancel():
    with ctx._run_state_lock:
        p = ctx.run_state.get("proc")
        if p and ctx.run_state.get("running"):
            p.terminate()
            ctx.run_state["error"] = "cancelled by user"
            ctx.run_state["running"] = False
            ctx.run_state["done"] = False
            ctx.run_state["current_channel"] = None
            ctx.run_state["channels"] = []
            ctx.run_state["startEpoch"] = None
            ctx._append_log("[cancel] user requested stop")
            return jsonify({"ok": True, "cancelled": True})
    return jsonify({"ok": False, "cancelled": False, "error": "no running process"}), 409


@bp.get("/api/logs")
def logs():
    return jsonify({"logs": ctx.run_state["logs"]})


@bp.get("/api/history")
def history():
    return jsonify({"history": ctx.run_state["history"]})


@bp.get("/api/export/methods-text")
def export_methods_text():
    params_files = sorted(ctx.OUTPUT_DIR.glob("run_params_*.json"), reverse=True)
    params = {}
    if params_files:
        try:
            params = json.loads(params_files[0].read_text(encoding="utf-8"))
        except Exception:
            pass
    detection_summary = {}
    detection_summary_path = ctx.OUTPUT_DIR / "detection_summary.json"
    if detection_summary_path.exists():
        try:
            detection_summary = json.loads(detection_summary_path.read_text(encoding="utf-8"))
        except Exception:
            detection_summary = {}
    align_mode = params.get("alignMode", "affine")
    align_cn = "仿射变换 (affine)" if align_mode == "affine" else "非线性变换 (nonlinear/TPS)"
    align_en = "affine" if align_mode == "affine" else "nonlinear (thin-plate spline)"
    pixel_size = params.get("pixelSizeUm", "0.65")
    channels = params.get("channels", ["red"])
    ch_str = ", ".join(channels)
    ts = params.get("timestamp", "—")
    detector_counts = detection_summary.get("dedup_detector_counts") or detection_summary.get(
        "detector_counts", {}
    )
    detector_names = ", ".join(detector_counts.keys()) if detector_counts else "configured detector"
    sampling_mode = str(detection_summary.get("sampling_mode", "single"))
    if sampling_mode == "merged":
        sampling_cn = "合并切片"
        sampling_en = "merged slices"
    else:
        sampling_cn = "原始单切片"
        sampling_en = "native single slices"
    if any(str(name).startswith("cellpose") for name in detector_counts):
        detection_cn = f"细胞实例分割与计数采用 {detector_names}，并基于{sampling_cn}完成去重和图谱统计。"
        detection_en = (
            f"Cell counting used instance segmentation via {detector_names} on {sampling_en}, "
            f"followed by deduplication and hierarchical atlas aggregation."
        )
    else:
        detection_cn = f"细胞计数采用 {detector_names}，并基于{sampling_cn}完成去重和图谱统计。"
        detection_en = (
            f"Cell counting used {detector_names} on {sampling_en}, followed by deduplication "
            f"and hierarchical atlas aggregation."
        )
    text_cn = (
        f"【方法段落参考（中文）】\n"
        f"脑图谱配准使用 Brainfast v0.3 完成（运行时间：{ts}）。"
        f"显微图像分辨率为 {pixel_size} μm/像素。"
        f"图谱配准参照 Allen 小鼠脑图谱（CCFv3，annotation_25.nii.gz，体素间距 25 μm），"
        f"采用{align_cn}方法对切片进行空间配准。配准质量通过边缘 SSIM（结构相似性指标）评估。"
        f"{detection_cn}荧光通道：{ch_str}。"
    )
    text_en = (
        f"\n【Methods paragraph reference (English)】\n"
        f"Brain atlas registration was performed using Brainfast v0.3 (run: {ts}). "
        f"Microscopy images were acquired at {pixel_size} μm/pixel. "
        f"Section registration was carried out against the Allen Mouse Brain Atlas "
        f"(CCFv3, annotation_25.nii.gz, 25 μm voxel spacing) using {align_en} transformation. "
        f"Alignment quality was evaluated by edge-SSIM. "
        f"{detection_en} Channels: {ch_str}."
    )
    return jsonify({"ok": True, "text": text_cn + text_en, "params": params})
