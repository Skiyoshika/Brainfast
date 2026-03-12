from __future__ import annotations

import os
import sys
import threading
import subprocess
import shutil
import json
from pathlib import Path
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
    ROOT = Path(sys._MEIPASS)
    PROJECT_ROOT = Path(sys.executable).resolve().parent
    if str(PROJECT_ROOT).endswith("frontend"):
        PROJECT_ROOT = PROJECT_ROOT.parent
else:
    ROOT = Path(__file__).resolve().parent
    PROJECT_ROOT = ROOT.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.overlay_render import render_overlay
from scripts.ai_landmark import propose_landmarks, score_alignment, score_alignment_edges
from scripts.align_ai import apply_landmark_affine
from scripts.align_nonlinear import apply_landmark_nonlinear
from scripts.compare_render import render_before_after
from scripts.atlas_autopick import autopick_best_z
from tifffile import imread, imwrite
import numpy as np

OUTPUT_DIR = PROJECT_ROOT / "outputs"

app = Flask(__name__, static_folder=str(ROOT), static_url_path="")

run_state = {
    "running": False,
    "done": False,
    "error": None,
    "logs": [],
    "channels": [],
    "proc": None,
    "current_channel": None,
    "history": [],
}


def _append_log(line: str):
    run_state["logs"].append(line.rstrip())
    if len(run_state["logs"]) > 500:
        run_state["logs"] = run_state["logs"][-500:]


def _runner(config_path: str, input_dir: str, channels: list[str]):
    run_state.update({"running": True, "done": False, "error": None, "channels": channels, "logs": []})

    for ch in channels:
        run_state["current_channel"] = ch
        _append_log(f"[run] channel={ch}")
        cmd = [
            "python",
            str(PROJECT_ROOT / "scripts" / "main.py"),
            "--config",
            config_path,
            "--run-real-input",
            input_dir,
        ]
        env = os.environ.copy()
        env["BRAINCOUNT_ACTIVE_CHANNEL"] = ch

        p = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        run_state["proc"] = p
        assert p.stdout is not None
        for line in p.stdout:
            _append_log(line)
        code = p.wait()
        _append_log(f"[exit] channel={ch} code={code}")
        if code == 0:
            leaf = OUTPUT_DIR / "cell_counts_leaf.csv"
            if leaf.exists():
                shutil.copy2(leaf, OUTPUT_DIR / f"cell_counts_leaf_{ch}.csv")
        else:
            run_state["error"] = f"channel {ch} failed with code {code}"
            break

    run_state["running"] = False
    run_state["done"] = run_state["error"] is None
    run_state["proc"] = None
    run_state["current_channel"] = None
    run_state["history"].append({
        "channels": channels,
        "ok": run_state["error"] is None,
        "error": run_state["error"],
        "logCount": len(run_state["logs"]),
    })
    if len(run_state["history"]) > 20:
        run_state["history"] = run_state["history"][-20:]


@app.get("/")
def index():
    return send_from_directory(ROOT, "index.html")


@app.get('/api/info')
def info():
    return jsonify({
        "app": "IdleBrainUI",
        "version": "0.3.0-desktop",
        "frontend": str(ROOT),
        "project": str(PROJECT_ROOT),
        "outputs": str(OUTPUT_DIR),
    })


@app.get("/api/validate")
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
        issues.append("Structure mapping CSV missing or not found")

    return jsonify({"ok": len(issues) == 0, "issues": issues})


@app.post("/api/run")
def run_pipeline():
    if run_state["running"]:
        return jsonify({"ok": False, "error": "pipeline already running"}), 409

    payload = request.get_json(force=True)
    config = payload.get("configPath") or str(PROJECT_ROOT / "configs" / "run_config.template.json")
    input_dir = payload.get("inputDir", "")
    channels = payload.get("channels", ["red"])
    if isinstance(channels, str):
        channels = [channels]

    t = threading.Thread(target=_runner, args=(config, input_dir, channels), daemon=True)
    t.start()
    return jsonify({"ok": True, "started": True})


@app.get("/api/status")
def status():
    return jsonify(
        {
            "running": run_state["running"],
            "done": run_state["done"],
            "error": run_state["error"],
            "channels": run_state["channels"],
            "currentChannel": run_state["current_channel"],
            "logCount": len(run_state["logs"]),
        }
    )


@app.post("/api/cancel")
def cancel():
    p = run_state.get("proc")
    if p and run_state.get("running"):
        p.terminate()
        run_state["error"] = "cancelled by user"
        run_state["running"] = False
        run_state["done"] = False
        _append_log("[cancel] user requested stop")
        return jsonify({"ok": True, "cancelled": True})
    return jsonify({"ok": False, "cancelled": False, "error": "no running process"}), 409


@app.get("/api/logs")
def logs():
    return jsonify({"logs": run_state["logs"]})


@app.get('/api/history')
def history():
    return jsonify({"history": run_state["history"]})


@app.post('/api/atlas/autopick-z')
def atlas_autopick_z():
    payload = request.get_json(force=True)
    real_path = Path(payload.get('realPath', ''))
    annotation_path = Path(payload.get('annotationPath', ''))
    z_step = int(payload.get('zStep', 1))
    pixel_size_um = float(payload.get('pixelSizeUm', 0.65))
    slicing_plane = str(payload.get('slicingPlane', 'coronal'))
    roi_mode = str(payload.get('roiMode', 'auto'))
    if not real_path.exists() or not annotation_path.exists():
        return jsonify({"ok": False, "error": "missing real or annotation path"}), 400

    out_label = OUTPUT_DIR / 'auto_label_slice.tif'
    try:
        res = autopick_best_z(
            real_path,
            annotation_path,
            out_label,
            z_step=z_step,
            pixel_size_um=pixel_size_um,
            slicing_plane=slicing_plane,
            roi_mode=roi_mode,
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    return jsonify({"ok": True, **res})


@app.post('/api/overlay/preview')
def overlay_preview():
    payload = request.get_json(force=True)
    real_path = Path(payload.get('realPath', ''))
    label_path = Path(payload.get('labelPath', ''))
    alpha = float(payload.get('alpha', 0.45))
    mode = payload.get('mode', 'fill')
    structure_csv = Path(payload.get('structureCsv', '')) if payload.get('structureCsv') else None
    min_mean = float(payload.get('minMeanThreshold', 8.0))
    pixel_size_um = float(payload.get('pixelSizeUm', 0.65))
    rotate_deg = float(payload.get('rotateAtlas', 0.0))
    flip_mode = payload.get('flipAtlas', 'none')
    fit_mode = payload.get('fitMode', 'contain')
    major_top_k = int(payload.get('majorTopK', 12))

    if not real_path.exists() or not label_path.exists():
        return jsonify({"ok": False, "error": "real or label path not found"}), 400

    out = OUTPUT_DIR / 'overlay_preview.png'
    try:
        _, diagnostic = render_overlay(
            real_path,
            label_path,
            out,
            alpha=alpha,
            mode=mode,
            structure_csv=structure_csv,
            min_mean_threshold=min_mean,
            pixel_size_um=pixel_size_um,
            rotate_deg=rotate_deg,
            flip_mode=flip_mode,
            return_meta=True,
            major_top_k=major_top_k,
            fit_mode=fit_mode,
        )
    except Exception as e:
        fail_dir = OUTPUT_DIR / 'fail_cases'
        fail_dir.mkdir(parents=True, exist_ok=True)
        fail_json = fail_dir / 'overlay_fail_last.json'
        fail_json.write_text(json.dumps({
            'realPath': str(real_path),
            'labelPath': str(label_path),
            'alpha': alpha,
            'mode': mode,
            'structureCsv': str(structure_csv) if structure_csv else '',
            'minMeanThreshold': min_mean,
            'pixelSizeUm': pixel_size_um,
            'rotateAtlas': rotate_deg,
            'flipAtlas': flip_mode,
            'fitMode': fit_mode,
            'error': str(e)
        }, indent=2), encoding='utf-8')
        return jsonify({"ok": False, "error": str(e), "failCase": str(fail_json)}), 400
    return jsonify({"ok": True, "preview": str(out), "minMeanThreshold": min_mean, "diagnostic": diagnostic})


@app.get("/api/outputs/leaf")
def outputs_leaf():
    fp = OUTPUT_DIR / "cell_counts_leaf.csv"
    if not fp.exists():
        return jsonify({"ok": False, "error": "output not found"}), 404
    return send_from_directory(fp.parent, fp.name)


@app.get('/api/outputs/leaf/<channel>')
def outputs_leaf_channel(channel: str):
    fp = OUTPUT_DIR / f"cell_counts_leaf_{channel}.csv"
    if not fp.exists():
        return jsonify({"ok": False, "error": f"channel output not found: {channel}"}), 404
    return send_from_directory(fp.parent, fp.name)


@app.get('/api/outputs/overlay-preview')
def outputs_overlay_preview():
    fp = OUTPUT_DIR / 'overlay_preview.png'
    if not fp.exists():
        return jsonify({"ok": False, "error": "overlay preview not found"}), 404
    return send_from_directory(fp.parent, fp.name)


@app.get('/api/outputs/overlay-compare')
def outputs_overlay_compare():
    fp = OUTPUT_DIR / 'overlay_compare.png'
    if not fp.exists():
        return jsonify({"ok": False, "error": "overlay compare not found"}), 404
    return send_from_directory(fp.parent, fp.name)


@app.post('/api/align/nonlinear')
def align_nonlinear():
    payload = request.get_json(force=True)
    real_path = Path(payload.get('realPath', ''))
    atlas_label_path = Path(payload.get('atlasLabelPath', ''))
    pairs_csv = OUTPUT_DIR / 'landmark_pairs.csv'
    if not real_path.exists() or not atlas_label_path.exists() or not pairs_csv.exists():
        return jsonify({"ok": False, "error": "missing real/atlas/pairs file"}), 400

    out_label = OUTPUT_DIR / 'aligned_label_nonlinear.tif'
    fail_log = OUTPUT_DIR / 'fail_cases' / 'align_nonlinear_last.json'
    try:
        meta = apply_landmark_nonlinear(real_path, atlas_label_path, pairs_csv, out_label)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e), "failLog": str(fail_log)}), 400

    real = imread(str(real_path)); atlas_before = imread(str(atlas_label_path)); atlas_after = imread(str(out_label))
    if real.ndim == 3: real = real[..., 0]
    if atlas_before.ndim == 3: atlas_before = atlas_before[..., 0]
    if atlas_after.ndim == 3: atlas_after = atlas_after[..., 0]

    before = score_alignment(real, atlas_before)
    after = score_alignment(real, atlas_after)
    before_edge = score_alignment_edges(real, atlas_before)
    after_edge = score_alignment_edges(real, atlas_after)

    compare_png = OUTPUT_DIR / 'overlay_compare_nonlinear.png'
    render_before_after(real_path, atlas_label_path, out_label, compare_png, alpha=0.45, before_score=before_edge, after_score=after_edge)
    return jsonify({
        "ok": True,
        "beforeScore": before,
        "afterScore": after,
        "beforeEdgeScore": before_edge,
        "afterEdgeScore": after_edge,
        "scoreWarning": after_edge < before_edge,
        "compareImage": str(compare_png),
        **meta,
    })


@app.get('/api/outputs/overlay-compare-nonlinear')
def outputs_overlay_compare_nonlinear():
    fp = OUTPUT_DIR / 'overlay_compare_nonlinear.png'
    if not fp.exists():
        return jsonify({"ok": False, "error": "nonlinear overlay compare not found"}), 404
    return send_from_directory(fp.parent, fp.name)


@app.get('/api/outputs/auto-label-slice')
def outputs_auto_label_slice():
    fp = OUTPUT_DIR / 'auto_label_slice.tif'
    if not fp.exists():
        return jsonify({"ok": False, "error": "auto label slice not found"}), 404
    return send_from_directory(fp.parent, fp.name)


@app.get('/api/outputs/landmark-preview')
def outputs_landmark_preview():
    fp = OUTPUT_DIR / 'landmark_preview.png'
    if not fp.exists():
        return jsonify({"ok": False, "error": "landmark preview not found"}), 404
    return send_from_directory(fp.parent, fp.name)


@app.post('/api/align/landmark-preview')
def align_landmark_preview():
    payload = request.get_json(force=True)
    real_path = Path(payload.get('realPath', ''))
    atlas_path = Path(payload.get('atlasPath', ''))
    pairs_csv = OUTPUT_DIR / 'landmark_pairs.csv'
    if not real_path.exists() or not atlas_path.exists() or not pairs_csv.exists():
        return jsonify({"ok": False, "error": "missing real/atlas or pairs file"}), 400

    real = imread(str(real_path)); atlas = imread(str(atlas_path))
    if real.ndim == 3: real = real[..., 0]
    if atlas.ndim == 3: atlas = atlas[..., 0]

    h = min(real.shape[0], atlas.shape[0]); w = min(real.shape[1], atlas.shape[1])
    real = real[:h,:w]; atlas = atlas[:h,:w]

    real_rgb = np.stack([real, real, real], axis=-1).astype(np.uint8)
    atlas_rgb = np.stack([atlas, atlas, atlas], axis=-1).astype(np.uint8)

    pairs = pd.read_csv(pairs_csv)
    for _, r in pairs.iterrows():
        rx, ry = int(r['real_x']), int(r['real_y'])
        ax, ay = int(r['atlas_x']), int(r['atlas_y'])
        if 0 <= ry < h and 0 <= rx < w:
            real_rgb[max(0,ry-2):ry+3, max(0,rx-2):rx+3] = [255, 255, 0]
        if 0 <= ay < h and 0 <= ax < w:
            atlas_rgb[max(0,ay-2):ay+3, max(0,ax-2):ax+3] = [0, 255, 255]

    pad = np.zeros((h, 8, 3), dtype=np.uint8)
    canvas = np.concatenate([real_rgb, pad, atlas_rgb], axis=1)
    fp = OUTPUT_DIR / 'landmark_preview.png'
    imwrite(str(fp), canvas)
    return jsonify({"ok": True, "preview": str(fp), "points": int(len(pairs))})


@app.post('/api/align/landmarks')
def align_landmarks():
    payload = request.get_json(force=True)
    real_path = Path(payload.get('realPath', ''))
    atlas_path = Path(payload.get('atlasPath', ''))
    if not real_path.exists() or not atlas_path.exists():
        return jsonify({"ok": False, "error": "real or atlas path not found"}), 400

    max_points = int(payload.get('maxPoints', 30))
    min_distance = int(payload.get('minDistance', 12))
    ransac_residual = float(payload.get('ransacResidual', 8.0))

    out_csv = OUTPUT_DIR / 'landmark_pairs.csv'
    res = propose_landmarks(
        real_path,
        atlas_path,
        out_csv,
        max_points=max_points,
        min_distance=min_distance,
        ransac_residual=ransac_residual,
    )
    return jsonify({"ok": True, **res})


@app.post('/api/align/apply')
def align_apply():
    payload = request.get_json(force=True)
    real_path = Path(payload.get('realPath', ''))
    atlas_label_path = Path(payload.get('atlasLabelPath', ''))
    pairs_csv = OUTPUT_DIR / 'landmark_pairs.csv'
    if not real_path.exists() or not atlas_label_path.exists() or not pairs_csv.exists():
        return jsonify({"ok": False, "error": "missing real/atlas/pairs file"}), 400

    out_label = OUTPUT_DIR / 'aligned_label_ai.tif'
    meta = apply_landmark_affine(real_path, atlas_label_path, pairs_csv, out_label)

    real = imread(str(real_path))
    atlas_before = imread(str(atlas_label_path))
    atlas_after = imread(str(out_label))
    if real.ndim == 3:
        real = real[..., 0]
    if atlas_before.ndim == 3:
        atlas_before = atlas_before[..., 0]
    if atlas_after.ndim == 3:
        atlas_after = atlas_after[..., 0]

    before = score_alignment(real, atlas_before)
    after = score_alignment(real, atlas_after)
    before_edge = score_alignment_edges(real, atlas_before)
    after_edge = score_alignment_edges(real, atlas_after)
    compare_png = OUTPUT_DIR / 'overlay_compare.png'
    render_before_after(real_path, atlas_label_path, out_label, compare_png, alpha=0.45, before_score=before_edge, after_score=after_edge)

    return jsonify({
        "ok": True,
        "beforeScore": before,
        "afterScore": after,
        "beforeEdgeScore": before_edge,
        "afterEdgeScore": after_edge,
        "scoreWarning": after_edge < before_edge,
        "compareImage": str(compare_png),
        **meta,
    })


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8787, debug=False)

