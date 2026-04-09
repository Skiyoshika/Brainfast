"""api_alignment.py — Landmark alignment, nonlinear alignment, slice info/extract routes."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Blueprint, jsonify, request, send_file, send_from_directory
from PIL import Image, ImageOps
from tifffile import TiffFile, imread, imwrite

import project.frontend.server_context as ctx
from project.frontend.services.alignment_service import (
    apply_affine_alignment,
    apply_nonlinear_alignment,
    propose_landmarks,
    render_landmark_preview,
)

bp = Blueprint("api_alignment", __name__, url_prefix="/api")


def _normalize_path(p: str) -> str:
    """Normalize Windows paths: resolve double-backslashes, forward slashes, etc."""
    if not p:
        return p
    # os.path.normpath handles \\, /, mixed separators
    import os
    return os.path.normpath(p)


_REAL_PREVIEW_PALETTES: dict[str, tuple[str, str, str] | None] = {
    "gray": None,
    "green": ("#000000", "#0b2d10", "#9dffab"),
    "magenta": ("#000000", "#38102f", "#ffb0f5"),
    "amber": ("#000000", "#4a2500", "#ffd47a"),
    "turbo": ("#1b1237", "#00b8c8", "#ffd95a"),
}


def _normalize_intensity_plane(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        arr = arr[..., :3]
        arr = arr.astype(np.float32)
        positive = arr[arr > 0]
        if positive.size:
            lo, hi = np.percentile(positive, [1, 99])
        else:
            lo, hi = float(arr.min(initial=0.0)), float(arr.max(initial=1.0))
        scale = max(hi - lo, 1e-6)
        return np.clip((arr - lo) / scale * 255.0, 0, 255).astype(np.uint8)
    if arr.ndim >= 3:
        arr = arr[0]
    arr = arr.astype(np.float32)
    positive = arr[arr > 0]
    if positive.size:
        lo, hi = np.percentile(positive, [1, 99])
    else:
        lo, hi = float(arr.min(initial=0.0)), float(arr.max(initial=1.0))
    scale = max(hi - lo, 1e-6)
    return np.clip((arr - lo) / scale * 255.0, 0, 255).astype(np.uint8)


def _normalize_real_preview(arr: np.ndarray, palette: str = "green") -> np.ndarray:
    norm = _normalize_intensity_plane(arr)
    if norm.ndim == 3 and norm.shape[-1] == 3:
        return norm

    palette_key = palette if palette in _REAL_PREVIEW_PALETTES else "green"
    palette_spec = _REAL_PREVIEW_PALETTES[palette_key]
    if palette_spec is None:
        return np.stack([norm, norm, norm], axis=-1)

    black, mid, white = palette_spec
    return np.asarray(
        ImageOps.colorize(
            Image.fromarray(norm, mode="L"),
            black=black,
            mid=mid,
            white=white,
            midpoint=96,
        )
    )


def _normalize_label_preview(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        return _normalize_real_preview(arr, palette="gray")
    if arr.ndim >= 3:
        arr = arr[0]
    arr = arr.astype(np.int64)
    rgb = np.zeros(arr.shape + (3,), dtype=np.uint8)
    mask = arr > 0
    rgb[..., 0][mask] = ((arr[mask] * 53) % 255).astype(np.uint8)
    rgb[..., 1][mask] = ((arr[mask] * 97) % 255).astype(np.uint8)
    rgb[..., 2][mask] = ((arr[mask] * 193) % 255).astype(np.uint8)
    return rgb


def _extract_preview_frame(path: Path, z_index: int | None = None) -> np.ndarray:
    with TiffFile(path) as tif:
        series = tif.series[0]
        shape = tuple(int(v) for v in series.shape)
        if len(shape) >= 3 and len(tif.pages) > 1 and shape[0] == len(tif.pages):
            idx = 0 if z_index is None else max(0, min(int(z_index), shape[0] - 1))
            return np.asarray(tif.asarray(key=idx, series=0))

        img = np.asarray(series.asarray())
    if img.ndim >= 3:
        idx = 0 if z_index is None else max(0, min(int(z_index), img.shape[0] - 1))
        return np.asarray(img[idx])
    return img


def _read_pixel_size_um(path: Path) -> float | None:
    try:
        with TiffFile(path) as tif:
            page = tif.pages[0]
            xres_tag = page.tags.get("XResolution")
            unit_tag = page.tags.get("ResolutionUnit")
            if xres_tag is None or unit_tag is None:
                return None
            num, den = xres_tag.value
            if not den:
                return None
            pixels_per_unit = float(num) / float(den)
            if pixels_per_unit <= 0:
                return None
            unit = int(unit_tag.value)
            if unit == 2:  # inch
                return 25400.0 / pixels_per_unit
            if unit == 3:  # centimeter
                return 10000.0 / pixels_per_unit
    except Exception:
        return None
    return None


def _store_job_config(job_id: str, payload: dict) -> None:
    """Persist atlas_version and registration_mode in the job config file."""
    import json

    atlas_version = payload.get("atlasVersion", "ccfv3")
    registration_mode = payload.get("registrationMode", "cross_modal")
    cfg_path = ctx._job_file(job_id, "job_config.json")
    existing: dict = {}
    if cfg_path.exists():
        try:
            existing = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    existing["atlas_version"] = atlas_version
    existing["registration_mode"] = registration_mode
    cfg_path.write_text(json.dumps(existing, indent=2), encoding="utf-8")


@bp.post("/align/nonlinear")
def align_nonlinear():
    payload = request.get_json(force=True)
    job_id = ctx._payload_job_id(payload)
    _store_job_config(job_id, payload)
    real_path = Path(payload.get("realPath", ""))
    atlas_label_path = Path(payload.get("atlasLabelPath", ""))
    pairs_csv = ctx._job_file(job_id, "landmark_pairs.csv")
    if not real_path.exists() or not atlas_label_path.exists() or not pairs_csv.exists():
        return jsonify({"ok": False, "error": "missing real/atlas/pairs file"}), 400

    hemisphere = str(payload.get("hemisphere", "auto"))
    out_label = ctx._job_file(job_id, "aligned_label_nonlinear.tif")
    compare_png = ctx._job_file(job_id, "overlay_compare_nonlinear.png")
    try:
        result = apply_nonlinear_alignment(
            real_path, atlas_label_path, pairs_csv, out_label, compare_png,
            hemisphere=hemisphere,
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    return jsonify({"ok": True, "jobId": job_id, **result})


@bp.get("/outputs/overlay-compare-nonlinear")
def outputs_overlay_compare_nonlinear():
    fp = ctx._job_file(ctx._query_job_id(), "overlay_compare_nonlinear.png")
    if not fp.exists():
        return jsonify({"ok": False, "error": "nonlinear overlay compare not found"}), 404
    return send_from_directory(fp.parent, fp.name)


@bp.get("/outputs/auto-label-slice")
def outputs_auto_label_slice():
    fp = ctx._job_file(ctx._query_job_id(), "auto_label_slice.tif")
    if not fp.exists():
        return jsonify({"ok": False, "error": "auto label slice not found"}), 404
    return send_from_directory(fp.parent, fp.name)


@bp.get("/outputs/landmark-preview")
def outputs_landmark_preview():
    fp = ctx._job_file(ctx._query_job_id(), "landmark_preview.png")
    if not fp.exists():
        return jsonify({"ok": False, "error": "landmark preview not found"}), 404
    return send_from_directory(fp.parent, fp.name)


@bp.post("/align/landmark-preview")
def align_landmark_preview():
    payload = request.get_json(force=True)
    job_id = ctx._payload_job_id(payload)
    real_path = Path(payload.get("realPath", ""))
    atlas_path = Path(payload.get("atlasPath", ""))
    pairs_csv = ctx._job_file(job_id, "landmark_pairs.csv")
    if not real_path.exists() or not atlas_path.exists() or not pairs_csv.exists():
        return jsonify({"ok": False, "error": "missing real/atlas or pairs file"}), 400

    fp = ctx._job_file(job_id, "landmark_preview.png")
    try:
        n_points = render_landmark_preview(real_path, atlas_path, pairs_csv, fp)
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    return jsonify({"ok": True, "preview": str(fp), "points": n_points, "jobId": job_id})


@bp.post("/align/landmarks")
def align_landmarks():
    payload = request.get_json(force=True)
    job_id = ctx._payload_job_id(payload)
    _store_job_config(job_id, payload)
    real_path = Path(payload.get("realPath", ""))
    atlas_path = Path(payload.get("atlasPath", ""))
    if not real_path.exists() or not atlas_path.exists():
        return jsonify({"ok": False, "error": "real or atlas path not found"}), 400

    out_csv = ctx._job_file(job_id, "landmark_pairs.csv")
    try:
        res = propose_landmarks(
            real_path,
            atlas_path,
            out_csv,
            max_points=int(payload.get("maxPoints", 30)),
            min_distance=int(payload.get("minDistance", 12)),
            ransac_residual=float(payload.get("ransacResidual", 8.0)),
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    return jsonify({"ok": True, "jobId": job_id, **res})


@bp.post("/align/apply")
def align_apply():
    payload = request.get_json(force=True)
    job_id = ctx._payload_job_id(payload)
    _store_job_config(job_id, payload)
    real_path = Path(payload.get("realPath", ""))
    atlas_label_path = Path(payload.get("atlasLabelPath", ""))
    pairs_csv = ctx._job_file(job_id, "landmark_pairs.csv")
    if not real_path.exists() or not atlas_label_path.exists() or not pairs_csv.exists():
        return jsonify({"ok": False, "error": "missing real/atlas/pairs file"}), 400

    hemisphere = str(payload.get("hemisphere", "auto"))
    out_label = ctx._job_file(job_id, "aligned_label_ai.tif")
    compare_png = ctx._job_file(job_id, "overlay_compare.png")
    try:
        result = apply_affine_alignment(
            real_path, atlas_label_path, pairs_csv, out_label, compare_png,
            hemisphere=hemisphere,
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400

    return jsonify({"ok": True, "jobId": job_id, **result})


@bp.post("/align/add-manual-landmarks")
def add_manual_landmarks():
    payload = request.get_json(force=True)
    job_id = ctx._payload_job_id(payload)
    pairs = payload.get("pairs", [])
    if not pairs:
        return jsonify({"ok": False, "error": "no pairs provided"}), 400
    pairs_csv = ctx._job_file(job_id, "landmark_pairs.csv")
    new_rows = pd.DataFrame(pairs)
    if pairs_csv.exists():
        try:
            existing = pd.read_csv(pairs_csv)
            combined = pd.concat([existing, new_rows], ignore_index=True)
        except Exception:
            combined = new_rows
    else:
        combined = new_rows
    combined.to_csv(pairs_csv, index=False)
    return jsonify({"ok": True, "total_pairs": int(len(combined)), "jobId": job_id})


@bp.get("/align/manual-image")
def align_manual_image():
    path = request.args.get("path", "")
    kind = str(request.args.get("kind", "real")).strip().lower()
    palette = str(request.args.get("palette", "green")).strip().lower() or "green"
    raw_z = request.args.get("z")
    z_index = None if raw_z in (None, "", "null") else int(raw_z)
    src = Path(path)
    if not path or not src.exists():
        return jsonify({"ok": False, "error": "file not found"}), 400
    try:
        frame = _extract_preview_frame(src, z_index=z_index)
        rgb = (
            _normalize_label_preview(frame)
            if kind == "atlas"
            else _normalize_real_preview(frame, palette=palette)
        )
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="PNG")
        buf.seek(0)
        download_name = f"{src.stem}_z{z_index or 0}_{kind}_{palette}.png"
        return send_file(buf, mimetype="image/png", download_name=download_name, max_age=3600)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc)}), 400


@bp.get("/slice/info")
def slice_info():
    path = _normalize_path(request.args.get("path", ""))
    if not path or not Path(path).exists():
        return jsonify({"ok": False, "error": "file not found"}), 400
    try:
        with TiffFile(path) as tif:
            shape = list(tif.series[0].shape)
        ndim = len(shape)
        pixel_size_um = _read_pixel_size_um(Path(path))
        return jsonify(
            {
                "ok": True,
                "shape": shape,
                "ndim": ndim,
                "is3d": ndim >= 3,
                "z_count": shape[0] if ndim >= 3 else 1,
                "pixel_size_um": pixel_size_um,
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@bp.get("/slice/thumbnail")
def slice_thumbnail():
    path = _normalize_path(request.args.get("path", ""))
    z = int(request.args.get("z", 0))
    size = int(request.args.get("size", 360))
    if not path or not Path(path).exists():
        return jsonify({"ok": False, "error": "file not found"}), 400
    try:
        with TiffFile(path) as tif:
            shape = list(tif.series[0].shape)
            ndim = len(shape)
            if ndim >= 3:
                z = max(0, min(z, shape[0] - 1))
                page_idx = z
            else:
                page_idx = 0
            page = tif.pages[page_idx]
            arr = page.asarray()
        # Percentile-based contrast normalization for fluorescence data
        arr = _normalize_intensity_plane(arr)
        img = Image.fromarray(arr)
        img.thumbnail((size, size), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=80)
        buf.seek(0)
        return send_file(buf, mimetype="image/jpeg")
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400


@bp.post("/slice/extract-z")
def slice_extract_z():
    payload = request.get_json(force=True)
    job_id = ctx._payload_job_id(payload)
    src = Path(_normalize_path(payload.get("path", "")))
    z = int(payload.get("z", 0))
    if not src.exists():
        return jsonify({"ok": False, "error": "source file not found"}), 400
    try:
        # Memory-safe: read only the requested page instead of loading entire stack
        with TiffFile(str(src)) as tif:
            shape = list(tif.series[0].shape)
            ndim = len(shape)
            if ndim >= 3:
                z = max(0, min(z, shape[0] - 1))
                slc = tif.pages[z].asarray()
            else:
                slc = tif.pages[0].asarray()
        out_path = ctx._job_file(job_id, f"extracted_z{z:04d}.tif")
        imwrite(str(out_path), slc)
        return jsonify(
            {
                "ok": True,
                "path": str(out_path),
                "z": z,
                "shape": list(slc.shape),
                "dtype": str(slc.dtype),
            }
        )
    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 400
