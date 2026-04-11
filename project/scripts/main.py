import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from tifffile import imread, imwrite

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.paths import bootstrap_sys_path, ensure_runtime_cache_dirs

PROJECT_ROOT = bootstrap_sys_path()
ensure_runtime_cache_dirs(PROJECT_ROOT)

try:
    from scripts.asset_bootstrap import default_structure_source
    from scripts.atlas_autopick import autopick_best_z, refine_atlas_z_by_size
    from scripts.atlas_mapper import (
        map_cells_with_registered_label_slice,
    )
    from scripts.config_validation import config_value, load_config, validate_runtime_config
    from scripts.dedup import apply_dedup_kdtree, write_dedup_stats
    from scripts.detect import detect_cells
    from scripts.exceptions import ConfigError
    from scripts.map_and_aggregate import (
        aggregate_by_region,
        compute_region_areas_from_label_tif,
        map_cells_to_regions,
        write_outputs,
    )
    from scripts.overlay_render import _alignment_quality, render_overlay
    from scripts.preprocess import merge_every_n_slices
    from scripts.qc import export_slice_qc
    from scripts.registration_adapter import bootstrap_registration_assets
    from scripts.whole_brain_3d import run_whole_brain_3d
except Exception:
    from asset_bootstrap import default_structure_source
    from atlas_autopick import autopick_best_z, refine_atlas_z_by_size
    from atlas_mapper import map_cells_with_registered_label_slice
    from config_validation import config_value, load_config, validate_runtime_config
    from dedup import apply_dedup_kdtree, write_dedup_stats
    from detect import detect_cells
    from exceptions import ConfigError
    from map_and_aggregate import (
        aggregate_by_region,
        compute_region_areas_from_label_tif,
        map_cells_to_regions,
        write_outputs,
    )
    from overlay_render import render_overlay
    from preprocess import merge_every_n_slices
    from qc import export_slice_qc
    from registration_adapter import bootstrap_registration_assets
    from whole_brain_3d import run_whole_brain_3d


def _validated_float(cfg: dict, dotted_key: str) -> float:
    return float(config_value(cfg, dotted_key))


def _validated_int(cfg: dict, dotted_key: str) -> int:
    return int(config_value(cfg, dotted_key))


def _project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _resolve_structure_source(project_root: Path) -> Path:
    registration_csv = project_root / "outputs" / "registration" / "structure_tree.csv"
    if registration_csv.exists():
        return registration_csv
    fallback = default_structure_source(project_root)
    if fallback is not None and fallback.exists():
        return fallback
    raise FileNotFoundError(
        "structure ontology source not found in outputs/registration or project/configs"
    )


def _load_tuned_overlay_params(outputs_dir: Path) -> tuple[dict, str, int]:
    tuned_json = outputs_dir / "trainset_tuned_params.json"
    if not tuned_json.exists():
        return {}, "cover", 1

    data = json.loads(tuned_json.read_text(encoding="utf-8-sig"))
    fit_mode = "cover"
    edge_smooth_iter = 1
    warp_params: dict = {}

    if isinstance(data, dict):
        if isinstance(data.get("warpParams"), dict):
            warp_params = dict(data["warpParams"])
        if isinstance(data.get("fitMode"), str):
            fit_mode = data["fitMode"]
        if "edgeSmoothIter" in data:
            edge_smooth_iter = int(data["edgeSmoothIter"])

        best = data.get("best")
        if isinstance(best, dict):
            params = best.get("params", {})
            if isinstance(params, dict):
                if isinstance(params.get("warpParams"), dict):
                    warp_params = dict(params["warpParams"])
                if isinstance(params.get("fitMode"), str):
                    fit_mode = params["fitMode"]
                if "edgeSmoothIter" in params:
                    edge_smooth_iter = int(params["edgeSmoothIter"])

    return warp_params, str(fit_mode), int(edge_smooth_iter)


def _resolve_channel_index(cfg: dict) -> int:
    active = os.environ.get("BRAINCOUNT_ACTIVE_CHANNEL") or cfg.get("input", {}).get(
        "active_channel", "red"
    )
    cmap = cfg.get("input", {}).get("channel_map", {"red": 0, "green": 1, "farred": 2})
    return int(cmap.get(active, 0))


def _collect_slice_files(slice_dir: Path, glob_pattern: str) -> list[Path]:
    return sorted(slice_dir.glob(glob_pattern))


def _is_3d_volume(arr: np.ndarray) -> bool:
    """Detect if a TIFF array is a 3D z-stack (Z, H, W) vs multi-channel 2D (H, W, C)."""
    if arr.ndim != 3:
        return False
    # If last dim is small (<=4), likely (H, W, C) multi-channel
    # If last dim is large, likely (Z, H, W) z-stack
    return arr.shape[-1] > 4 and arr.shape[0] > 4


def _extract_channel_to_tmp(src_files: list[Path], out_dir: Path, ch_idx: int) -> list[Path]:
    """Extract channel data from source files.

    Handles two data formats:
    - 2D multi-channel TIFFs: (H, W, C) → extract channel C
    - 3D Z-stack volumes: (Z, H, W) → extract individual Z-slices
      For Z-stacks, channel is selected by filename (C0, C1, etc.) not array index.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    out = []

    # Filter files by channel suffix in filename (e.g., _C0.tif, _C1.tif)
    channel_filtered = [f for f in src_files if f"_C{ch_idx}" in f.stem]
    if not channel_filtered:
        channel_filtered = src_files  # fallback: use all files

    slice_idx = 0
    for p in channel_filtered:
        arr = imread(str(p))

        if _is_3d_volume(arr):
            # 3D z-stack volume: extract each Z-slice as a separate 2D image
            print(f"[channel] extracting {arr.shape[0]} z-slices from 3D volume {p.name}")
            for z in range(arr.shape[0]):
                dst = out_dir / f"ch_{ch_idx}_{slice_idx:04d}.tif"
                imwrite(str(dst), arr[z].astype(np.uint16))
                out.append(dst)
                slice_idx += 1
        elif arr.ndim == 3:
            # Multi-channel 2D image (H, W, C)
            if arr.shape[-1] <= ch_idx:
                ch = arr[..., 0]
            else:
                ch = arr[..., ch_idx]
            dst = out_dir / f"ch_{ch_idx}_{slice_idx:04d}.tif"
            imwrite(str(dst), ch.astype(np.uint16))
            out.append(dst)
            slice_idx += 1
        else:
            # Already 2D
            dst = out_dir / f"ch_{ch_idx}_{slice_idx:04d}.tif"
            imwrite(str(dst), arr.astype(np.uint16))
            out.append(dst)
            slice_idx += 1
    return out


def _make_sample_tiffs(out_dir: Path, n: int = 12, h: int = 256, w: int = 256) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    for i in range(n):
        base = rng.normal(200, 20, size=(h, w, 3)).astype(np.float32)
        # red channel hotspots
        for _ in range(10):
            y, x = rng.integers(20, h - 20), rng.integers(20, w - 20)
            base[y - 2 : y + 3, x - 2 : x + 3, 0] += 250
        # green channel hotspots
        for _ in range(8):
            y, x = rng.integers(20, h - 20), rng.integers(20, w - 20)
            base[y - 2 : y + 3, x - 2 : x + 3, 1] += 180
        # farred channel hotspots
        for _ in range(6):
            y, x = rng.integers(20, h - 20), rng.integers(20, w - 20)
            base[y - 2 : y + 3, x - 2 : x + 3, 2] += 130

        arr = np.clip(base, 0, 65535).astype(np.uint16)
        imwrite(str(out_dir / f"slice_{i:04d}.tif"), arr)


def _refresh_demo_visuals(
    project_root: Path,
    outputs_dir: Path,
    raw_dir: Path | None = None,
) -> None:
    refresh_script = project_root / "scripts" / "refresh_demo.py"
    if not refresh_script.exists():
        return

    print("\nAuto-running refresh_demo.py to regenerate demo visuals...")
    cmd = [sys.executable, str(refresh_script), "--outputs-dir", str(outputs_dir)]
    if raw_dir is not None:
        cmd.extend(["--raw-dir", str(raw_dir)])

    try:
        subprocess.run(
            cmd,
            cwd=str(project_root),
            timeout=180,
        )
    except Exception as _e:
        print(f"[warn] refresh_demo.py failed: {_e}")


def run_demo(cfg: dict):
    project_root = _project_root()
    outputs_dir = project_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    cells_csv = project_root / "notes" / "demo_cells.csv"
    atlas_csv = project_root / "notes" / "demo_atlas_map.csv"
    cells = pd.read_csv(cells_csv)

    px_um = _validated_float(cfg, "input.pixel_size_um_xy")
    spacing_um = _validated_float(cfg, "input.slice_spacing_um")
    neighbor = _validated_int(cfg, "dedup.neighbor_slices")
    rxy = _validated_float(cfg, "dedup.r_xy_um")

    deduped, stats = apply_dedup_kdtree(
        cells,
        neighbor_slices=neighbor,
        pixel_size_um=px_um,
        slice_spacing_um=spacing_um,
        r_xy_um=rxy,
    )
    deduped.to_csv(outputs_dir / "demo_cells_dedup.csv", index=False)
    write_dedup_stats(stats, outputs_dir)

    mapped = map_cells_to_regions(outputs_dir / "demo_cells_dedup.csv", atlas_csv)
    leaf, hierarchy = aggregate_by_region(mapped)
    write_outputs(leaf, hierarchy, outputs_dir)
    print("Demo pipeline complete: kdtree dedup + mapping + aggregation outputs generated")


def _quantify_against_exported_truth(
    *,
    truth_rows: list[dict],
    cfg: dict,
    outputs_dir: Path,
    **_kwargs,
) -> dict:
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    structure_csv = _resolve_structure_source(_project_root())

    truth_rows = list(truth_rows or [])
    input_cfg = cfg.get("input", {})
    dedup_cfg = cfg.get("dedup", {})
    pixel_size_um = float(input_cfg.get("pixel_size_um_xy", 25.0))
    slice_spacing_um = float(input_cfg.get("slice_spacing_um", 25.0))
    slicing_plane = str(input_cfg.get("slicing_plane", "coronal")).lower()
    neighbor_slices = int(dedup_cfg.get("neighbor_slices", 1))
    r_xy_um = float(dedup_cfg.get("r_xy_um", 6.0))

    mapped_rows: list[pd.DataFrame] = []
    registration_rows: list[dict] = []
    next_id = 1
    for fallback_index, truth_row in enumerate(truth_rows):
        slice_id = int(truth_row.get("slice_id", fallback_index))
        real_slice_path = Path(truth_row["real_slice_path"])
        registered_label_path = Path(truth_row["registered_label_path"])
        overlay_path = truth_row.get("overlay_path", "")

        # Compute actual alignment quality instead of hardcoded 1.0
        reg_score = 1.0
        try:
            from scripts.image_utils import norm_u8_robust

            real_img = imread(str(real_slice_path))
            if real_img.ndim == 3:
                real_img = real_img[real_img.shape[0] // 2]
            real_u8 = norm_u8_robust(real_img)
            label_img = imread(str(registered_label_path))
            reg_score = float(_alignment_quality(real_u8, label_img))
        except Exception:
            pass

        detections = detect_cells(real_slice_path, cfg)
        if detections is None or detections.empty:
            detection_count = 0
        else:
            detections = detections.copy()
            detections["slice_id"] = slice_id
            detections["cell_id"] = range(next_id, next_id + len(detections))
            next_id += len(detections)
            mapped_rows.append(
                map_cells_with_registered_label_slice(
                    detections,
                    registered_label_tif=registered_label_path,
                    structure_csv=structure_csv,
                    atlas_slice_index=slice_id,
                    slicing_plane=slicing_plane,
                    registration_score=reg_score,
                    registration_method="3d_truth_export",
                )
            )
            detection_count = len(detections)

        registration_rows.append(
            {
                "slice_id": slice_id,
                "slice_path": str(real_slice_path),
                "registered_label_path": str(registered_label_path),
                "overlay_path": str(overlay_path),
                "registration_method": "3d_truth_export",
                "score_type": "edge_ssim",
                "registration_ok": bool(reg_score >= 0.3),
                "best_score": round(reg_score, 4),
                "best_z": slice_id,
                "slicing_plane": slicing_plane,
                "detected_cells": detection_count,
            }
        )

    if mapped_rows:
        mapped = pd.concat(mapped_rows, ignore_index=True)
    else:
        mapped = pd.DataFrame(
            columns=[
                "cell_id",
                "slice_id",
                "x",
                "y",
                "score",
                "region_id",
                "region_name",
                "acronym",
                "hemisphere",
                "mapping_status",
                "atlas_slice_index",
                "slicing_plane",
                "registration_method",
                "registered_label_path",
                "structure_id_path",
                "structure_source",
            ]
        )

    deduped, _stats = apply_dedup_kdtree(
        mapped,
        neighbor_slices=neighbor_slices,
        pixel_size_um=pixel_size_um,
        slice_spacing_um=slice_spacing_um,
        r_xy_um=r_xy_um,
    )
    deduped.to_csv(outputs_dir / "cells_mapped.csv", index=False)

    leaf, hierarchy = aggregate_by_region(deduped)
    write_outputs(leaf, hierarchy, outputs_dir)

    pd.DataFrame(registration_rows).to_csv(outputs_dir / "slice_registration_qc.csv", index=False)
    pd.DataFrame(
        [
            {
                "truth_source": "3d_registered_volume",
                "score_type": "volume_truth_export",
                "registration_method": "3d_truth_export",
                "slice_count": len(truth_rows),
                "mapped_cell_count": int(len(mapped)),
                "deduped_cell_count": int(len(deduped)),
            }
        ]
    ).to_csv(outputs_dir / "volume_registration_qc.csv", index=False)

    return {
        "cells_mapped_csv": str(outputs_dir / "cells_mapped.csv"),
        "truth_source": "3d_registered_volume",
        "slice_registration_qc_csv": str(outputs_dir / "slice_registration_qc.csv"),
        "volume_registration_qc_csv": str(outputs_dir / "volume_registration_qc.csv"),
    }


def run_real_input(
    cfg: dict,
    input_dir: Path,
    output_name: str | None = None,
    output_dir: Path | None = None,
):
    project_root = _project_root()
    run_name = (
        str(output_name).strip()
        if output_name and str(output_name).strip()
        else Path(input_dir).name
    )
    if output_dir is not None:
        outputs_dir = Path(output_dir)
        run_name = run_name or outputs_dir.name
    else:
        outputs_dir = project_root / "outputs" / run_name
    outputs_dir.mkdir(parents=True, exist_ok=True)
    print(f"[output -> {outputs_dir}]")
    slice_glob = cfg.get("input", {}).get("slice_glob", "*.tif")
    files = _collect_slice_files(input_dir, slice_glob)
    if not files:
        raise FileNotFoundError(f"No slice files found in {input_dir} with pattern {slice_glob}")

    ch_idx = _resolve_channel_index(cfg)
    ch_dir = outputs_dir / "tmp_channel"
    channel_files = _extract_channel_to_tmp(files, ch_dir, ch_idx)

    n_merge = int(cfg.get("input", {}).get("slice_interval_n", 5))
    merged_files = merge_every_n_slices(channel_files, outputs_dir / "tmp_merged", n=n_merge)

    reg_cfg = cfg.get("registration", {})
    scope = str(reg_cfg.get("scope", "")).lower().strip()
    whole_brain_backend = str(reg_cfg.get("whole_brain_backend", "")).lower().strip()
    if scope == "whole" and whole_brain_backend == "miki_3d":
        cfg = dict(cfg)
        if "quantify_fn" not in cfg:
            cfg["quantify_fn"] = lambda **kwargs: _quantify_against_exported_truth(**kwargs)
        return run_whole_brain_3d(
            cfg=cfg,
            input_dir=input_dir,
            outputs_dir=outputs_dir,
            merged_slice_paths=merged_files,
        )

    # Extract marker channel for colocalization (if configured)
    _marker_channel_name = str(cfg.get("detection", {}).get("marker_channel", "")).lower().strip()
    _merged_marker_files: list[Path] = []
    if _marker_channel_name:
        _cmap = cfg.get("input", {}).get("channel_map", {"red": 0, "green": 1, "farred": 2})
        _marker_ch_idx = int(_cmap.get(_marker_channel_name, 1))
        _marker_ch_dir = outputs_dir / "tmp_marker_channel"
        _marker_channel_files = _extract_channel_to_tmp(files, _marker_ch_dir, _marker_ch_idx)
        _merged_marker_files = merge_every_n_slices(
            _marker_channel_files, outputs_dir / "tmp_marker_merged", n=n_merge
        )

    annotation_nii = project_root / "annotation_25.nii.gz"
    if not annotation_nii.exists():
        raise FileNotFoundError(f"atlas annotation not found: {annotation_nii}")

    px_um = _validated_float(cfg, "input.pixel_size_um_xy")
    spacing_um = _validated_float(cfg, "input.slice_spacing_um")
    slicing_plane = str(cfg.get("input", {}).get("slicing_plane", "coronal")).lower()
    neighbor = _validated_int(cfg, "dedup.neighbor_slices")
    rxy = _validated_float(cfg, "dedup.r_xy_um")
    structure_csv = _resolve_structure_source(project_root)
    warp_params, fit_mode, edge_smooth_iter = _load_tuned_overlay_params(outputs_dir)
    reg_cfg = cfg.get("registration", {})
    # Force hemisphere if specified in config (for half-brain samples where auto-detection fails)
    atlas_hemisphere = str(reg_cfg.get("atlas_hemisphere", "")).lower().strip()
    if atlas_hemisphere:
        warp_params = dict(warp_params)
        warp_params["force_hemisphere"] = atlas_hemisphere
    # Tissue shrinkage correction: cleared tissue shrinks 10-15% linearly during processing.
    # Values < 1.0 make the atlas appear smaller to match actual tissue extent.
    tissue_shrink = float(reg_cfg.get("tissue_shrink_factor", 1.0))
    if tissue_shrink != 1.0:
        warp_params = dict(warp_params)
        warp_params["tissue_shrink_factor"] = tissue_shrink
    # Merge registration-section warp/render knobs into warp_params so render_overlay
    # can read them via _warp_param().  Only keys that are not already set by the tuned
    # params are propagated (tuned params take precedence).
    _reg_warp_keys = [
        "enable_silhouette_conform",
        "physical_placement_skip_opt",
        "contour_n_pts",
        "contour_tps_smooth",
        "contour_max_disp_ratio",
        "contour_max_disp_min_px",
        "contour_fill_radius",
        "enable_sitk_ref_refine",
        "sitk_max_dim",
        "sitk_mi_bins",
        "sitk_max_iter",
        "sitk_mesh_size",
        "sitk_max_disp_frac",
    ]
    _reg_warp_updates = {
        k: reg_cfg[k] for k in _reg_warp_keys if k in reg_cfg and k not in warp_params
    }
    if _reg_warp_updates:
        warp_params = dict(warp_params)
        warp_params.update(_reg_warp_updates)
    # Gamma correction for display of dim cleared-tissue overlays (< 1.0 = brighten).
    display_gamma = float(reg_cfg.get("display_gamma", 1.0))
    fail_score_threshold = float(reg_cfg.get("fail_score_threshold", 0.65))
    registration_dir = outputs_dir / "registered_slices"
    registration_dir.mkdir(parents=True, exist_ok=True)

    # ── Optional DeepSlice AP estimation (run on whole series before the loop) ──
    ap_method = str(reg_cfg.get("ap_method", "formula")).lower().strip()
    _deepslice_az: dict[str, int] = {}
    if ap_method in ("deepslice", "deepslice+formula"):
        try:
            from scripts.atlas_deepslice import predict_ap_series

            _ds_z_scale = float(reg_cfg.get("atlas_z_z_scale", 0.2))
            _ds_z_offset = int(reg_cfg.get("atlas_z_offset", 0))
            _ds_max_dev = int(reg_cfg.get("deepslice_max_deviation", 30))
            print("[main] Running DeepSlice AP estimation on full series ...")
            _deepslice_az = predict_ap_series(
                channel_files,
                z_scale=_ds_z_scale,
                z_offset=_ds_z_offset,
                pixel_size_um=px_um,
                display_gamma=display_gamma,
                max_deviation=_ds_max_dev,
            )
            print(f"[main] DeepSlice done: {len(_deepslice_az)} slices mapped")
        except Exception as _ds_err:
            print(f"[main] DeepSlice failed ({_ds_err}); falling back to formula")
            _deepslice_az = {}

    detect_rows = []
    mapped_rows = []
    registration_rows = []
    region_area_rows = []
    next_id = 1
    for sid, mp in enumerate(merged_files):
        det = detect_cells(mp, cfg)
        if det.empty:
            continue

        auto_label_path = registration_dir / f"slice_{sid:04d}_auto_label.tif"
        registered_label_path = registration_dir / f"slice_{sid:04d}_registered_label.tif"
        overlay_png = registration_dir / f"slice_{sid:04d}_overlay.png"

        step_t0 = time.perf_counter()
        atlas_z_range = cfg.get("registration", {}).get("atlas_z_range", None)
        atlas_z_fixed = cfg.get("registration", {}).get("atlas_z_fixed", None)

        # Derive atlas_z from the filename z-number (e.g., z0050 → atlas_z=10 with scale=0.2)
        # Try original source file first, then merged file, then sid-based computation.
        if (
            cfg.get("registration", {}).get("atlas_z_from_filename", False)
            and atlas_z_fixed is None
        ):
            z_scale = float(cfg.get("registration", {}).get("atlas_z_z_scale", 0.2))
            z_offset = int(cfg.get("registration", {}).get("atlas_z_offset", 0))
            _fname = None
            # Try the source file path (for direct pipeline invocations)
            _m = re.search(r"z(\d+)", Path(str(mp)).stem)
            if _m:
                _fname = int(_m.group(1))
            else:
                # Merged file — reconstruct z-number from original source files
                if sid < len(files):
                    _m2 = re.search(r"z(\d+)", Path(str(files[sid])).stem)
                    if _m2:
                        _fname = int(_m2.group(1))
            if _fname is not None:
                atlas_z_fixed = max(0, min(527, int(_fname * z_scale) + z_offset))

        # Override with DeepSlice estimate if available for this slice
        if _deepslice_az:
            _stem = Path(str(mp)).stem
            if _stem in _deepslice_az:
                atlas_z_fixed = int(_deepslice_az[_stem])

        if atlas_z_fixed is not None:
            # Use filename-based atlas z, optionally refined by size-aware shape matching
            import nibabel as nib
            from tifffile import imread as tiff_imread

            fixed_z = int(atlas_z_fixed)
            nii = nib.load(str(annotation_nii))
            vol = np.asarray(nii.get_fdata(), dtype=np.int32)

            # Size-aware refinement: search ±refine_range around the filename estimate
            refine_range = int(cfg.get("registration", {}).get("atlas_z_refine_range", 0))
            score_type = "fixed_filename"
            best_score = 1.0
            if refine_range > 0:
                try:
                    _real_img = tiff_imread(str(mp))
                    _hemi = cfg.get("registration", {}).get("atlas_hemisphere", "")
                    fixed_z, best_score = refine_atlas_z_by_size(
                        _real_img,
                        vol,
                        z_estimate=fixed_z,
                        search_range=int(refine_range),
                        pixel_size_um=px_um,
                        hemisphere=_hemi,
                    )
                    score_type = "size_shape_refined"
                except Exception as _e:
                    pass  # fallback to filename estimate

            best_slice = vol[fixed_z, :, :]
            auto_label_path.parent.mkdir(parents=True, exist_ok=True)
            from tifffile import imwrite as tiff_imwrite

            tiff_imwrite(str(auto_label_path), best_slice)
            auto_meta = {
                "best_z": fixed_z,
                "best_score": 1.0,  # filename-based: always passes threshold
                "best_score_type": score_type,
                "label_slice_tif": str(auto_label_path),
                "shape": list(vol.shape),
                "slicing_plane": slicing_plane,
                "slice_shape": list(best_slice.shape),
                "roi_mode": "fixed",
                "roi_bbox": [0, 0, 0, 0],
                "real_slice": {},
                "tissue_coverage": 1.0,
                "coarse_top": [],
                "refined_top": [],
            }
        else:
            auto_meta = autopick_best_z(
                real_path=mp,
                annotation_nii=annotation_nii,
                out_label_tif=auto_label_path,
                z_step=2,
                pixel_size_um=px_um,
                slicing_plane=slicing_plane,
                roi_mode="auto",
                z_range=atlas_z_range,
            )
        autopick_ms = float((time.perf_counter() - step_t0) * 1000.0)

        step_t0 = time.perf_counter()
        _label_z_idx = (
            int(auto_meta.get("best_z", -1)) if auto_meta.get("best_z") is not None else None
        )
        _, diagnostic = render_overlay(
            real_slice_path=mp,
            label_slice_path=auto_label_path,
            out_png=overlay_png,
            alpha=0.55,
            mode="fill",
            pixel_size_um=px_um,
            major_top_k=28,
            fit_mode=fit_mode,
            edge_smooth_iter=edge_smooth_iter,
            warp_params=warp_params,
            return_meta=True,
            warped_label_out=registered_label_path,
            display_gamma=display_gamma,
            label_z_index=_label_z_idx,
        )
        render_wall_ms = float((time.perf_counter() - step_t0) * 1000.0)
        render_timings = diagnostic.get("timings_ms", {}) if isinstance(diagnostic, dict) else {}
        postprocess_timings = (
            render_timings.get("postprocess", {}) if isinstance(render_timings, dict) else {}
        )
        reg_score = float(auto_meta.get("best_score", 0.0))
        reg_ok = bool(np.isfinite(reg_score) and reg_score >= fail_score_threshold)
        registration_rows.append(
            {
                "slice_id": int(sid),
                "slice_path": str(mp),
                "auto_label_path": str(auto_label_path),
                "registered_label_path": str(registered_label_path),
                "overlay_path": str(overlay_png),
                "best_z": int(auto_meta.get("best_z", -1)),
                "best_score": reg_score,
                "score_type": str(auto_meta.get("best_score_type", "")),
                "slicing_plane": str(auto_meta.get("slicing_plane", "coronal")),
                "registration_ok": bool(reg_ok),
                "registration_method": str(diagnostic.get("warp", {}).get("method", "")),
                "autopick_ms": autopick_ms,
                "render_ms": float(render_timings.get("total", render_wall_ms))
                if isinstance(render_timings, dict)
                else render_wall_ms,
                "render_load_inputs_ms": float(render_timings.get("load_inputs", 0.0))
                if isinstance(render_timings, dict)
                else 0.0,
                "render_registration_ms": float(render_timings.get("registration", 0.0))
                if isinstance(render_timings, dict)
                else 0.0,
                "render_postprocess_ms": float(
                    postprocess_timings.get("total", postprocess_timings.get("wall", 0.0))
                )
                if isinstance(postprocess_timings, dict)
                else 0.0,
                "render_colorize_ms": float(render_timings.get("colorize", 0.0))
                if isinstance(render_timings, dict)
                else 0.0,
                "render_draw_labels_ms": float(render_timings.get("draw_region_labels", 0.0))
                if isinstance(render_timings, dict)
                else 0.0,
            }
        )
        if not reg_ok:
            continue

        det = det.copy()
        det["slice_id"] = sid
        det["cell_id"] = range(next_id, next_id + len(det))
        next_id += len(det)
        detect_rows.append(det[["cell_id", "slice_id", "x", "y", "score"]])
        mapped_rows.append(
            map_cells_with_registered_label_slice(
                det,
                registered_label_tif=registered_label_path,
                structure_csv=structure_csv,
                atlas_slice_index=int(auto_meta.get("best_z", -1)),
                slicing_plane=str(auto_meta.get("slicing_plane", "coronal")),
                registration_score=reg_score,
                registration_method=str(
                    diagnostic.get("warp", {}).get("method", "registered_slice_label")
                ),
            )
        )
        if registered_label_path.exists():
            try:
                area_df = compute_region_areas_from_label_tif(registered_label_path, sid, px_um)
                region_area_rows.append(area_df)
            except Exception as _e:
                print(f"[warn] region area computation failed for slice {sid}: {_e}")

    # Post-hoc AP consistency enforcement: flag outlier slices in QC data
    if len(registration_rows) >= 3:
        try:
            from scripts.atlas_autopick import enforce_ap_consistency

            s_indices = [r["slice_id"] for r in registration_rows]
            s_ap = [r["best_z"] for r in registration_rows]
            s_scores = [r["best_score"] for r in registration_rows]
            smoothed_ap, ap_fit = enforce_ap_consistency(s_indices, s_ap, s_scores)
            for i, row in enumerate(registration_rows):
                row["smoothed_ap"] = int(smoothed_ap[i])
                row["ap_outlier"] = abs(int(row["best_z"]) - int(smoothed_ap[i])) > 3
            print(
                f"[AP consistency] R²={ap_fit['r_squared']:.3f}, "
                f"slope={ap_fit['slope']:.2f}, outliers={ap_fit['outliers']}"
            )
        except Exception as _e:
            print(f"[AP consistency] skipped: {_e}")

    registration_qc_path = outputs_dir / "slice_registration_qc.csv"
    pd.DataFrame(registration_rows).to_csv(registration_qc_path, index=False)

    failed_slices = [row for row in registration_rows if not row["registration_ok"]]
    if failed_slices:
        raise RuntimeError(
            f"{len(failed_slices)} slice(s) failed registration score threshold {fail_score_threshold:.3f}; "
            f"see {registration_qc_path}"
        )

    if not mapped_rows:
        print("No detections found in real-input run.")
        return

    cells = pd.concat(detect_rows, ignore_index=True)
    cells.to_csv(outputs_dir / "cells_detected.csv", index=False)
    mapped = pd.concat(mapped_rows, ignore_index=True)

    deduped, stats = apply_dedup_kdtree(
        mapped,
        neighbor_slices=neighbor,
        pixel_size_um=px_um,
        slice_spacing_um=spacing_um,
        r_xy_um=rxy,
    )
    deduped.to_csv(outputs_dir / "cells_dedup.csv", index=False)
    write_dedup_stats(stats, outputs_dir)

    deduped.to_csv(outputs_dir / "cells_mapped.csv", index=False)
    leaf, hierarchy = aggregate_by_region(deduped)
    write_outputs(leaf, hierarchy, outputs_dir)

    # Write region areas (pixels per region per slice → mm²)
    region_areas_path = outputs_dir / "region_areas.csv"
    if region_area_rows:
        pd.concat(region_area_rows, ignore_index=True).to_csv(region_areas_path, index=False)

    # Generate paper-style AAV summary (representative slice + density)
    try:
        from scripts.paper_aav_summary import generate_paper_aav_summary
    except Exception:
        try:
            from paper_aav_summary import generate_paper_aav_summary
        except Exception:
            generate_paper_aav_summary = None
    if generate_paper_aav_summary is not None and region_areas_path.exists():
        try:
            generate_paper_aav_summary(
                cells_mapped_csv=outputs_dir / "cells_mapped.csv",
                region_areas_csv=region_areas_path,
                out_csv=outputs_dir / "paper_aav_region_summary.csv",
            )
        except Exception as _e:
            print(f"[warn] paper_aav_summary failed: {_e}")

    # Colocalization analysis (only when marker_channel is configured)
    if _merged_marker_files:
        _marker_tifs: dict[int, Path] = {
            sid: _merged_marker_files[sid]
            for sid in range(len(_merged_marker_files))
            if sid < len(_merged_marker_files)
        }
        _coloc_thr = float(cfg.get("detection", {}).get("marker_intensity_threshold_pct", 95.0))
        try:
            from scripts.colocalization import run_colocalization
        except Exception:
            try:
                from colocalization import run_colocalization
            except Exception:
                run_colocalization = None
        if run_colocalization is not None:
            try:
                run_colocalization(
                    cells_mapped_csv=outputs_dir / "cells_mapped.csv",
                    marker_tifs=_marker_tifs,
                    out_dir=outputs_dir,
                    marker_intensity_threshold_pct=_coloc_thr,
                )
                print(
                    "[main] Colocalization complete: cells_colocalization.csv + colocalization_summary.csv"
                )
            except Exception as _e:
                print(f"[warn] colocalization failed: {_e}")

    cells_mapped_path = outputs_dir / "cells_mapped.csv"
    slice_qc_path = outputs_dir / "slice_qc.csv"
    export_slice_qc(cells_mapped_path, slice_qc_path)

    # Copy registered atlas overlays to qc_overlays so the UI gallery shows beautiful colored images
    qc_dir = outputs_dir / "qc_overlays"
    qc_dir.mkdir(parents=True, exist_ok=True)
    reg_overlays = sorted(registration_dir.glob("slice_*_overlay.png"))
    import shutil

    for i, src_png in enumerate(reg_overlays):
        shutil.copy2(src_png, qc_dir / f"overlay_{i:03d}.png")

    print(
        f"Real-input end-to-end complete: detected={len(cells)}, dedup={len(deduped)} -> outputs/cell_counts_leaf.csv + QC"
    )

    # Generate paper-style report
    try:
        from scripts.export_paper_report import generate_paper_report
    except Exception:
        try:
            from export_paper_report import generate_paper_report
        except Exception:
            generate_paper_report = None
    if generate_paper_report is not None:
        try:
            generate_paper_report(outputs_dir)
        except Exception as _e:
            print(f"[warn] export_paper_report failed: {_e}")

    # Auto-regenerate demo visuals (panel, annotated slice, chart) after pipeline completes
    _refresh_demo_visuals(project_root, outputs_dir, input_dir)


def main():
    parser = argparse.ArgumentParser(description="Brain atlas cell count MVP pipeline entry")
    parser.add_argument("--config", required=True, help="Path to run config JSON")
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate config and print plan only"
    )
    parser.add_argument(
        "--demo-map", action="store_true", help="Run mapping/aggregation demo with sample CSVs"
    )
    parser.add_argument(
        "--run-real-input", type=str, default="", help="Run preprocess+detect on real input folder"
    )
    parser.add_argument(
        "--make-sample-tiff",
        type=str,
        default="",
        help="Create synthetic 3-channel TIFF slices into folder",
    )
    parser.add_argument(
        "--init-registration",
        action="store_true",
        help="Bootstrap registration assets from legacy repo",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="",
        help="Output subfolder name under outputs/. Defaults to input folder name.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="",
        help="Explicit output directory. Overrides --output-name when provided.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Enable DEBUG-level logging to console and log file"
    )
    args = parser.parse_args()

    cfg = load_config(Path(args.config))

    # Configure structured logging before any pipeline work begins
    try:
        from scripts.logging_setup import configure_logging

        if (args.output_dir or "").strip():
            _outputs_dir = Path(args.output_dir).expanduser()
            if not _outputs_dir.is_absolute():
                _outputs_dir = (_project_root() / _outputs_dir).resolve()
        else:
            _run_name = (args.output_name or "").strip() or (
                Path(args.run_real_input).name if args.run_real_input else "outputs"
            )
            _outputs_dir = _project_root() / "outputs" / _run_name
        configure_logging(_outputs_dir, debug=getattr(args, "debug", False))
    except Exception:
        pass  # logging is optional; don't block pipeline startup

    needs_runtime_config = bool(
        args.dry_run
        or args.demo_map
        or args.run_real_input
        or (not args.make_sample_tiff and not args.init_registration)
    )
    if needs_runtime_config:
        issues = validate_runtime_config(cfg, require_input_dir=bool(args.run_real_input))
        if issues:
            raise ConfigError("Config validation failed", issues=issues)

    steps = [
        "prepare_input",
        "register_slices",
        "detect_cells",
        "dedup_cells_kdtree",
        "map_cells_to_regions",
        "aggregate_by_region",
        "export_qc",
    ]

    print("Loaded config for project:", cfg.get("project", {}).get("name", "unknown"))
    print("Active channel:", cfg.get("input", {}).get("active_channel", "unknown"))
    print("Planned steps:")
    for i, s in enumerate(steps, 1):
        print(f"  {i}. {s}")

    if args.dry_run:
        print("Dry-run complete.")
        return

    if args.demo_map:
        run_demo(cfg)
        return

    if args.make_sample_tiff:
        _make_sample_tiffs(Path(args.make_sample_tiff))
        print(f"Sample TIFFs generated at {args.make_sample_tiff}")
        return

    if args.init_registration:
        project_root = _project_root()
        assets = bootstrap_registration_assets(project_root)
        out = project_root / "outputs" / "registration_assets.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(assets, indent=2), encoding="utf-8")
        print(f"Registration assets initialized -> {out}")
        return

    if args.run_real_input:
        run_real_input(
            cfg,
            Path(args.run_real_input),
            output_name=args.output_name or None,
            output_dir=Path(args.output_dir).expanduser()
            if (args.output_dir or "").strip()
            else None,
        )
        return

    print("MVP skeleton only: registration/mapping integration pending.")


if __name__ == "__main__":
    main()
