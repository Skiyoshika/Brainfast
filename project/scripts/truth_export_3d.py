from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from scripts.overlay_render import _detect_tissue, render_overlay
from tifffile import imread as tif_read
from tifffile import imwrite


def _select_volume_slice(volume: np.ndarray, index: int, slicing_plane: str) -> np.ndarray:
    plane = str(slicing_plane).lower()
    if plane == "sagittal":
        return volume[:, :, index]
    if plane == "horizontal":
        return volume[:, index, :]
    return volume[index, :, :]


def _volume_slice_count(volume: np.ndarray, slicing_plane: str) -> int:
    plane = str(slicing_plane).lower()
    if plane == "sagittal":
        return int(volume.shape[2])
    if plane == "horizontal":
        return int(volume.shape[1])
    return int(volume.shape[0])


def _coerce_bool(value, default: bool = True) -> bool:
    if value is None:
        return bool(default)
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


def _write_prewarped_label_only(
    *,
    real_slice_path: Path,
    label_slice: np.ndarray,
    label_path: Path,
) -> dict:
    """Write the registered label raster without rendering a PNG overlay."""
    real_img = tif_read(str(real_slice_path))
    if real_img.ndim == 3:
        real_img = real_img[0]

    label = np.asarray(label_slice, dtype=np.int32)
    if label.shape != real_img.shape:
        from scipy.ndimage import zoom as _zoom_label

        zf = (real_img.shape[0] / label.shape[0], real_img.shape[1] / label.shape[1])
        label = _zoom_label(label, zf, order=0).astype(np.int32)
        if label.shape != real_img.shape:
            tmp = np.zeros(real_img.shape, dtype=np.int32)
            h = min(label.shape[0], real_img.shape[0])
            w = min(label.shape[1], real_img.shape[1])
            tmp[:h, :w] = label[:h, :w]
            label = tmp

    tissue = _detect_tissue(real_img)
    tissue_mask = tissue.get("mask") if isinstance(tissue, dict) else None
    if tissue_mask is not None and tissue_mask.shape == label.shape:
        label = label * tissue_mask.astype(np.int32)

    label_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(str(label_path), label.astype(np.int32))
    return {"warp": {"method": "3d_truth_export_label_only"}}


def export_registered_truth_slices(
    real_slice_paths: list[Path],
    annotation_volume_path: Path,
    out_dir: Path,
    pixel_size_um: float,
    slicing_plane: str,
    *,
    warp_params: dict | None = None,
    atlas_hemisphere: str = "",
    overlay_alpha: float = 0.72,
    fit_mode: str = "cover",
    edge_smooth_iter: int = 0,
    progress_cb=None,
    write_overlays: bool = True,
    overlay_stride: int = 1,
) -> list[dict]:
    """Export per-slice registered-label rasters + overlays.

    ``fit_mode`` and ``edge_smooth_iter`` are caller-controlled so a UI-learned
    calibration can reach the default whole-brain path. Previously these were
    hard-coded to ``"cover"`` / ``0`` which silently bypassed calibration.

    The annotation volume is assumed to already be in sample space (the 3D
    ANTs registration has warped it there). ``render_overlay`` is called with
    ``prewarped_label=True`` to do only a nearest-neighbor resize to the
    real-image pixel grid, skipping the in-plane 2D warp.

    For the Xu Lab-canonical alternative (warp cell points into CCF instead
    of warping annotation into sample space), see
    :func:`scripts.cell_to_ccf.map_cells_via_ccf_transform`. The old
    ``annotation_sampling_mode='per_slice_native'`` toggle and its
    ``annotation_prewarped=False`` downstream flag were spike work addressing
    symptoms of a stale RAS affine bug in legacy ``input_volume.nii.gz``
    files; the root-cause fix lives in :mod:`scripts.migrate_volume_affine`.
    """
    annotation_img = nib.load(str(annotation_volume_path))
    volume = np.asarray(annotation_img.dataobj, dtype=np.int32)
    expected_slice_count = _volume_slice_count(volume, slicing_plane)
    if len(real_slice_paths) != expected_slice_count:
        raise ValueError(
            f"slice count mismatch for {str(slicing_plane).lower()} plane: "
            f"expected {expected_slice_count}, got {len(real_slice_paths)}"
        )

    out_dir.mkdir(parents=True, exist_ok=True)

    # Detect downsample factor from volume vs first real image
    _first_img = tif_read(str(real_slice_paths[0]))
    if _first_img.ndim == 3:
        _first_img = _first_img[0]
    vol_hw = _select_volume_slice(volume, 0, slicing_plane).shape
    # Volume was built with padding to max size; compute the downsample factor
    # from the largest dimension ratio across all slices.
    _ds_factor = None

    rows: list[dict] = []
    total_slices = len(real_slice_paths)
    write_overlays = _coerce_bool(write_overlays, True)
    overlay_stride = max(1, int(overlay_stride or 1))
    for idx, real_slice_path in enumerate(real_slice_paths):
        label_slice = _select_volume_slice(volume, idx, slicing_plane).astype(np.int32, copy=False)
        label_path = out_dir / f"slice_{idx:04d}_registered_label.tif"
        overlay_path = out_dir / f"slice_{idx:04d}_overlay.png"

        # Crop label to match this slice's actual footprint in the volume.
        # The volume was built by downsampling each slice independently, then
        # zero-padding to the max dimensions.  For slices smaller than the max,
        # the right/bottom of label_slice is padding that must be removed before
        # zoom to the real image size.
        real_img = tif_read(str(real_slice_path))
        if real_img.ndim == 3:
            real_img = real_img[0]
        real_h, real_w = real_img.shape[:2]
        if _ds_factor is None:
            # Infer from max real image covering the full volume slice
            _ds_factor = max(1, round(max(real_h, real_w) / max(vol_hw)))
        crop_h = min((real_h + _ds_factor - 1) // _ds_factor, label_slice.shape[0])
        crop_w = min((real_w + _ds_factor - 1) // _ds_factor, label_slice.shape[1])
        label_slice = label_slice[:crop_h, :crop_w]

        imwrite(str(label_path), label_slice)

        should_write_overlay = bool(write_overlays and idx % overlay_stride == 0)
        if should_write_overlay:
            # The 3D ANTs registration already placed annotation in sample space;
            # we only need a nearest-neighbor resize. ``warped_label_out=label_path``
            # tells render_overlay to rewrite the resized label back to disk so the
            # downstream mapping step uses the same canonical raster shown in the
            # overlay.
            _, diagnostic = render_overlay(
                real_slice_path=real_slice_path,
                label_slice_path=label_path,
                out_png=overlay_path,
                alpha=float(overlay_alpha),
                mode="fill",
                pixel_size_um=float(pixel_size_um),
                major_top_k=28,
                fit_mode=str(fit_mode),
                edge_smooth_iter=int(edge_smooth_iter),
                warp_params=dict(warp_params or {}),
                return_meta=True,
                prewarped_label=True,
                warped_label_out=label_path,
                min_mean_threshold=1.0,
            )
        else:
            diagnostic = _write_prewarped_label_only(
                real_slice_path=real_slice_path,
                label_slice=label_slice,
                label_path=label_path,
            )

        if idx % 20 == 0:
            method = diagnostic.get("warp", {}).get("method", "unknown")
            hemi = diagnostic.get("warp", {}).get("hemisphere_chosen", "?")
            print(
                f"  [truth-export] slice {idx}/{len(real_slice_paths)}: method={method}, hemisphere={hemi}"
            )
        if progress_cb is not None and (
            idx == 0 or (idx + 1) % 20 == 0 or (idx + 1) == total_slices
        ):
            try:
                progress_cb(idx + 1, total_slices)
            except Exception:
                pass

        rows.append(
            {
                "slice_id": int(idx),
                "real_slice_path": str(real_slice_path),
                "registered_label_path": str(label_path),
                "overlay_path": str(overlay_path) if should_write_overlay else "",
                "registration_method": str(
                    diagnostic.get("warp", {}).get("method", "3d_truth_export")
                ),
            }
        )

    return rows
