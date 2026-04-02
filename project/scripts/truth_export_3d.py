from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from tifffile import imwrite

from scripts.overlay_render import render_overlay


def _select_volume_slice(volume: np.ndarray, index: int, slicing_plane: str) -> np.ndarray:
    plane = str(slicing_plane).lower()
    if plane == "sagittal":
        return volume[:, :, index]
    if plane == "horizontal":
        return volume[:, index, :]
    return volume[index, :, :]


def export_registered_truth_slices(
    real_slice_paths: list[Path],
    annotation_volume_path: Path,
    out_dir: Path,
    pixel_size_um: float,
    slicing_plane: str,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)

    annotation_img = nib.load(str(annotation_volume_path))
    volume = np.asarray(annotation_img.dataobj, dtype=np.int32)

    rows: list[dict] = []
    for idx, real_slice_path in enumerate(real_slice_paths):
        label_slice = _select_volume_slice(volume, idx, slicing_plane).astype(np.int32, copy=False)
        label_path = out_dir / f"slice_{idx:04d}_registered_label.tif"
        overlay_path = out_dir / f"slice_{idx:04d}_overlay.png"

        imwrite(str(label_path), label_slice)
        _, diagnostic = render_overlay(
            real_slice_path=real_slice_path,
            label_slice_path=label_path,
            out_png=overlay_path,
            alpha=0.72,
            mode="fill",
            pixel_size_um=float(pixel_size_um),
            major_top_k=28,
            fit_mode="cover",
            edge_smooth_iter=0,
            warp_params={},
            return_meta=True,
            warped_label_out=label_path,
        )

        rows.append(
            {
                "slice_id": int(idx),
                "real_slice_path": str(real_slice_path),
                "registered_label_path": str(label_path),
                "overlay_path": str(overlay_path),
                "registration_method": str(
                    diagnostic.get("warp", {}).get("method", "3d_truth_export")
                ),
            }
        )

    return rows
