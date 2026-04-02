from __future__ import annotations

from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
from tifffile import imread

Hemisphere = Literal["left", "right", "right_flipped"]


def _crop_affine(
    affine: np.ndarray,
    ap_start: int,
    ml_start: int,
    ml_size: int,
    flipped: bool,
) -> np.ndarray:
    transform = np.eye(4, dtype=float)
    transform[0, 3] = float(ap_start)
    if flipped:
        transform[2, 2] = -1.0
        transform[2, 3] = float(ml_start + ml_size - 1)
    else:
        transform[2, 3] = float(ml_start)
    return affine @ transform


def build_volume_from_tiffs(
    slice_dir: Path | str,
    output_path: Path | str,
    pixel_um_xy: float,
    z_spacing_um: float,
    target_um: float = 25.0,
    glob_pattern: str = "z*.tif",
) -> dict[str, object]:
    slice_dir = Path(slice_dir)
    output_path = Path(output_path)

    slices = sorted(slice_dir.glob(glob_pattern))
    if not slices:
        raise FileNotFoundError(f"No {glob_pattern} in {slice_dir}")

    pixel_um_xy = float(pixel_um_xy)
    z_spacing_um = float(z_spacing_um)
    target_um = float(target_um)
    downsample_factor = max(1, round(target_um / pixel_um_xy))

    stack = []
    for p in slices:
        arr = imread(str(p)).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        stack.append(arr[::downsample_factor, ::downsample_factor])

    vol = np.stack(stack, axis=0)
    lo = float(np.percentile(vol, 1))
    hi = float(np.percentile(vol, 99.5))
    scaled = np.clip((vol - lo) / max(hi - lo, 1.0) * 65535, 0, 65535).astype(np.uint16)

    voxel_mm = (
        z_spacing_um / 1000.0,
        pixel_um_xy * downsample_factor / 1000.0,
        pixel_um_xy * downsample_factor / 1000.0,
    )
    affine = np.diag([voxel_mm[0], voxel_mm[1], voxel_mm[2], 1.0])
    img = nib.Nifti1Image(scaled, affine)
    img.header.set_zooms(voxel_mm)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(output_path))

    return {
        "volume_path": output_path,
        "shape": list(scaled.shape),
        "downsample_factor": downsample_factor,
        "voxel_mm": voxel_mm,
        "slice_count": len(slices),
    }


def prepare_half_template_inputs(
    template_path: Path | str,
    annotation_path: Path | str,
    hemisphere: Hemisphere,
    ap_start: int,
    ap_end: int,
    out_dir: Path | str,
) -> dict[str, object]:
    template_path = Path(template_path)
    annotation_path = Path(annotation_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    template_img = nib.load(str(template_path))
    annotation_img = nib.load(str(annotation_path))

    template_data = np.asarray(template_img.dataobj)
    annotation_data = np.asarray(annotation_img.dataobj)

    ap_slice = slice(ap_start, ap_end)
    template_data = template_data[ap_slice, :, :]
    annotation_data = annotation_data[ap_slice, :, :]

    mid = template_data.shape[2] // 2
    if hemisphere == "right_flipped":
        template_half = template_data[:, :, mid:][:, :, ::-1].copy()
        annotation_half = annotation_data[:, :, mid:][:, :, ::-1].copy()
        affine = _crop_affine(template_img.affine, ap_start, mid, template_half.shape[2], True)
        ann_affine = _crop_affine(annotation_img.affine, ap_start, mid, annotation_half.shape[2], True)
    elif hemisphere == "right":
        template_half = template_data[:, :, mid:]
        annotation_half = annotation_data[:, :, mid:]
        affine = _crop_affine(template_img.affine, ap_start, mid, template_half.shape[2], False)
        ann_affine = _crop_affine(annotation_img.affine, ap_start, mid, annotation_half.shape[2], False)
    else:
        template_half = template_data[:, :, :mid]
        annotation_half = annotation_data[:, :, :mid]
        affine = _crop_affine(template_img.affine, ap_start, 0, template_half.shape[2], False)
        ann_affine = _crop_affine(annotation_img.affine, ap_start, 0, annotation_half.shape[2], False)

    tmpl_out = out_dir / "template_half.nii.gz"
    ann_out = out_dir / "annotation_half.nii.gz"

    tmpl_img = nib.Nifti1Image(template_half.astype(np.float32), affine)
    ann_img = nib.Nifti1Image(annotation_half.astype(np.int32), ann_affine)
    tmpl_img.header.set_zooms(template_img.header.get_zooms()[:3])
    ann_img.header.set_zooms(annotation_img.header.get_zooms()[:3])
    nib.save(tmpl_img, str(tmpl_out))
    nib.save(ann_img, str(ann_out))

    return {
        "template_path": tmpl_out,
        "annotation_path": ann_out,
        "hemisphere": hemisphere,
        "ap_range": [ap_start, ap_end],
        "shape": list(template_half.shape),
    }
