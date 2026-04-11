from __future__ import annotations

from pathlib import Path
from typing import Literal

import nibabel as nib
import numpy as np
from tifffile import imread

Hemisphere = Literal["left", "right", "right_flipped"]


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
    # Limit downsampling to preserve tissue morphology for registration.
    # Miki's pipeline used 5 µm pixels; we cap at 4x to stay within ANTs
    # memory limits while keeping ≤ 3 µm resolution for sub-micron inputs.
    raw_factor = target_um / pixel_um_xy
    downsample_factor = max(1, min(4, round(raw_factor)))

    stack = []
    for p in slices:
        arr = imread(str(p)).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        stack.append(arr[::downsample_factor, ::downsample_factor])

    # Pad slices to common shape (different samples may have different dimensions)
    if stack:
        max_h = max(s.shape[0] for s in stack)
        max_w = max(s.shape[1] for s in stack)
        for i, s in enumerate(stack):
            if s.shape[0] != max_h or s.shape[1] != max_w:
                padded = np.zeros((max_h, max_w), dtype=s.dtype)
                padded[: s.shape[0], : s.shape[1]] = s
                stack[i] = padded

    vol = np.stack(stack, axis=0)
    lo = float(np.percentile(vol, 1))
    hi = float(np.percentile(vol, 99.5))
    scaled = np.clip((vol - lo) / max(hi - lo, 1.0) * 65535, 0, 65535).astype(np.uint16)

    # NIfTI convention: header zooms are in millimeters, not micrometers.
    # Allen atlas annotation_25.nii.gz uses 0.025 mm (= 25 µm) zooms.
    voxel_um = (
        z_spacing_um,
        pixel_um_xy * downsample_factor,
        pixel_um_xy * downsample_factor,
    )
    voxel_mm = tuple(v / 1000.0 for v in voxel_um)
    affine = np.diag([voxel_mm[0], voxel_mm[1], voxel_mm[2], 1.0])
    img = nib.Nifti1Image(scaled, affine)
    img.header.set_zooms(voxel_mm)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(output_path))

    return {
        "volume_path": output_path,
        "shape": list(scaled.shape),
        "downsample_factor": downsample_factor,
        "voxel_um": voxel_um,
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
    elif hemisphere == "right":
        template_half = template_data[:, :, mid:]
        annotation_half = annotation_data[:, :, mid:]
    else:
        template_half = template_data[:, :, :mid]
        annotation_half = annotation_data[:, :, :mid]

    tmpl_out = out_dir / "template_half.nii.gz"
    ann_out = out_dir / "annotation_half.nii.gz"

    # Allen CCFv3 atlas uses µm in its affine/zooms (diagonal = 25.0).
    # NIfTI convention is mm.  Convert zooms to mm.
    raw_zooms = template_img.header.get_zooms()[:3]
    if all(z > 1.0 for z in raw_zooms):
        zooms_mm = tuple(float(z) / 1000.0 for z in raw_zooms)
    else:
        zooms_mm = tuple(float(z) for z in raw_zooms)

    # Use a simple zero-origin diagonal affine so the template occupies the
    # same physical neighbourhood as the input volume (also at origin).  The
    # _crop_affine embeds atlas-space offsets that push the template millimetres
    # away from origin, causing ANTs to see zero overlap and produce an empty
    # registration result.  Voxel sizes are what matter for deformable
    # registration, not absolute position.
    simple_affine = np.diag([zooms_mm[0], zooms_mm[1], zooms_mm[2], 1.0])

    tmpl_img = nib.Nifti1Image(template_half.astype(np.float32), simple_affine)
    ann_img = nib.Nifti1Image(annotation_half.astype(np.int32), simple_affine)
    tmpl_img.header.set_zooms(zooms_mm)
    ann_img.header.set_zooms(zooms_mm)
    nib.save(tmpl_img, str(tmpl_out))
    nib.save(ann_img, str(ann_out))

    return {
        "template_path": tmpl_out,
        "annotation_path": ann_out,
        "hemisphere": hemisphere,
        "ap_range": [ap_start, ap_end],
        "shape": list(template_half.shape),
    }
