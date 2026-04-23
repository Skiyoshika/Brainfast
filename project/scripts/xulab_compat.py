"""Bridge Brainfast registration artifacts to Xu Lab RegTool's expected layout.

Brainfast's 3D registration writes ANTs transforms under
``<outputs>/ants_registration/`` with filenames:

    fwd_transform_0.nii.gz    — forward warp field
    fwd_transform_1.mat       — forward affine
    inv_transform_0.mat       — inverse affine
    inv_transform_1.nii.gz    — inverse warp field
    ants_result.nii.gz        — sample warped to CCF

Xu Lab's ``transform_input_points`` and ``count_registered_points``
expect a specific directory layout (see
``D:/UCI-XuLab-RegTools/regtools/registration/pipeline.py::_setup_stage_dirs``
and ``core/registration/ants.py::_transform_search_dirs``):

    <root>/01_axis_alignment/axisAlignA.npz            — identity when skipped
    <root>/02_nonlinear/fwd_transforms/ants_warp_0.nii.gz
    <root>/02_nonlinear/fwd_transforms/ants_affine_0.mat
    <root>/02_nonlinear/inv_transforms/ants_invwarp_0.nii.gz
    <root>/02_nonlinear/inv_transforms/ants_affine_0.mat
    <root>/02_nonlinear/result.nii.gz
    <root>/registration_metadata.json

This module materializes a Xu Lab-compatible directory from Brainfast's
output (via symlink where possible, else copy). Call it once per pipeline
run; Xu Lab functions can then be invoked with ``reg_output_dir = xu_root``.

Use case:
    - Reproduce Xu Lab's cell counting on Brainfast data without re-registering.
    - Cross-validate Brainfast's own region mapping against Xu Lab's.
    - Provide a clean interop point for any future Brainfast ↔ Xu Lab tooling.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

try:
    from scripts.logging_setup import get_logger
except ImportError:  # pragma: no cover
    from logging_setup import get_logger

log = get_logger(__name__)


def _link_or_copy(src: Path, dst: Path) -> None:
    """Prefer hardlink (Windows-friendly, no admin needed), else copy."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def build_xulab_compat_dir(
    *,
    brainfast_ants_dir: Path | str,
    sample_volume_path: Path | str,
    ccf_template_path: Path | str,
    xulab_out_dir: Path | str,
    axis_align_matrix: np.ndarray | None = None,
) -> Path:
    """Materialize a Xu Lab-layout registration output dir from Brainfast artifacts.

    Parameters
    ----------
    brainfast_ants_dir
        Path to Brainfast's ``<outputs>/ants_registration/`` directory.
    sample_volume_path
        Path to the sample ``input_volume.nii.gz`` used as moving image during
        registration. Written to metadata so Xu Lab can reconstruct moving
        coord system.
    ccf_template_path
        Path to the CCF template used as fixed image.
    xulab_out_dir
        Destination directory that will be populated with Xu Lab-layout files.
    axis_align_matrix
        Optional 4×4 affine to store in ``01_axis_alignment/axisAlignA.npz``.
        ``None`` → identity (Brainfast doesn't do the midline-fissure axis
        alignment that Xu Lab's ``axisAlignData`` provides, so points go
        straight to the nonlinear stage).

    Returns
    -------
    Path
        ``xulab_out_dir`` (the caller can pass this to Xu Lab as
        ``reg_output_dir``).
    """
    brainfast = Path(brainfast_ants_dir)
    xu_root = Path(xulab_out_dir)
    xu_root.mkdir(parents=True, exist_ok=True)

    stage1 = xu_root / "01_axis_alignment"
    stage2 = xu_root / "02_nonlinear"
    fwd_dir = stage2 / "fwd_transforms"
    inv_dir = stage2 / "inv_transforms"
    for d in (stage1, stage2, fwd_dir, inv_dir):
        d.mkdir(parents=True, exist_ok=True)

    # --- 01_axis_alignment: store identity (Brainfast doesn't axis-align) ---
    A = axis_align_matrix if axis_align_matrix is not None else np.eye(4)
    np.savez(stage1 / "axisAlignA.npz", A=A)

    # --- 02_nonlinear: link Brainfast's transforms into Xu Lab-named slots ---
    fwd_warp_src = brainfast / "fwd_transform_0.nii.gz"
    fwd_aff_src = brainfast / "fwd_transform_1.mat"
    inv_aff_src = brainfast / "inv_transform_0.mat"
    inv_warp_src = brainfast / "inv_transform_1.nii.gz"
    result_src = brainfast / "ants_result.nii.gz"

    missing = [
        p
        for p in [fwd_warp_src, fwd_aff_src, inv_aff_src, inv_warp_src, result_src]
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            f"Brainfast ants_registration/ is missing expected files: {[str(p) for p in missing]}"
        )

    _link_or_copy(fwd_warp_src, fwd_dir / "ants_warp_0.nii.gz")
    _link_or_copy(fwd_aff_src, fwd_dir / "ants_affine_0.mat")
    _link_or_copy(inv_aff_src, inv_dir / "ants_affine_0.mat")
    _link_or_copy(inv_warp_src, inv_dir / "ants_invwarp_0.nii.gz")
    _link_or_copy(result_src, stage2 / "result.nii.gz")

    # --- registration_metadata.json so Xu Lab can reconstruct axes ---
    sample_img = nib.load(str(sample_volume_path))
    ccf_img = nib.load(str(ccf_template_path))
    metadata = {
        "source": "brainfast-xulab-compat-bridge",
        "moving_image": str(Path(sample_volume_path).resolve()),
        "fixed_image": str(Path(ccf_template_path).resolve()),
        "moving_shape": [int(s) for s in sample_img.shape],
        "moving_spacing": [float(s) for s in sample_img.header.get_zooms()[:3]],
        "fixed_shape": [int(s) for s in ccf_img.shape],
        "fixed_spacing": [float(s) for s in ccf_img.header.get_zooms()[:3]],
        "registration_method": "ants",
        "half_mode": "whole",  # Brainfast does its own half-template crop upstream
    }
    (xu_root / "registration_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )

    log.info(
        "built Xu Lab-compat dir at %s from Brainfast %s (moving=%s, fixed=%s)",
        xu_root,
        brainfast,
        sample_img.shape,
        ccf_img.shape,
    )
    return xu_root
