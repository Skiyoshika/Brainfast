"""Laplacian refinement — Xu Lab sliceToSlice3DLaplacian parity.

Before 2026-04-22 Brainfast used a generic 3D surface-mask + KDTree
correspondence matcher, which was a Brainfast-specific approximation.
We now delegate correspondence extraction *and* the Laplacian solve to the
vendored Xu Lab pipeline (``regtools_laplacian.sliceToSlice3DLaplacian``):

- Per-slice Canny edges (sigma=3) + Otsu threshold on both template & data
- PCA-based 2D normals (oriented toward low intensity)
- Angle-constrained nearest-neighbour matching (degree_thresh=5°, k=30)
- Solve only in-plane displacements (dy, dx); dz is zero (slices pre-aligned)

This matches the canonical Xu Lab algorithm exactly. Brainfast's previous
3D-KDTree extractor has been removed — the user directive is to prefer
Xu Lab when algorithms conflict and to keep the package lean.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
from scripts.registration_3d_ants import compute_registration_metrics

_log = logging.getLogger(__name__)


def _apply_displacement_field(
    volume: np.ndarray,
    displacement_field: np.ndarray,
) -> np.ndarray:
    """Warp volume using displacement field via trilinear interpolation.

    displacement_field: (3, D, H, W) — displacement in voxels for each axis.
    """
    from scipy.ndimage import map_coordinates

    D, H, W = volume.shape
    zz, yy, xx = np.mgrid[:D, :H, :W]

    new_z = zz.astype(np.float64) + displacement_field[0]
    new_y = yy.astype(np.float64) + displacement_field[1]
    new_x = xx.astype(np.float64) + displacement_field[2]

    coords = np.array([new_z.ravel(), new_y.ravel(), new_x.ravel()])
    warped = map_coordinates(
        volume.astype(np.float64),
        coords,
        order=1,
        mode="constant",
        cval=0,
    )
    return warped.reshape(D, H, W).astype(volume.dtype)


def refine_registered_volume(
    fixed_path: Path,
    moving_path: Path,
    out_dir: Path,
    iterations: int = 500,
    lambda_: float = 0.18,
) -> dict[str, Path]:
    """Laplacian refinement via Xu Lab sliceToSlice3DLaplacian.

    Args:
        fixed_path: Template volume (registration target).
        moving_path: ANTs-registered volume.
        out_dir: Output directory.
        iterations: CG solver max iterations.
        lambda_: Kept for API compatibility; not used by Dirichlet BC solver.
    """
    fixed_path = Path(fixed_path)
    moving_path = Path(moving_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_img = nib.load(str(fixed_path))
    moving_img = nib.load(str(moving_path))
    fixed = np.asarray(fixed_img.dataobj, dtype=np.float32)
    moving = np.asarray(moving_img.dataobj, dtype=np.float32)

    if fixed.ndim != 3 or moving.ndim != 3:
        raise ValueError("fixed and moving volumes must both be 3D")
    if fixed.shape != moving.shape:
        raise ValueError("fixed and moving volumes must have the same shape")

    spacing = tuple(float(z) for z in fixed_img.header.get_zooms()[:3])
    if any(s <= 0 for s in spacing):
        spacing = (1.0, 1.0, 1.0)

    _ = lambda_  # acknowledged, preserved for API compat

    try:
        from scripts.regtools_laplacian import sliceToSlice3DLaplacian
    except ImportError:
        from regtools_laplacian import sliceToSlice3DLaplacian

    def _log_adapter(msg, level="info"):
        if level == "warn":
            _log.warning(msg)
        else:
            _log.info(msg)

    _log.info(
        "Laplacian refinement: calling Xu Lab sliceToSlice3DLaplacian (axis=0, "
        "rtol=1e-2, maxiter=%d, spacing=%s)",
        int(iterations),
        spacing,
    )

    displacement_field = sliceToSlice3DLaplacian(
        fixedImage=fixed,
        movingImage=moving,
        sliceMatchList="same",
        axis=0,
        output_dir=str(out_dir),
        rtol=1e-2,
        maxiter=int(iterations),
        return_residuals=False,
        spacing=spacing,
        solver_dtype="float64",
        solver_method="cg",
        log_fn=_log_adapter,
    ).astype(np.float32)

    refined = _apply_displacement_field(moving, displacement_field)

    final_path = out_dir / "final_registered.nii.gz"
    nib.save(
        nib.Nifti1Image(refined.astype(np.float32), moving_img.affine, moving_img.header),
        str(final_path),
    )

    field_path = out_dir / "laplacian_deformation_field.npy"
    np.save(str(field_path), displacement_field)

    before = compute_registration_metrics(fixed, moving)
    after = compute_registration_metrics(fixed, refined)
    _write_metrics(out_dir / "refinement_metrics.csv", before, after)

    _log.info(
        "Laplacian refinement: NCC %.4f -> %.4f, SSIM %.4f -> %.4f",
        before["NCC"],
        after["NCC"],
        before["SSIM"],
        after["SSIM"],
    )

    return {
        "final_registered_path": final_path,
        "field_path": field_path,
        "metrics_csv": out_dir / "refinement_metrics.csv",
    }


def _write_metrics(path: Path, before: dict, after: dict) -> None:
    metric_order = list(before.keys())
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric", "before", "after", "change"])
        writer.writeheader()
        for metric in metric_order:
            b = float(before[metric])
            a = float(after[metric])
            writer.writerow({"metric": metric, "before": b, "after": a, "change": a - b})
