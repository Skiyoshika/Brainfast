"""Laplacian refinement via Laplace-equation boundary-value solve.

Algorithm (matches Miki/UCI-XuLab-Registration approach):
1. Extract surface boundary correspondences between template and registered volume
2. Build a sparse Laplacian matrix on the 3D grid
3. Set boundary conditions from surface displacement
4. Solve Laplace equation with CG to get smooth interior displacement field
5. Apply displacement field to warp the registered volume

The Laplacian guarantees smooth, physically plausible deformation fields.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scripts.registration_3d_ants import compute_registration_metrics

_log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Surface correspondence extraction
# ---------------------------------------------------------------------------


def _extract_surface_mask(volume: np.ndarray, threshold: float = 0.05) -> np.ndarray:
    """Binary surface mask: dilated tissue minus eroded tissue."""
    tissue = volume > threshold * volume.max()
    dilated = binary_dilation(tissue, iterations=1)
    eroded = binary_erosion(tissue, iterations=1)
    return (dilated ^ eroded).astype(bool)


def _extract_boundary_correspondences(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    max_points: int = 300_000,
    block_size: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Find surface correspondences between fixed (template) and moving (registered).

    For each surface voxel in fixed, find the displacement to the closest
    surface voxel in moving using block-based local search.

    Returns:
        coords: (N, 3) int array of voxel coordinates in the grid
        displacements: (N, 3) float array of displacement vectors
        weights: (N,) float array of correspondence confidence
    """
    fixed_surf = _extract_surface_mask(fixed)
    moving_surf = _extract_surface_mask(moving)

    # Get surface voxel coordinates
    fz, fy, fx = np.where(fixed_surf)
    mz, my, mx = np.where(moving_surf)

    if len(fz) == 0 or len(mz) == 0:
        _log.warning("No surface voxels found; returning empty correspondences")
        return np.zeros((0, 3), int), np.zeros((0, 3), float), np.zeros(0, float)

    # Subsample if too many points
    if len(fz) > max_points:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(fz), max_points, replace=False)
        fz, fy, fx = fz[idx], fy[idx], fx[idx]

    # For each fixed surface point, find nearest moving surface point
    # within a local block (fast approximate nearest neighbour)
    from scipy.spatial import cKDTree

    moving_pts = np.column_stack([mz, my, mx]).astype(np.float64)
    fixed_pts = np.column_stack([fz, fy, fx]).astype(np.float64)

    tree = cKDTree(moving_pts)
    dists, indices = tree.query(fixed_pts, k=1, distance_upper_bound=block_size)

    # Filter out unmatched points (distance = inf)
    valid = np.isfinite(dists)
    coords = fixed_pts[valid].astype(int)
    matched_moving = moving_pts[indices[valid]].astype(np.float64)
    displacements = (matched_moving - coords.astype(np.float64)).astype(np.float32)

    # Weight by inverse distance (closer = more confident)
    weights = 1.0 / (dists[valid] + 1.0)
    weights = weights.astype(np.float32)

    _log.info(
        "Extracted %d boundary correspondences (from %d fixed surface voxels, "
        "median displacement=%.2f voxels)",
        len(coords),
        len(fz),
        float(np.median(np.linalg.norm(displacements, axis=1))),
    )

    return coords, displacements, weights


# ---------------------------------------------------------------------------
# 2. Laplace PDE solve with boundary conditions
#    (Sparse matrix construction delegated to regtools_laplacian — see
#    solveLaplacianFromCorrespondences for the Dirichlet BC + CG+Jacobi
#    implementation.)
# ---------------------------------------------------------------------------


def _solve_laplace_displacement(
    shape: tuple[int, int, int],
    spacing: tuple[float, ...],
    bc_coords: np.ndarray,
    bc_displacements: np.ndarray,
    bc_weights: np.ndarray,
    *,
    rtol: float = 1e-2,
    maxiter: int = 500,
) -> np.ndarray:
    """Solve Laplace equation for a smooth displacement field.

    Given boundary correspondences (surface points with known displacements),
    delegate to the vendored RegTools solver which uses true Dirichlet
    boundary conditions + CG with Jacobi preconditioner. This replaces the
    earlier penalty-method implementation that OOM'd on large volumes
    (see docs/.../2026-04-16-registration-quality-investigation.md Bug #4).

    ``bc_weights`` is no longer used: RegTools' Dirichlet solver treats
    boundary voxels as hard constraints, so per-point confidence weighting
    is not meaningful inside the solve. Weights remain in the signature so
    upstream callers that still pass them are unaffected.

    Returns displacement field of shape (3, D, H, W).
    """
    try:
        from scripts.regtools_laplacian import solveLaplacianFromCorrespondences
    except ImportError:
        from regtools_laplacian import solveLaplacianFromCorrespondences

    # Boundary correspondences are already template-space voxel coordinates,
    # so ``bc_coords`` are the Dirichlet *target* points; the matching moving-
    # space locations are ``bc_coords + bc_displacements`` which become the
    # *source* points. The solver returns a (3, D, H, W) field where axis d
    # contains the displacement along axis d at every voxel.
    source_pts = bc_coords.astype(float) + bc_displacements.astype(float)
    target_pts = bc_coords.astype(float)

    _ = bc_weights  # acknowledged, unused under Dirichlet BC

    def _log_adapter(msg, level="info"):
        if level == "warning":
            _log.warning(msg)
        else:
            _log.info(msg)

    deformation_field = solveLaplacianFromCorrespondences(
        vol_shape=shape,
        source_pts=source_pts,
        target_pts=target_pts,
        axes=(0, 1, 2),
        rtol=rtol,
        maxiter=maxiter,
        spacing=spacing,
        log_fn=_log_adapter,
    )
    return deformation_field.astype(np.float32)


# ---------------------------------------------------------------------------
# 4. Apply displacement field
# ---------------------------------------------------------------------------


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

    # Displaced coordinates
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


# ---------------------------------------------------------------------------
# 5. Main entry point
# ---------------------------------------------------------------------------


def refine_registered_volume(
    fixed_path: Path,
    moving_path: Path,
    out_dir: Path,
    iterations: int = 500,
    lambda_: float = 0.18,
) -> dict[str, Path]:
    """Laplacian refinement of ANTs-registered volume.

    Uses Laplace-equation boundary-value solve to compute a smooth
    displacement field from surface correspondences, then applies it.

    Args:
        fixed_path: Template volume (registration target).
        moving_path: ANTs-registered volume.
        out_dir: Output directory.
        iterations: CG solver max iterations.
        lambda_: Not used in PDE approach (kept for API compatibility).
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

    # Normalize volumes for surface extraction
    fixed_norm = fixed / max(float(np.percentile(fixed, 99.5)), 1.0)
    moving_norm = moving / max(float(np.percentile(moving, 99.5)), 1.0)

    # Step 1: Extract boundary correspondences
    bc_coords, bc_displacements, bc_weights = _extract_boundary_correspondences(
        fixed_norm,
        moving_norm,
        max_points=200_000,
    )

    if len(bc_coords) < 100:
        _log.warning(
            "Too few boundary correspondences (%d); skipping Laplacian refinement",
            len(bc_coords),
        )
        # Copy moving as-is
        final_path = out_dir / "final_registered.nii.gz"
        nib.save(
            nib.Nifti1Image(moving, moving_img.affine, moving_img.header),
            str(final_path),
        )
        field = np.zeros((3,) + moving.shape, dtype=np.float32)
        field_path = out_dir / "laplacian_deformation_field.npy"
        np.save(str(field_path), field)

        before = compute_registration_metrics(fixed, moving)
        _write_metrics(out_dir / "refinement_metrics.csv", before, before)
        return {
            "final_registered_path": final_path,
            "field_path": field_path,
            "metrics_csv": out_dir / "refinement_metrics.csv",
        }

    # Step 2-3: Solve Laplace equation for displacement field
    displacement_field = _solve_laplace_displacement(
        shape=fixed.shape,
        spacing=spacing,
        bc_coords=bc_coords,
        bc_displacements=bc_displacements,
        bc_weights=bc_weights,
        rtol=1e-2,
        maxiter=int(iterations),
    )

    # Step 4: Apply displacement to moving volume
    refined = _apply_displacement_field(moving, displacement_field)

    # Save outputs
    final_path = out_dir / "final_registered.nii.gz"
    nib.save(
        nib.Nifti1Image(refined.astype(np.float32), moving_img.affine, moving_img.header),
        str(final_path),
    )

    field_path = out_dir / "laplacian_deformation_field.npy"
    np.save(str(field_path), displacement_field)

    # Save boundary correspondences for visualization/debugging
    bc_path = out_dir / "boundary_conditions.csv"
    with bc_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["fz", "fy", "fx", "dz", "dy", "dx", "weight"])
        for i in range(len(bc_coords)):
            writer.writerow(
                [
                    bc_coords[i, 0],
                    bc_coords[i, 1],
                    bc_coords[i, 2],
                    f"{bc_displacements[i, 0]:.4f}",
                    f"{bc_displacements[i, 1]:.4f}",
                    f"{bc_displacements[i, 2]:.4f}",
                    f"{bc_weights[i]:.4f}",
                ]
            )
    _log.info("Saved %d boundary conditions to %s", len(bc_coords), bc_path)

    # Metrics
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
