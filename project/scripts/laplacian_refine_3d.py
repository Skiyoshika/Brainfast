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
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import cg
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
# 2. Sparse Laplacian matrix
# ---------------------------------------------------------------------------


def _build_laplacian_3d(shape: tuple[int, int, int], spacing: tuple[float, ...]) -> csr_matrix:
    """Build 3D Laplacian matrix with spacing-weighted finite differences.

    For a grid of shape (D, H, W), the Laplacian at voxel (i,j,k) is:
        L[v] = sum_neighbors (u_neighbor - u_v) / h^2

    Returns sparse matrix of size (D*H*W, D*H*W).
    """
    D, H, W = shape
    N = D * H * W
    sz, sy, sx = [float(s) for s in spacing[:3]]

    # Weights for each axis
    wz = 1.0 / (sz * sz)
    wy = 1.0 / (sy * sy)
    wx = 1.0 / (sx * sx)

    rows, cols, vals = [], [], []

    def _idx(z, y, x):
        return z * H * W + y * W + x

    for z in range(D):
        for y in range(H):
            for x in range(W):
                v = _idx(z, y, x)
                diag_val = 0.0

                if z > 0:
                    rows.append(v)
                    cols.append(_idx(z - 1, y, x))
                    vals.append(wz)
                    diag_val -= wz
                if z < D - 1:
                    rows.append(v)
                    cols.append(_idx(z + 1, y, x))
                    vals.append(wz)
                    diag_val -= wz
                if y > 0:
                    rows.append(v)
                    cols.append(_idx(z, y - 1, x))
                    vals.append(wy)
                    diag_val -= wy
                if y < H - 1:
                    rows.append(v)
                    cols.append(_idx(z, y + 1, x))
                    vals.append(wy)
                    diag_val -= wy
                if x > 0:
                    rows.append(v)
                    cols.append(_idx(z, y, x - 1))
                    vals.append(wx)
                    diag_val -= wx
                if x < W - 1:
                    rows.append(v)
                    cols.append(_idx(z, y, x + 1))
                    vals.append(wx)
                    diag_val -= wx

                rows.append(v)
                cols.append(v)
                vals.append(diag_val)

    return csr_matrix(
        (np.array(vals, dtype=np.float64), (np.array(rows), np.array(cols))),
        shape=(N, N),
    )


def _build_laplacian_3d_fast(
    shape: tuple[int, int, int],
    spacing: tuple[float, ...],
) -> csr_matrix:
    """Vectorized Laplacian construction (much faster than triple loop)."""
    D, H, W = shape
    N = D * H * W
    sz, sy, sx = [float(s) for s in spacing[:3]]
    wz = 1.0 / (sz * sz)
    wy = 1.0 / (sy * sy)
    wx = 1.0 / (sx * sx)

    # Each voxel connects to up to 6 neighbours
    idx = np.arange(N, dtype=np.int64)
    z_idx = idx // (H * W)
    y_idx = (idx % (H * W)) // W
    x_idx = idx % W

    rows_list, cols_list, vals_list = [], [], []

    # z-axis neighbours
    mask = z_idx > 0
    rows_list.append(idx[mask])
    cols_list.append(idx[mask] - H * W)
    vals_list.append(np.full(mask.sum(), wz))

    mask = z_idx < D - 1
    rows_list.append(idx[mask])
    cols_list.append(idx[mask] + H * W)
    vals_list.append(np.full(mask.sum(), wz))

    # y-axis neighbours
    mask = y_idx > 0
    rows_list.append(idx[mask])
    cols_list.append(idx[mask] - W)
    vals_list.append(np.full(mask.sum(), wy))

    mask = y_idx < H - 1
    rows_list.append(idx[mask])
    cols_list.append(idx[mask] + W)
    vals_list.append(np.full(mask.sum(), wy))

    # x-axis neighbours
    mask = x_idx > 0
    rows_list.append(idx[mask])
    cols_list.append(idx[mask] - 1)
    vals_list.append(np.full(mask.sum(), wx))

    mask = x_idx < W - 1
    rows_list.append(idx[mask])
    cols_list.append(idx[mask] + 1)
    vals_list.append(np.full(mask.sum(), wx))

    # Diagonal: negative sum of neighbour weights
    diag_val = np.zeros(N, dtype=np.float64)
    nz_lo = (z_idx > 0).astype(np.float64)
    nz_hi = (z_idx < D - 1).astype(np.float64)
    ny_lo = (y_idx > 0).astype(np.float64)
    ny_hi = (y_idx < H - 1).astype(np.float64)
    nx_lo = (x_idx > 0).astype(np.float64)
    nx_hi = (x_idx < W - 1).astype(np.float64)
    diag_val = -(nz_lo + nz_hi) * wz - (ny_lo + ny_hi) * wy - (nx_lo + nx_hi) * wx

    rows_list.append(idx)
    cols_list.append(idx)
    vals_list.append(diag_val)

    all_rows = np.concatenate(rows_list)
    all_cols = np.concatenate(cols_list)
    all_vals = np.concatenate(vals_list)

    return csr_matrix((all_vals, (all_rows, all_cols)), shape=(N, N))


# ---------------------------------------------------------------------------
# 3. Laplace PDE solve with boundary conditions
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
    solve ∇²u = 0 in the interior with Dirichlet boundary conditions.

    Instead of modifying the Laplacian rows (expensive for large grids),
    we use a penalty method: add large weights to boundary voxels so the
    solution is strongly attracted to the boundary displacement values.

    Returns displacement field of shape (3, D, H, W).
    """
    D, H, W = shape
    N = D * H * W

    _log.info("Building Laplacian matrix for %dx%dx%d grid (%d DOF)...", D, H, W, N)
    L = _build_laplacian_3d_fast(shape, spacing)

    # Boundary condition penalty weight — large enough to enforce BC
    # but not so large as to make the system ill-conditioned
    penalty = 1e4

    # Build penalty matrix and RHS for each displacement component
    bc_flat_idx = (bc_coords[:, 0] * H * W + bc_coords[:, 1] * W + bc_coords[:, 2]).astype(np.int64)

    # Penalty diagonal: penalty at BC voxels, 0 elsewhere
    penalty_diag = np.zeros(N, dtype=np.float64)
    np.add.at(penalty_diag, bc_flat_idx, penalty * bc_weights)

    A = L + diags(penalty_diag, 0, shape=(N, N), format="csr")

    displacement_field = np.zeros((3, D, H, W), dtype=np.float32)

    for axis in range(3):
        # RHS: penalty * bc_displacement at boundary voxels
        rhs = np.zeros(N, dtype=np.float64)
        np.add.at(rhs, bc_flat_idx, penalty * bc_weights * bc_displacements[:, axis])

        _log.info("Solving Laplace equation for axis %d (CG, rtol=%.0e)...", axis, rtol)
        solution, info = cg(A, rhs, rtol=rtol, maxiter=maxiter)
        if info == 0:
            _log.info("  Axis %d converged", axis)
        else:
            _log.warning("  Axis %d did not converge (info=%d)", axis, info)

        displacement_field[axis] = solution.reshape(D, H, W).astype(np.float32)

    return displacement_field


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
