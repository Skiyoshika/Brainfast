"""Transform sample-space cell coordinates into CCF atlas voxel space and
look up their native-resolution region labels.

This module mirrors the Xu Lab reference implementation
(`regtools/registration/point_registration.py::transform_input_points` +
`regtools/registration/core/registration/ants.py::antsTransformPoints` +
`count_registered_points`) — Brainfast's older cell-mapping path instead
warped the CCF annotation *into* sample space, which dropped thin leaf
regions during Z downsampling. By transforming cell points *into* CCF
space and querying the native annotation, we preserve the full Allen
CCF leaf granularity that the Xu Lab pipeline has been producing all
along.

Pipeline (cell row → region label):

    (slice_id, x_pixel, y_pixel)
        │
        ▼   sample voxel coord (slice_id, y_vox, x_vox)
        │   derived from real-image pixel size + sample-volume affine
        ▼   physical coord in sample (moving) NIfTI frame
        │   physical = origin + direction @ (voxel * spacing)
        ▼   ANTs forward-point transform (moving → fixed / CCF)
        │   ants.apply_transforms_to_points with inverse transforms +
        │   affine inverted (affine is always inverted for point warps)
        ▼   physical coord in CCF frame
        │
        ▼   CCF voxel coord (int-rounded)
        │   voxel = direction.T @ (physical - origin) / spacing
        ▼   annotation[z, y, x]  →  Allen leaf region_id

Xu Lab references:
    - Point transform semantics: `antsTransformPoints` in
      `D:/UCI-XuLab-RegTools/regtools/registration/core/registration/ants.py`
      lines 547-814 (voxel↔physical conversion + ANTs invtransforms).
    - Pipeline orchestration: `transform_input_points` in
      `D:/UCI-XuLab-RegTools/regtools/registration/point_registration.py`
      lines 457-613.
    - Region lookup: `count_registered_points` in the same file,
      lines 1049-1180.
"""

from __future__ import annotations

import importlib
from collections.abc import Iterable
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

try:
    from scripts.logging_setup import get_logger
except ImportError:  # pragma: no cover — script-context fallback
    from logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# NIfTI affine helpers
# ---------------------------------------------------------------------------


def _affine_to_spacing_origin_direction(
    affine: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decompose a 4×4 NIfTI affine into (spacing, origin, direction).

    Matches ``ants.image_header_info`` semantics: spacing = per-axis
    magnitude, direction = 3×3 unit-length direction cosines. A caller
    that passes in a pure diagonal affine (typical for synthetic volumes)
    will get an identity direction matrix back.
    """
    A = np.asarray(affine, dtype=np.float64)
    M = A[:3, :3]
    spacing = np.linalg.norm(M, axis=0)
    spacing[spacing == 0] = 1.0  # guard against zero spacing → NaN direction
    direction = M / spacing[np.newaxis, :]
    origin = A[:3, 3]
    return spacing, origin, direction


def voxel_to_physical(
    voxel_zyx: np.ndarray,
    spacing: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Convert NIfTI voxel coords to physical coords in the frame defined
    by the image's affine. Expects input as ``(N, 3)`` in (z, y, x) order;
    returns ``(N, 3)`` physical coords in the same axis order.

    Formula: ``physical = origin + direction @ (voxel * spacing)``
    (applied row-wise).
    """
    v = np.asarray(voxel_zyx, dtype=np.float64)
    if v.ndim == 1:
        v = v.reshape(1, -1)
    scaled = v * spacing[np.newaxis, :]
    return origin[np.newaxis, :] + (direction @ scaled.T).T


def physical_to_voxel(
    physical_zyx: np.ndarray,
    spacing: np.ndarray,
    origin: np.ndarray,
    direction: np.ndarray,
) -> np.ndarray:
    """Inverse of :func:`voxel_to_physical`.

    Formula: ``voxel = direction.T @ (physical - origin) / spacing``.
    """
    p = np.asarray(physical_zyx, dtype=np.float64)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    shifted = p - origin[np.newaxis, :]
    rotated = (direction.T @ shifted.T).T
    return rotated / spacing[np.newaxis, :]


# ---------------------------------------------------------------------------
# ANTs point transform — mirrors Xu Lab antsTransformPoints
# ---------------------------------------------------------------------------


def _classify_transform(path: str) -> str:
    """Return ``"warp"`` for .nii/.nii.gz displacement fields, ``"affine"`` for .mat/.txt."""
    lower = str(path).lower()
    if lower.endswith(".mat") or lower.endswith(".txt"):
        return "affine"
    return "warp"


def _header_info_via_ants(path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read (spacing, origin, direction) through ``ants.image_header_info``.

    ANTs uses LPS physical coords while nibabel's NIfTI affine is RAS+ by
    convention, so reading the affine through nibabel and feeding it to
    ``ants.apply_transforms_to_points`` produces origin/direction sign
    mismatches — Xu Lab's code in ``antsTransformPoints`` (ants.py:650-653)
    specifically pulls spacing/origin/direction from
    ``ants.image_header_info``. We do the same here.
    """
    ants = importlib.import_module("ants")
    info = ants.image_header_info(str(path))
    spacing = np.asarray(info["spacing"], dtype=np.float64)
    origin = np.asarray(info["origin"], dtype=np.float64)
    # Per Xu Lab's ants.py:653 comment, ants.image_header_info returns
    # direction as a flat column-major array; np.reshape(3,3) with default
    # C-order gives the TRANSPOSE of the true direction cosine matrix, so
    # we transpose after reshaping (equivalent to order='F').
    direction = np.asarray(info["direction"], dtype=np.float64).reshape(3, 3).T
    return spacing, origin, direction


def _apply_axis_align_to_points(
    points_zyx: np.ndarray,
    axis_align_matrix: np.ndarray | None,
) -> np.ndarray:
    """Apply a 4×4 rigid axis-align affine to Nx3 voxel points (z, y, x).

    Matches Xu Lab's ``transform_input_points`` step 1: homogeneous coord
    multiply + strip trailing 1. Skip when the matrix is None or close to
    identity (``atol=1e-6``).
    """
    if axis_align_matrix is None:
        return points_zyx
    A = np.asarray(axis_align_matrix, dtype=np.float64)
    if A.shape != (4, 4):
        raise ValueError(f"axis_align_matrix must be 4x4, got {A.shape}")
    if np.allclose(A, np.eye(4), atol=1e-6):
        return points_zyx
    pts = np.asarray(points_zyx, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    homo = np.hstack([pts, ones])
    aligned = (A @ homo.T).T
    return aligned[:, :3]


def transform_points_sample_to_ccf(
    points_sample_voxel_zyx: np.ndarray,
    sample_volume_path: Path | str,
    ccf_template_path: Path | str,
    inverse_transforms: list[str],
    *,
    axis_align_matrix: np.ndarray | None = None,
) -> np.ndarray:
    """Transform Nx3 sample-voxel points (z, y, x) into CCF voxel space.

    Mirrors Xu Lab's ``antsTransformPoints`` in
    ``registration/core/registration/ants.py`` — point transformation is the
    *opposite direction* of image transformation, and the affine is always
    inverted for point warps regardless of the overall direction.

    For moving (sample) → fixed (CCF) point transformation we use the
    inverse transform list (``inv_transform_0.mat`` + ``inv_transform_1.nii.gz``
    in Brainfast's layout) with the affine's ``whichtoinvert`` flag set to
    ``True`` and warp fields kept as-is (``False``). Xu Lab's code note:
    "The affine is ALWAYS inverted for point transformation (in both
    directions). Warp fields are already the correct direction."

    ``sample_volume_path`` / ``ccf_template_path`` must point at the actual
    NIfTI files — the function reads spacing/origin/direction through
    ``ants.image_header_info`` to avoid nibabel's RAS+ vs ANTs's LPS
    coordinate-system mismatch (see ``_header_info_via_ants``).

    Parameters
    ----------
    points_sample_voxel_zyx
        ``(N, 3)`` sample input-volume voxel indices, in (z, y, x) order.
        Fractional values allowed — the caller may round to int *after* the
        round trip if integer voxel indexing is needed.
    sample_volume_path
        Path to the sample ``input_volume.nii.gz`` that was registered
        (moving image during ANTs registration).
    ccf_template_path
        Path to the CCF template (fixed image). Must be the same file — or
        at least same coord system — as was used during registration.
    inverse_transforms
        Brainfast's ``inv_transform_*`` list in any order. The function will
        sort them so affine is first (Xu Lab's expected input order for point
        transforms).

    Returns
    -------
    numpy.ndarray
        ``(N, 3)`` CCF voxel coordinates in (z, y, x) order. Caller should
        int-round + bounds-check before indexing annotation.
    """
    ants = importlib.import_module("ants")
    import pandas as _pd

    sample_spacing, sample_origin, sample_direction = _header_info_via_ants(sample_volume_path)
    ccf_spacing, ccf_origin, ccf_direction = _header_info_via_ants(ccf_template_path)

    # Sort inverse transforms so ANTs sees [affine, warp] — Xu Lab's expected
    # order for moving→fixed point warps.
    ordered: list[tuple[str, str]] = []
    for tf in inverse_transforms:
        ordered.append((_classify_transform(tf), str(tf)))
    ordered.sort(key=lambda e: 0 if e[0] == "affine" else 1)
    transform_paths = [p for _, p in ordered]
    which_to_invert = [kind == "affine" for kind, _ in ordered]

    # Optional axis-align affine (Xu Lab's transform_input_points step 1).
    points_aligned = _apply_axis_align_to_points(points_sample_voxel_zyx, axis_align_matrix)

    # Convert aligned-sample voxel → sample physical.
    physical_sample = voxel_to_physical(
        points_aligned, sample_spacing, sample_origin, sample_direction
    )

    # ANTs internally orders points as (x, y, z) even when the image axes are
    # (z, y, x). Xu Lab's comment (ants.py:782-790): "points[:,0] (our 'z' =
    # dim0) -> ANTs 'x'; points[:,1] (our 'y' = dim1) -> ANTs 'y'; points[:,2]
    # (our 'x' = dim2) -> ANTs 'z'."
    points_df = _pd.DataFrame(
        {
            "x": physical_sample[:, 0],
            "y": physical_sample[:, 1],
            "z": physical_sample[:, 2],
        }
    )
    transformed_df = ants.apply_transforms_to_points(
        dim=3,
        points=points_df,
        transformlist=transform_paths,
        whichtoinvert=which_to_invert,
    )

    physical_ccf = np.column_stack(
        [
            transformed_df["x"].values,
            transformed_df["y"].values,
            transformed_df["z"].values,
        ]
    ).astype(np.float64)

    # CCF physical → CCF voxel.
    return physical_to_voxel(physical_ccf, ccf_spacing, ccf_origin, ccf_direction)


# ---------------------------------------------------------------------------
# Volume warp sample → CCF — mirrors Xu Lab transform_input_data
# ---------------------------------------------------------------------------


def transform_volume_sample_to_ccf(
    sample_volume_path: Path | str,
    *,
    ccf_template_path: Path | str,
    inverse_transforms: list[str],
    output_path: Path | str,
    interpolator: str = "linear",
    axis_align_matrix: np.ndarray | None = None,
) -> Path:
    """Warp a sample-space 3D volume into CCF space using the same ANTs
    transforms that :func:`transform_points_sample_to_ccf` uses for points.

    Mirrors Xu Lab's ``transform_input_data``
    (``regtools/registration/point_registration.py:620``) — axis-align step
    first (pre-rotate the volume via ``scipy.ndimage.affine_transform`` if
    ``axis_align_matrix`` is non-identity), then ANTs image warp.

    For label volumes (discrete IDs) pass ``interpolator='genericLabel'`` or
    ``'nearestNeighbor'``; for intensity volumes use ``'linear'``.
    """
    ants = importlib.import_module("ants")
    from scipy.ndimage import affine_transform as _aff_tx

    sample_volume_path = Path(sample_volume_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Optional axis-align pre-rotation on the voxel grid
    intermediate_path = sample_volume_path
    if axis_align_matrix is not None and not np.allclose(axis_align_matrix, np.eye(4), atol=1e-6):
        img = nib.load(str(sample_volume_path))
        data = np.asarray(img.dataobj)
        aligned = _aff_tx(
            data,
            np.linalg.inv(np.asarray(axis_align_matrix, dtype=np.float64)),
            output_shape=data.shape,
            order=1,
        )
        aligned = np.where(aligned < 0, 0, aligned).astype(data.dtype)
        intermediate_path = output_path.with_name(output_path.stem + "_axisAligned.nii.gz")
        nib.save(nib.Nifti1Image(aligned, img.affine, img.header), str(intermediate_path))

    # Sort transforms [affine, warp] with affine inverted
    ordered: list[tuple[str, str]] = [
        (_classify_transform(tf), str(tf)) for tf in inverse_transforms
    ]
    ordered.sort(key=lambda e: 0 if e[0] == "affine" else 1)
    transform_paths = [p for _, p in ordered]

    fixed = ants.image_read(str(ccf_template_path))
    moving = ants.image_read(str(intermediate_path))
    warped = ants.apply_transforms(
        fixed=fixed,
        moving=moving,
        transformlist=transform_paths,
        interpolator=interpolator,
    )
    ants.image_write(warped, str(output_path))
    return output_path


# ---------------------------------------------------------------------------
# Region lookup — mirrors Xu Lab count_registered_points core loop
# ---------------------------------------------------------------------------


def lookup_region_ids_from_ccf_voxels(
    ccf_voxel_zyx: np.ndarray,
    annotation_path: Path | str,
) -> tuple[np.ndarray, int]:
    """Query the CCF annotation volume at integer-rounded voxel indices.

    Matches Xu Lab's ``count_registered_points`` lookup loop
    (``point_registration.py`` lines 1139-1148): out-of-bounds points are
    assigned region_id 0, otherwise the native annotation voxel is returned.

    Returns ``(region_ids, out_of_bounds_count)``.
    """
    ann = np.asarray(nib.load(str(annotation_path)).dataobj, dtype=np.int64)
    pts = np.rint(ccf_voxel_zyx).astype(np.int64)
    nz, ny, nx = ann.shape
    region_ids = np.zeros(pts.shape[0], dtype=np.int64)
    in_bounds = (
        (pts[:, 0] >= 0)
        & (pts[:, 0] < nz)
        & (pts[:, 1] >= 0)
        & (pts[:, 1] < ny)
        & (pts[:, 2] >= 0)
        & (pts[:, 2] < nx)
    )
    valid = pts[in_bounds]
    region_ids[in_bounds] = ann[valid[:, 0], valid[:, 1], valid[:, 2]]
    return region_ids, int((~in_bounds).sum())


# ---------------------------------------------------------------------------
# Top-level Brainfast cell-dataframe helper
# ---------------------------------------------------------------------------


def sample_pixel_to_volume_voxel(
    cell_x_pixel: np.ndarray,
    cell_y_pixel: np.ndarray,
    cell_slice_id: np.ndarray,
    pixel_size_um: float,
    sample_volume_zooms_mm: Iterable[float],
) -> np.ndarray:
    """Convert real-image cell pixel coords to sample NIfTI voxel coords (z, y, x).

    Brainfast stores detected cells with ``(x, y)`` in original TIFF pixel
    units (``pixel_size_um`` µm per pixel) and ``slice_id`` as the sample's
    Z-index. The sample volume that ANTs saw was resampled to a coarser XY
    grid (per ``volume_io._resolve_downsample``) but the Z axis is the same
    ``slice_id``.

    We derive the XY voxel index by dividing the real pixel coords by the
    volume's XY spacing / pixel_size ratio.
    """
    zooms = tuple(float(z) for z in sample_volume_zooms_mm)
    if len(zooms) < 3:
        raise ValueError("sample_volume_zooms_mm must have at least 3 entries (z, y, x)")
    vol_xy_um = float(zooms[1]) * 1000.0
    xy_ratio = vol_xy_um / float(pixel_size_um)
    if xy_ratio <= 0:
        raise ValueError(
            f"Invalid XY ratio: volume_xy_um={vol_xy_um}, pixel_size_um={pixel_size_um}"
        )
    z = np.asarray(cell_slice_id, dtype=np.float64)
    y = np.asarray(cell_y_pixel, dtype=np.float64) / xy_ratio
    x = np.asarray(cell_x_pixel, dtype=np.float64) / xy_ratio
    return np.column_stack([z, y, x])


def map_cells_via_ccf_transform(
    cells: pd.DataFrame,
    *,
    sample_volume_path: Path | str,
    ccf_annotation_path: Path | str,
    inverse_transforms: list[str],
    pixel_size_um: float,
    ccf_template_path: Path | str | None = None,
    axis_align_matrix: np.ndarray | None = None,
) -> pd.DataFrame:
    """End-to-end Brainfast cell → CCF region_id lookup (Xu Lab-aligned).

    The returned DataFrame is the input ``cells`` plus columns:
    ``ccf_z_voxel, ccf_y_voxel, ccf_x_voxel, region_id, mapping_status``.

    Structure metadata enrichment (region_name / acronym / hierarchy path)
    is the caller's responsibility — reuse ``atlas_mapper._attach_structure_metadata``.
    """
    sample_img = nib.load(str(sample_volume_path))
    sample_zooms = sample_img.header.get_zooms()

    ccf_coord_src: Path | str = ccf_template_path or ccf_annotation_path

    if cells.empty:
        out = cells.copy()
        for col in ("ccf_z_voxel", "ccf_y_voxel", "ccf_x_voxel", "region_id"):
            out[col] = pd.Series(dtype=np.int64)
        out["mapping_status"] = pd.Series(dtype=object)
        return out

    sample_voxel_zyx = sample_pixel_to_volume_voxel(
        cell_x_pixel=cells["x"].to_numpy(),
        cell_y_pixel=cells["y"].to_numpy(),
        cell_slice_id=cells["slice_id"].to_numpy(),
        pixel_size_um=pixel_size_um,
        sample_volume_zooms_mm=sample_zooms,
    )
    ccf_voxel_zyx = transform_points_sample_to_ccf(
        sample_voxel_zyx,
        sample_volume_path=sample_volume_path,
        ccf_template_path=ccf_coord_src,
        inverse_transforms=list(inverse_transforms),
        axis_align_matrix=axis_align_matrix,
    )
    region_ids, oob = lookup_region_ids_from_ccf_voxels(ccf_voxel_zyx, ccf_annotation_path)

    out = cells.copy().reset_index(drop=True)
    ccf_round = np.rint(ccf_voxel_zyx).astype(np.int64)
    out["ccf_z_voxel"] = ccf_round[:, 0]
    out["ccf_y_voxel"] = ccf_round[:, 1]
    out["ccf_x_voxel"] = ccf_round[:, 2]
    out["region_id"] = region_ids.astype(np.int64)
    out["mapping_status"] = np.where(region_ids > 0, "ok", "outside_ccf_or_unassigned")
    log.info(
        "map_cells_via_ccf_transform: %d cells → CCF; %d out-of-bounds; %d unique region_ids",
        len(out),
        oob,
        int(pd.Series(region_ids).nunique()),
    )
    return out
