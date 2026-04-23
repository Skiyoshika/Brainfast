"""Midline-fissure-based rigid pre-alignment for 3D volumes.

Brainfast-native port of Xu Lab RegTool's ``axisAlignData`` +
``getAlignAxisAffineMatrix`` + ``vol2affine`` (registration/core/io_utils.py,
vol2affine.py, align_utils.py). The algorithm:

  1. For each coronal slice of both volumes (moving and template):
     a. Otsu-threshold + largest-contour bounding box to find tissue extent.
     b. Restrict the line search to a narrow centre column (midpoint ± ratio × width).
     c. Canny edge detection + probabilistic Hough transform to find nearly-vertical
        line segments (angles 80-100° from horizontal) within that centre column —
        these are candidate longitudinal-fissure fragments.
     d. Average all valid fragments per slice → one representative fissure line.
     e. Record the two endpoints as 3D points (slice_index, y, x).
  2. SVD-fit a plane to each volume's accumulated fissure points → plane normal.
  3. Compute the rotation that maps the moving-normal to the template-normal
     (using :class:`scipy.spatial.transform.Rotation`).
  4. Build a 4×4 affine ``A`` that rotates the moving volume around its own centre.
  5. Apply ``A`` to the moving array via ``scipy.ndimage.affine_transform``.

Independent implementation — not a copy. Follows the same algorithm to produce
the same output semantics (rotation angle + aligned volume + fissure points),
but renames helpers, drops OpenCV-specific debug plumbing, and uses clearer
control flow. Xu Lab's PIR orientation convention assumed (axis 0 = posterior,
axis 1 = inferior, axis 2 = right).
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation

try:
    from scripts.logging_setup import get_logger
except ImportError:  # pragma: no cover
    from logging_setup import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Slice-level helpers
# ---------------------------------------------------------------------------


def _preprocess_slice(slice_2d: np.ndarray, max_val: int = 400) -> np.ndarray:
    """Clip to ``max_val`` + normalise to uint8 for OpenCV."""
    arr = np.asarray(slice_2d, dtype=np.float32)
    arr = np.clip(arr, 0, float(max_val))
    if arr.max() > 0:
        arr = arr / arr.max() * 255.0
    return arr.astype(np.uint8)


def _centre_column_bounds(slice_u8: np.ndarray, ratio: float) -> tuple[float, float] | None:
    """Return (left, right) x-bounds of the narrow centre column to search for fissure lines.

    Otsu-threshold → largest contour → bounding box midpoint ± ``ratio`` × width.
    Returns ``None`` if no tissue contour is found.
    """
    _, thresh = cv2.threshold(slice_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    if not len(cnts):
        return None
    largest_area = -1
    out: tuple[float, float] | None = None
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if w * h > largest_area:
            largest_area = w * h
            midpoint = x + w / 2.0
            out = (midpoint - ratio * w, midpoint + ratio * w)
    return out


def _hough_lines(
    slice_u8: np.ndarray,
    *,
    min_thresh: int = 150,
    max_thresh: int = 250,
    rho: float = 1.0,
    theta: float = np.pi / 180,
    line_thresh: int = 15,
    min_line_length: int = 30,
    max_line_gap: int = 20,
    blur_kernel_size: int = 0,
) -> np.ndarray | None:
    """Canny + probabilistic Hough → Nx1x4 array of (x1,y1,x2,y2), or None."""
    img = slice_u8
    if blur_kernel_size > 0:
        img = cv2.GaussianBlur(img, (blur_kernel_size, blur_kernel_size), 0)
    edges = cv2.Canny(img, min_thresh, max_thresh, None, 3)
    return cv2.HoughLinesP(
        edges, rho, theta, line_thresh, np.array([]), min_line_length, max_line_gap
    )


def _average_fissure_line(
    lines: np.ndarray | None,
    left: float,
    right: float,
    *,
    min_angle: float = 80.0,
    max_angle: float = 100.0,
) -> tuple[int, int, int, int] | None:
    """Filter lines to nearly-vertical segments inside the centre column, average the survivors.

    Returns ``(x1, y1, x2, y2)`` of the averaged line or ``None`` if no valid lines.
    """
    if lines is None:
        return None
    valid: list[tuple[int, int, int, int]] = []
    for line in lines:
        x1, y1, x2, y2 = [int(v) for v in line[0]]
        angle_deg = math.degrees(math.atan2(y2 - y1, x2 - x1))
        in_bounds = (left <= x1 <= right) and (left <= x2 <= right)
        if min_angle < abs(angle_deg) < max_angle and in_bounds:
            # Canonicalise: swap endpoints so the line points 'up' (consistent y ordering)
            if angle_deg < 0:
                valid.append((x2, y2, x1, y1))
            else:
                valid.append((x1, y1, x2, y2))
    if not valid:
        return None
    arr = np.asarray(valid, dtype=np.float32)
    avg = arr.mean(axis=0)
    return (int(avg[0]), int(avg[1]), int(avg[2]), int(avg[3]))


def _collect_fissure_points(
    volume: np.ndarray,
    *,
    slice_range: tuple[int, int] | None = None,
    max_val: int = 400,
    bbox_ratio: float = 0.05,
    line_kwargs: dict | None = None,
    avg_angle_bounds: tuple[float, float] = (80.0, 100.0),
) -> np.ndarray:
    """For each slice in ``volume[slice_range[0]:slice_range[1]]``, detect the
    longitudinal-fissure line and append its two endpoints as 3D points
    ``(slice_index, y, x)``.

    Returns an ``(N, 3)`` float array (possibly empty).
    """
    if slice_range is None:
        n = int(volume.shape[0])
        slice_range = (max(0, int(0.078 * n)), min(n, int(0.72 * n)))
    pts: list[tuple[int, int, int]] = []
    line_kwargs = dict(line_kwargs or {})
    for i in range(slice_range[0], slice_range[1]):
        img_u8 = _preprocess_slice(volume[i], max_val=max_val)
        bounds = _centre_column_bounds(img_u8, bbox_ratio)
        if bounds is None:
            continue
        left, right = bounds
        lines = _hough_lines(img_u8, **line_kwargs)
        avg = _average_fissure_line(
            lines, left, right, min_angle=avg_angle_bounds[0], max_angle=avg_angle_bounds[1]
        )
        if avg is None:
            continue
        x1, y1, x2, y2 = avg
        pts.append((i, y1, x1))
        pts.append((i, y2, x2))
    return np.asarray(pts, dtype=np.float64) if pts else np.empty((0, 3), dtype=np.float64)


# ---------------------------------------------------------------------------
# Plane fitting + rotation
# ---------------------------------------------------------------------------


def _svd_plane_normal(points: np.ndarray) -> np.ndarray:
    """Return the unit normal vector of the best-fit plane through ``points``
    using SVD on the centred coordinates. Raises ``ValueError`` if < 3 points.
    """
    if points.shape[0] < 3:
        raise ValueError(f"need at least 3 points for plane fit, got {points.shape[0]}")
    centroid = points.mean(axis=0)
    centered = points - centroid
    # Last right-singular vector is the smallest variance direction → plane normal
    _, _, vh = np.linalg.svd(centered.T @ centered)
    normal = vh[-1]
    n = np.linalg.norm(normal)
    if n == 0:
        raise ValueError("degenerate point cloud (zero-norm normal)")
    return normal / n


def _rotation_matrix_between(v_from: np.ndarray, v_target: np.ndarray) -> np.ndarray:
    """3×3 rotation mapping unit vector ``v_from`` onto ``v_target``."""
    a = v_from / np.linalg.norm(v_from)
    b = v_target / np.linalg.norm(v_target)
    axis = np.cross(a, b)
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        # Parallel or antiparallel vectors
        if np.dot(a, b) > 0:
            return np.eye(3)
        # Antiparallel: 180° rotation around any axis perpendicular to a
        # Pick a world axis that isn't parallel to a
        perp = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, perp)
        axis /= np.linalg.norm(axis)
        return Rotation.from_rotvec(np.pi * axis).as_matrix()
    axis = axis / axis_norm
    angle = float(np.arccos(np.clip(np.dot(a, b), -1.0, 1.0)))
    return Rotation.from_rotvec(angle * axis).as_matrix()


def _affine_rotation_around_centre(R3: np.ndarray, centre: np.ndarray) -> np.ndarray:
    """4×4 affine that rotates by ``R3`` around ``centre`` in the source volume."""
    A1 = np.eye(4)
    A1[:3, 3] = -centre
    A2 = np.eye(4)
    A2[:3, :3] = R3
    A3 = np.eye(4)
    A3[:3, 3] = centre
    return A3 @ A2 @ A1


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def align_volume_to_template_by_fissure(
    moving_volume: np.ndarray | Path | str,
    template_volume: np.ndarray | Path | str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Rigidly align ``moving_volume`` to ``template_volume`` using midline-fissure
    plane detection on both.

    Parameters
    ----------
    moving_volume
        Either a 3D numpy array (PIR-oriented) or a path to a NIfTI file.
    template_volume
        Same as ``moving_volume``. The CCF template in PIR orientation.

    Returns
    -------
    A : ndarray, shape (4, 4)
        Affine rotation that aligns the moving volume to the template.
    aligned_moving : ndarray
        ``moving_volume`` after applying ``A`` via
        ``scipy.ndimage.affine_transform`` (same shape as input).
    info : dict
        Diagnostic info:
        - ``moving_normal``, ``template_normal`` : plane normal vectors
        - ``moving_points``, ``template_points`` : detected fissure points
        - ``rotation_angle_deg`` : total rotation magnitude in degrees
        - ``moving_shape`` : input shape
    """
    mdata = _load_volume(moving_volume)
    fdata = _load_volume(template_volume)

    moving_pts = _collect_fissure_points(mdata)
    template_pts = _collect_fissure_points(fdata)

    if moving_pts.shape[0] < 3:
        raise ValueError(
            f"axis alignment failed: only {moving_pts.shape[0]} fissure point(s) detected in moving image. "
            "May lack a visible midline fissure or thresholds need tuning."
        )
    if template_pts.shape[0] < 3:
        raise ValueError(
            f"axis alignment failed: only {template_pts.shape[0]} fissure point(s) detected in template image."
        )

    moving_normal = _svd_plane_normal(moving_pts)
    template_normal = _svd_plane_normal(template_pts)

    R3 = _rotation_matrix_between(moving_normal, template_normal)
    centre = np.asarray(mdata.shape, dtype=np.float64) / 2.0
    A = _affine_rotation_around_centre(R3, centre)

    aligned = affine_transform(mdata, np.linalg.inv(A), output_shape=mdata.shape, order=1)
    aligned[aligned < 0] = 0

    cos_angle = (np.trace(R3) - 1.0) / 2.0
    cos_angle = float(np.clip(cos_angle, -1.0, 1.0))
    rotation_angle_deg = float(np.degrees(np.arccos(cos_angle)))

    info = {
        "moving_normal": moving_normal,
        "template_normal": template_normal,
        "moving_points": moving_pts,
        "template_points": template_pts,
        "rotation_angle_deg": rotation_angle_deg,
        "moving_shape": tuple(int(s) for s in mdata.shape),
    }
    log.info(
        "axis align: %d moving pts, %d template pts, rotation = %.2f deg",
        moving_pts.shape[0],
        template_pts.shape[0],
        rotation_angle_deg,
    )
    return A, aligned, info


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _load_volume(source: np.ndarray | Path | str) -> np.ndarray:
    if isinstance(source, np.ndarray):
        return np.ascontiguousarray(source)
    img = nib.load(str(source))
    return np.ascontiguousarray(np.asarray(img.dataobj))


def save_axis_align_matrix(path: Path | str, matrix: np.ndarray) -> Path:
    """Save an axis-alignment affine as ``.npz`` with key ``arr`` (matching Xu Lab's
    ``_save_npz`` / ``_load_npz`` convention). Returns the final path with ``.npz``.
    """
    base = Path(path).with_suffix("")
    out = base.with_suffix(".npz")
    np.savez_compressed(str(out), arr=np.asarray(matrix))
    return out


def load_axis_align_matrix(path: Path | str) -> np.ndarray | None:
    """Load an axis-align affine from ``.npy`` (preferred) or ``.npz`` (key=arr).
    Returns ``None`` if neither file exists.
    """
    base = Path(path).with_suffix("")
    npy = base.with_suffix(".npy")
    npz = base.with_suffix(".npz")
    if npy.exists():
        return np.load(str(npy))
    if npz.exists():
        return np.load(str(npz))["arr"]
    return None
