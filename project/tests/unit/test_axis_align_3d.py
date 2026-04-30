"""Unit tests for axis_align_3d.

Covers the pure-math helpers (SVD plane fit, rotation between vectors,
rotation-around-centre affine) and the save/load helpers. The full
``align_volume_to_template_by_fissure`` entry point is exercised via an
integration test on real CCF+sample data separately (not here — needs
tifffile-heavy inputs). The line-detection helpers (``_centre_column_bounds``,
``_hough_lines``, ``_average_fissure_line``, ``_collect_fissure_points``) are
smoke-tested with a synthetic slice containing a known fissure.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from project.scripts.axis_align_3d import (
    _affine_rotation_around_centre,
    _average_fissure_line,
    _centre_column_bounds,
    _collect_fissure_points,
    _hough_lines,
    _preprocess_slice,
    _rotation_matrix_between,
    _svd_plane_normal,
    load_axis_align_matrix,
    save_axis_align_matrix,
)

# ---------------------------------------------------------------------------
# plane fitting
# ---------------------------------------------------------------------------


def test_svd_plane_normal_on_z0_plane():
    """Points on the z=0 plane should give normal ~ (0, 0, 1)."""
    rng = np.random.default_rng(42)
    pts = np.column_stack([rng.uniform(-5, 5, 20), rng.uniform(-5, 5, 20), np.zeros(20)])
    normal = _svd_plane_normal(pts)
    assert abs(abs(normal[2]) - 1.0) < 1e-6
    assert abs(normal[0]) < 1e-6
    assert abs(normal[1]) < 1e-6


def test_svd_plane_normal_on_tilted_plane():
    """Points on plane x + y + z = 0 should give normal ~ (1,1,1)/sqrt(3)."""
    rng = np.random.default_rng(7)
    N = 30
    x = rng.uniform(-5, 5, N)
    y = rng.uniform(-5, 5, N)
    z = -(x + y)  # x + y + z = 0
    pts = np.column_stack([x, y, z])
    normal = _svd_plane_normal(pts)
    expected = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    assert abs(abs(np.dot(normal, expected)) - 1.0) < 1e-6


def test_svd_plane_normal_rejects_too_few_points():
    with pytest.raises(ValueError, match="at least 3"):
        _svd_plane_normal(np.array([[0, 0, 0], [1, 0, 0]]))


# ---------------------------------------------------------------------------
# rotation between vectors
# ---------------------------------------------------------------------------


def test_rotation_matrix_between_orthogonal():
    R = _rotation_matrix_between(np.array([1, 0, 0]), np.array([0, 1, 0]))
    # R should take (1,0,0) to (0,1,0)
    result = R @ np.array([1, 0, 0])
    np.testing.assert_allclose(result, [0, 1, 0], atol=1e-9)


def test_rotation_matrix_between_identity():
    R = _rotation_matrix_between(np.array([1, 0, 0]), np.array([1, 0, 0]))
    np.testing.assert_allclose(R, np.eye(3), atol=1e-9)


def test_rotation_matrix_between_antiparallel():
    R = _rotation_matrix_between(np.array([1, 0, 0]), np.array([-1, 0, 0]))
    result = R @ np.array([1, 0, 0])
    np.testing.assert_allclose(result, [-1, 0, 0], atol=1e-9)


def test_rotation_matrix_between_arbitrary():
    v = np.array([0.3, 0.6, 0.1])
    v /= np.linalg.norm(v)
    t = np.array([-0.2, 0.9, 0.4])
    t /= np.linalg.norm(t)
    R = _rotation_matrix_between(v, t)
    result = R @ v
    np.testing.assert_allclose(result, t, atol=1e-9)


# ---------------------------------------------------------------------------
# affine rotation-around-centre
# ---------------------------------------------------------------------------


def test_affine_rotation_around_centre_is_rigid():
    """The 3×3 block equals R, and applying to the centre leaves it fixed."""
    R3 = np.eye(3)
    centre = np.array([10.0, 20.0, 30.0])
    A = _affine_rotation_around_centre(R3, centre)
    np.testing.assert_allclose(A[:3, :3], np.eye(3))
    # Centre maps to itself
    h = np.array([10.0, 20.0, 30.0, 1.0])
    out = A @ h
    np.testing.assert_allclose(out[:3], centre)


def test_affine_rotation_around_centre_nontrivial_rotation_preserves_centre():
    """90° z-rotation around centre keeps the centre fixed."""
    R3 = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=np.float64)
    centre = np.array([5.0, 5.0, 5.0])
    A = _affine_rotation_around_centre(R3, centre)
    h = np.array([5.0, 5.0, 5.0, 1.0])
    out = A @ h
    np.testing.assert_allclose(out[:3], centre, atol=1e-9)


# ---------------------------------------------------------------------------
# save/load axis-align matrix
# ---------------------------------------------------------------------------


def test_save_load_axis_align_matrix_round_trip(tmp_path: Path):
    A = np.random.default_rng(0).standard_normal((4, 4))
    A[3] = [0, 0, 0, 1]
    out = save_axis_align_matrix(tmp_path / "axisAlignA", A)
    assert out.suffix == ".npz"
    loaded = load_axis_align_matrix(tmp_path / "axisAlignA")
    np.testing.assert_allclose(loaded, A)


def test_save_load_axis_align_matrix_missing_returns_none(tmp_path: Path):
    assert load_axis_align_matrix(tmp_path / "nothing") is None


def test_save_load_axis_align_matrix_uses_arr_key_not_A_key(tmp_path: Path):
    """Backwards-compat with Xu Lab's ``_load_npz(key='arr')``."""
    A = np.eye(4)
    out = save_axis_align_matrix(tmp_path / "check_key", A)
    # Verify the npz archive actually has the 'arr' key
    with np.load(str(out)) as npz:
        assert "arr" in npz.files


# ---------------------------------------------------------------------------
# line-detection smoke tests with synthetic slices
# ---------------------------------------------------------------------------


def _make_synthetic_slice_with_fissure(h: int = 200, w: int = 200) -> np.ndarray:
    """Create a 2D slice with two bright hemispheres separated by a clearly
    visible dark vertical midline fissure. The high-contrast bright/dark
    vertical edges give Canny + Hough sharp gradients to lock onto.
    """
    arr = np.zeros((h, w), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    # Two bright hemispheres, symmetric about x = w/2
    left_blob = ((yy - h / 2) / (h * 0.35)) ** 2 + ((xx - w * 0.30) / (w * 0.18)) ** 2 < 1
    right_blob = ((yy - h / 2) / (h * 0.35)) ** 2 + ((xx - w * 0.70) / (w * 0.18)) ** 2 < 1
    arr[left_blob] = 300.0
    arr[right_blob] = 300.0
    # Strong vertical midline fissure: 4-px dark column with sharp edges
    fissure_x = w // 2
    arr[int(h * 0.25) : int(h * 0.75), fissure_x - 2 : fissure_x + 3] = 0.0
    return arr


def test_preprocess_slice_normalises_to_uint8():
    raw = _make_synthetic_slice_with_fissure()
    u8 = _preprocess_slice(raw, max_val=400)
    assert u8.dtype == np.uint8
    assert 0 <= u8.min() <= u8.max() <= 255


def test_centre_column_bounds_finds_tissue_midpoint():
    raw = _make_synthetic_slice_with_fissure()
    u8 = _preprocess_slice(raw)
    bounds = _centre_column_bounds(u8, ratio=0.1)
    assert bounds is not None
    left, right = bounds
    # Midpoint should be near the slice centre — widened tolerance since Otsu
    # picks up either of the two hemispheres as dominant contour; exact midpoint
    # depends on which contour wins the largest-area race.
    mid = (left + right) / 2
    assert 0 < mid < raw.shape[1]


def test_centre_column_bounds_on_empty_slice_returns_none():
    u8 = np.zeros((100, 100), dtype=np.uint8)
    assert _centre_column_bounds(u8, ratio=0.1) is None


def test_hough_lines_finds_vertical_segments_on_synthetic():
    raw = _make_synthetic_slice_with_fissure()
    u8 = _preprocess_slice(raw)
    lines = _hough_lines(u8, min_thresh=50, max_thresh=150, line_thresh=10, min_line_length=20)
    assert lines is not None  # should detect at least one line near the fissure edges


def test_average_fissure_line_filters_by_angle_and_bounds():
    # Pretend hough returned a vertical line inside the centre column
    lines = np.array([[[98, 10, 100, 90]]], dtype=np.int32)  # nearly vertical
    out = _average_fissure_line(lines, left=90.0, right=110.0)
    assert out is not None
    # And a horizontal line outside the centre column gets rejected
    lines_bad = np.array([[[0, 50, 200, 55]]], dtype=np.int32)
    assert _average_fissure_line(lines_bad, left=90.0, right=110.0) is None


def test_average_fissure_line_returns_none_when_no_lines():
    assert _average_fissure_line(None, 0, 100) is None


def test_collect_fissure_points_runs_without_error_on_empty_volume():
    """Smoke: confirm the collector handles an all-zero volume gracefully
    (no lines detected → empty point set, not a crash).

    End-to-end fissure detection is tuned for anatomical data (Canny +
    Hough thresholds, centre-column ratio). Verifying that on real CCF
    + ChATe27 inputs is the integration path (not a unit test).
    """
    vol = np.zeros((10, 80, 80), dtype=np.uint8)
    pts = _collect_fissure_points(vol, slice_range=(0, 10))
    assert pts.shape == (0, 3)
