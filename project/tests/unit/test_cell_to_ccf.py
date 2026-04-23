"""Unit tests for cell_to_ccf — the Xu Lab-aligned cell-points-to-CCF path.

These tests cover the pure-math helpers (voxel↔physical conversion, pixel
→ volume voxel, annotation voxel lookup) without requiring antspy runtime.
The full transform (``transform_points_sample_to_ccf`` / ``map_cells_via_ccf_transform``)
that calls ``ants.apply_transforms_to_points`` is covered by a separate
integration test that needs antspy installed.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest

from project.scripts.cell_to_ccf import (
    _affine_to_spacing_origin_direction,
    lookup_region_ids_from_ccf_voxels,
    physical_to_voxel,
    sample_pixel_to_volume_voxel,
    voxel_to_physical,
)


# ---------------------------------------------------------------------------
# affine decomposition
# ---------------------------------------------------------------------------


def test_affine_decomposition_identity():
    A = np.eye(4)
    spacing, origin, direction = _affine_to_spacing_origin_direction(A)
    np.testing.assert_allclose(spacing, [1.0, 1.0, 1.0])
    np.testing.assert_allclose(origin, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(direction, np.eye(3))


def test_affine_decomposition_diagonal_spacing():
    """A pure-diagonal affine should give diagonal spacing + identity direction."""
    A = np.diag([0.025, 0.02, 0.02, 1.0])  # Brainfast sample input_volume-ish
    spacing, origin, direction = _affine_to_spacing_origin_direction(A)
    np.testing.assert_allclose(spacing, [0.025, 0.02, 0.02])
    np.testing.assert_allclose(origin, [0.0, 0.0, 0.0])
    np.testing.assert_allclose(direction, np.eye(3))


def test_affine_decomposition_nonidentity_direction():
    """Allen CCF's 4×4 has non-trivial direction cosines; decomposition must preserve them."""
    # Matches project/configs/allen_ref_cache/average_template_25.nii.gz layout.
    A = np.array(
        [
            [0.0, 0.0, 0.025, -5.695],
            [-0.025, 0.0, 0.0, 5.35],
            [0.0, -0.025, 0.0, 5.22],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    spacing, origin, direction = _affine_to_spacing_origin_direction(A)
    np.testing.assert_allclose(spacing, [0.025, 0.025, 0.025])
    np.testing.assert_allclose(origin, [-5.695, 5.35, 5.22])
    expected_dir = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    np.testing.assert_allclose(direction, expected_dir)


# ---------------------------------------------------------------------------
# voxel ↔ physical round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "affine",
    [
        np.eye(4),
        np.diag([0.025, 0.02, 0.02, 1.0]),
        np.array(
            [
                [0.0, 0.0, 0.025, -5.695],
                [-0.025, 0.0, 0.0, 5.35],
                [0.0, -0.025, 0.0, 5.22],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ),
    ],
)
def test_voxel_physical_round_trip(affine):
    spacing, origin, direction = _affine_to_spacing_origin_direction(affine)
    rng = np.random.default_rng(42)
    voxels = rng.uniform(low=0, high=300, size=(50, 3))
    physical = voxel_to_physical(voxels, spacing, origin, direction)
    roundtrip = physical_to_voxel(physical, spacing, origin, direction)
    np.testing.assert_allclose(roundtrip, voxels, atol=1e-9)


# ---------------------------------------------------------------------------
# sample_pixel_to_volume_voxel
# ---------------------------------------------------------------------------


def test_sample_pixel_to_volume_voxel_25um_vol_5um_pixel():
    """pixel=5µm, volume spacing=25µm → 5x downsample on XY."""
    # zooms (mm): (0.025, 0.025, 0.025) — 25µm isotropic volume
    zooms = (0.025, 0.025, 0.025)
    out = sample_pixel_to_volume_voxel(
        cell_x_pixel=np.array([0.0, 100.0, 1000.0]),
        cell_y_pixel=np.array([0.0, 50.0, 250.0]),
        cell_slice_id=np.array([0, 5, 10]),
        pixel_size_um=5.0,
        sample_volume_zooms_mm=zooms,
    )
    np.testing.assert_allclose(out[:, 0], [0, 5, 10])  # z = slice_id
    np.testing.assert_allclose(out[:, 1], [0, 10, 50])  # y / 5
    np.testing.assert_allclose(out[:, 2], [0, 20, 200])  # x / 5


def test_sample_pixel_to_volume_voxel_brainfast_sample_35_spacing():
    """Brainfast sample 35: pixel=5µm, volume_xy=20µm → 4x downsample, sample 35 demo."""
    zooms = (0.025, 0.02, 0.02)  # 25µm Z, 20µm XY
    out = sample_pixel_to_volume_voxel(
        cell_x_pixel=np.array([1000.0, 2000.0]),
        cell_y_pixel=np.array([800.0, 1600.0]),
        cell_slice_id=np.array([50, 100]),
        pixel_size_um=5.0,
        sample_volume_zooms_mm=zooms,
    )
    np.testing.assert_allclose(out[:, 0], [50, 100])
    np.testing.assert_allclose(out[:, 1], [200, 400])  # 800/4, 1600/4
    np.testing.assert_allclose(out[:, 2], [250, 500])  # 1000/4, 2000/4


def test_sample_pixel_to_volume_voxel_rejects_bad_spacing():
    with pytest.raises(ValueError, match="at least 3 entries"):
        sample_pixel_to_volume_voxel(
            cell_x_pixel=np.array([0.0]),
            cell_y_pixel=np.array([0.0]),
            cell_slice_id=np.array([0]),
            pixel_size_um=5.0,
            sample_volume_zooms_mm=(0.025, 0.025),  # too short
        )


# ---------------------------------------------------------------------------
# lookup_region_ids_from_ccf_voxels
# ---------------------------------------------------------------------------


def _write_annotation(tmp_path: Path, arr: np.ndarray) -> Path:
    p = tmp_path / "annotation.nii.gz"
    img = nib.Nifti1Image(arr.astype(np.int32), affine=np.eye(4))
    nib.save(img, str(p))
    return p


def test_lookup_region_ids_basic(tmp_path: Path):
    ann = np.zeros((10, 10, 10), dtype=np.int32)
    ann[5, 5, 5] = 42
    ann[1, 2, 3] = 7
    ann[9, 9, 9] = 99
    path = _write_annotation(tmp_path, ann)

    pts = np.array([[5, 5, 5], [1, 2, 3], [9, 9, 9], [0, 0, 0]], dtype=np.float64)
    region_ids, oob = lookup_region_ids_from_ccf_voxels(pts, path)
    np.testing.assert_array_equal(region_ids, [42, 7, 99, 0])
    assert oob == 0


def test_lookup_region_ids_out_of_bounds(tmp_path: Path):
    ann = np.ones((5, 5, 5), dtype=np.int32) * 3
    path = _write_annotation(tmp_path, ann)

    pts = np.array(
        [
            [-1, 0, 0],  # z < 0
            [5, 0, 0],   # z == nz (out)
            [0, -1, 0],  # y < 0
            [0, 0, 5],   # x == nx (out)
            [0, 0, 0],   # in-bounds
        ],
        dtype=np.float64,
    )
    region_ids, oob = lookup_region_ids_from_ccf_voxels(pts, path)
    np.testing.assert_array_equal(region_ids, [0, 0, 0, 0, 3])
    assert oob == 4


def test_lookup_region_ids_fractional_rounds_to_nearest(tmp_path: Path):
    ann = np.arange(27, dtype=np.int32).reshape(3, 3, 3)
    path = _write_annotation(tmp_path, ann)

    # 1.4 → 1, 1.6 → 2; 0.5 rounds to even (0); 2.5 rounds to even (2)
    pts = np.array([[1.4, 1.4, 1.4], [1.6, 1.6, 1.6], [0.4, 0.4, 0.4]], dtype=np.float64)
    region_ids, oob = lookup_region_ids_from_ccf_voxels(pts, path)
    # idx_from_zyx(i, j, k) = i*9 + j*3 + k
    assert region_ids.tolist() == [1 * 9 + 1 * 3 + 1, 2 * 9 + 2 * 3 + 2, 0]
    assert oob == 0


# ---------------------------------------------------------------------------
# transform_points_sample_to_ccf with identity transforms (no ANTs runtime)
# ---------------------------------------------------------------------------

# The full transform is covered by integration tests that require antspy.
# Here we just verify the voxel→physical→voxel round-trip when the two affines
# are identical and there are no ANTs transforms to apply.


def test_round_trip_with_matching_affines(tmp_path):
    """When sample and CCF affines match and we skip ANTs, round-trip is identity.

    This is a *sanity check* of the voxel↔physical conversion math — the full
    transform with real ANTs transforms is validated end-to-end on sample 35
    artifacts separately (integration test / manual spike).
    """
    spacing, origin, direction = _affine_to_spacing_origin_direction(
        np.diag([0.025, 0.025, 0.025, 1.0])
    )
    voxels = np.array([[50, 100, 150], [10, 20, 30]], dtype=np.float64)
    physical = voxel_to_physical(voxels, spacing, origin, direction)
    recovered = physical_to_voxel(physical, spacing, origin, direction)
    np.testing.assert_allclose(recovered, voxels)


# ---------------------------------------------------------------------------
# map_cells_via_ccf_transform — empty-input short-circuit
# ---------------------------------------------------------------------------


def test_map_cells_empty_dataframe(tmp_path):
    """Empty input should short-circuit with the expected output schema."""
    from project.scripts.cell_to_ccf import map_cells_via_ccf_transform

    sample_vol = tmp_path / "sample.nii.gz"
    nib.save(
        nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.int32), affine=np.eye(4)),
        str(sample_vol),
    )
    ann = tmp_path / "ann.nii.gz"
    nib.save(
        nib.Nifti1Image(np.zeros((10, 10, 10), dtype=np.int32), affine=np.eye(4)),
        str(ann),
    )

    empty = pd.DataFrame(columns=["x", "y", "slice_id"])
    out = map_cells_via_ccf_transform(
        empty,
        sample_volume_path=sample_vol,
        ccf_annotation_path=ann,
        inverse_transforms=[],
        pixel_size_um=5.0,
    )
    assert len(out) == 0
    for col in ("ccf_z_voxel", "ccf_y_voxel", "ccf_x_voxel", "region_id", "mapping_status"):
        assert col in out.columns
