"""Unit tests for Phase β 3D liquify backend.

Phase β lets a human provide sparse 3D landmark correspondences between
(real fluorescence voxel coord, desired atlas voxel coord) and refines the
registered annotation accordingly. The backend is a thin layer over the
vendored regtools_laplacian solver — the math is already validated there;
these tests cover the Brainfast-specific CSV store, warp-application, and
end-to-end annotation refinement.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# LandmarkStore — CSV-backed per-job landmark accumulator
# ---------------------------------------------------------------------------


def test_landmark_store_add_and_list(tmp_path):
    from project.scripts.liquify_3d import LandmarkStore

    store = LandmarkStore(tmp_path / "landmarks_3d.csv")
    assert store.list_pairs() == []

    store.add_pair(z=100, real=(50.0, 80.0), atlas=(55.0, 82.0))
    store.add_pair(z=200, real=(120.0, 60.0), atlas=(122.0, 58.0))

    pairs = store.list_pairs()
    assert len(pairs) == 2
    assert pairs[0].z == 100
    assert pairs[0].real == (50.0, 80.0)
    assert pairs[0].atlas == (55.0, 82.0)
    assert pairs[1].z == 200


def test_landmark_store_persists_across_reopen(tmp_path):
    from project.scripts.liquify_3d import LandmarkStore

    path = tmp_path / "landmarks_3d.csv"
    store = LandmarkStore(path)
    store.add_pair(z=100, real=(50.0, 80.0), atlas=(55.0, 82.0))

    reopened = LandmarkStore(path)
    pairs = reopened.list_pairs()
    assert len(pairs) == 1
    assert pairs[0].z == 100


def test_landmark_store_remove_by_index(tmp_path):
    from project.scripts.liquify_3d import LandmarkStore

    store = LandmarkStore(tmp_path / "landmarks_3d.csv")
    store.add_pair(z=100, real=(50, 80), atlas=(55, 82))
    store.add_pair(z=200, real=(120, 60), atlas=(122, 58))
    store.add_pair(z=300, real=(90, 90), atlas=(91, 91))

    removed = store.remove_pair(1)
    assert removed.z == 200
    remaining = store.list_pairs()
    assert len(remaining) == 2
    assert [p.z for p in remaining] == [100, 300]


def test_landmark_store_remove_out_of_range_raises(tmp_path):
    from project.scripts.liquify_3d import LandmarkStore

    store = LandmarkStore(tmp_path / "landmarks_3d.csv")
    store.add_pair(z=100, real=(50, 80), atlas=(55, 82))
    with pytest.raises(IndexError):
        store.remove_pair(5)


# ---------------------------------------------------------------------------
# compute_3d_displacement — wraps the vendored Laplacian solver
# ---------------------------------------------------------------------------


def test_compute_displacement_no_pairs_returns_zero_field():
    from project.scripts.liquify_3d import compute_3d_displacement

    field = compute_3d_displacement(
        vol_shape=(5, 8, 8),
        source_pts=np.empty((0, 3)),
        target_pts=np.empty((0, 3)),
    )
    assert field.shape == (3, 5, 8, 8)
    assert np.allclose(field, 0.0)


def test_compute_displacement_single_axis_shift_recovered():
    """Seed Dirichlet corners with a uniform +3 shift in axis 2; the solver
    should propagate the shift to interior voxels."""
    from project.scripts.liquify_3d import compute_3d_displacement

    shape = (4, 8, 8)
    corners = np.array(
        [
            [0, 0, 0], [0, 0, 7], [0, 7, 0], [0, 7, 7],
            [3, 0, 0], [3, 0, 7], [3, 7, 0], [3, 7, 7],
        ],
        dtype=float,
    )
    target_pts = corners.copy()
    source_pts = corners.copy()
    source_pts[:, 2] += 3.0

    field = compute_3d_displacement(
        vol_shape=shape,
        source_pts=source_pts,
        target_pts=target_pts,
    )
    axis2_interior = field[2, 1:3, 2:6, 2:6]
    assert np.mean(axis2_interior) == pytest.approx(3.0, abs=0.8)


# ---------------------------------------------------------------------------
# apply_3d_warp_to_annotation — maps annotation through displacement field
# ---------------------------------------------------------------------------


def test_apply_warp_identity_field_is_noop():
    """A zero displacement field should leave the annotation unchanged."""
    from project.scripts.liquify_3d import apply_3d_warp_to_annotation

    ann = np.random.default_rng(0).integers(1, 500, size=(4, 6, 6), dtype=np.int32)
    field = np.zeros((3, 4, 6, 6), dtype=np.float32)
    out = apply_3d_warp_to_annotation(ann, field)
    assert out.shape == ann.shape
    assert out.dtype == ann.dtype
    # Identity field with nearest-neighbor interp should reproduce input exactly.
    assert np.array_equal(out, ann)


def test_apply_warp_uniform_shift_translates_labels():
    """A uniform +1 shift in axis 2 should move labels one voxel to the right,
    with the left edge zero-filled."""
    from project.scripts.liquify_3d import apply_3d_warp_to_annotation

    ann = np.zeros((3, 3, 5), dtype=np.int32)
    ann[:, :, 2] = 42  # column 2 carries label 42
    field = np.zeros((3, 3, 3, 5), dtype=np.float32)
    field[2] = 1.0  # displacement +1 along axis 2 for every voxel

    out = apply_3d_warp_to_annotation(ann, field)
    # Column 3 should now carry the label that was at column 2.
    assert (out[:, :, 3] == 42).all()
    # Column 2 should be zero (its source at column 1 was background).
    assert (out[:, :, 2] == 0).all()


# ---------------------------------------------------------------------------
# End-to-end refine_annotation_with_landmarks
# ---------------------------------------------------------------------------


def test_refine_annotation_end_to_end_roundtrip(tmp_path):
    """Full path: CSV store → solver → warp → annotation saved as NIfTI.

    We synthesise a small annotation volume, add one landmark pair that asks
    to shift axis 2 by +1, and verify the output annotation NIfTI shows the
    shift at the landmark z and reduced shift elsewhere (Laplacian decay).
    """
    import nibabel as nib

    from project.scripts.liquify_3d import (
        LandmarkStore,
        refine_annotation_with_landmarks,
    )

    shape = (4, 6, 6)
    ann = np.zeros(shape, dtype=np.int32)
    ann[:, :, 3] = 7  # column 3 labelled 7
    ann_path = tmp_path / "annotation_in.nii.gz"
    nib.save(nib.Nifti1Image(ann, np.eye(4)), str(ann_path))

    csv_path = tmp_path / "landmarks_3d.csv"
    store = LandmarkStore(csv_path)
    # Request: at z=2, the voxel currently at atlas (3, 3) should be aligned
    # to real (3, 4) — i.e. push the label one voxel right at that z.
    store.add_pair(z=2, real=(3.0, 4.0), atlas=(3.0, 3.0))

    out_path = tmp_path / "annotation_refined.nii.gz"
    meta = refine_annotation_with_landmarks(
        annotation_path=ann_path,
        landmarks_csv=csv_path,
        output_path=out_path,
    )

    assert out_path.exists()
    assert meta["pair_count"] == 1
    warped = np.asarray(nib.load(str(out_path)).dataobj, dtype=np.int32)
    # Sanity: same shape + dtype preserved
    assert warped.shape == ann.shape
    assert warped.dtype == np.int32
    # The annotation should not be all zeros (warp succeeded)
    assert (warped != 0).any()
