"""Unit tests for the annotation sampling modes introduced by the
2026-04-22 root-cause fix (see docs/superpowers/plans/
2026-04-22-annotation-granularity-root-cause.md).

Covers three things:

1. ``_compute_atlas_z_indices`` correctly distributes *N* sample slices
   across the CCF atlas-Z range the input volume occupies (fixes the
   old ``atlas_z = z_offset + i`` 1:1 bug that only worked when input
   and atlas Z-spacings happened to match).

2. ``_direct_z_mapping_fallback`` preserves the full set of unique
   region IDs in the CCF annotation — in contrast to the 3D-reslice
   path which nearest-neighbor downsamples and silently drops thin
   leaf regions.

3. The sidecar helpers (``write_annotation_sidecar`` /
   ``read_annotation_sampling_mode`` /
   ``annotation_prewarped_for_mode``) round-trip correctly so
   downstream ``liquify_3d_finalize`` picks the right
   ``prewarped_label`` flag.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from project.scripts.annotation_sidecar import (
    annotation_prewarped_for_mode,
    read_annotation_sampling_mode,
    write_annotation_sidecar,
)
from project.scripts.whole_brain_3d import (
    _compute_atlas_z_indices,
    _direct_z_mapping_fallback,
)


# ---------------------------------------------------------------------------
# _compute_atlas_z_indices
# ---------------------------------------------------------------------------


def test_compute_atlas_z_indices_identity_mapping_when_spacings_match():
    """When input Z-count == atlas-Z span, mapping is effectively 1:1."""
    # Input has 11 slices; they occupy atlas Z [100, 110] (11 atlas slices)
    nonzero = np.arange(100, 111, dtype=np.int32)
    out = _compute_atlas_z_indices(num_input_z=11, num_atlas_z=528, nonzero_atlas_z=nonzero)
    assert out.tolist() == list(range(100, 111))


def test_compute_atlas_z_indices_scales_across_coarser_input_spacing():
    """111 sample slices across a 555-atlas-slice span should stride ~5."""
    nonzero = np.arange(0, 555, dtype=np.int32)
    out = _compute_atlas_z_indices(num_input_z=111, num_atlas_z=555, nonzero_atlas_z=nonzero)
    assert len(out) == 111
    assert int(out[0]) == 0
    assert int(out[-1]) == 554
    # Stride should be roughly (555-1)/110 ≈ 5
    diffs = np.diff(out)
    assert diffs.min() >= 4
    assert diffs.max() <= 6


def test_compute_atlas_z_indices_single_slice():
    nonzero = np.arange(200, 400, dtype=np.int32)
    out = _compute_atlas_z_indices(num_input_z=1, num_atlas_z=528, nonzero_atlas_z=nonzero)
    assert out.tolist() == [200]


def test_compute_atlas_z_indices_clips_to_atlas_bounds():
    """Rounding must never produce an index ≥ num_atlas_z."""
    nonzero = np.array([0, 527], dtype=np.int32)
    out = _compute_atlas_z_indices(num_input_z=5, num_atlas_z=528, nonzero_atlas_z=nonzero)
    assert out.max() < 528
    assert out.min() >= 0


def test_compute_atlas_z_indices_falls_back_to_full_span_when_no_ants_hint():
    """No ANTs reference → assume input spans whole atlas Z."""
    out = _compute_atlas_z_indices(num_input_z=5, num_atlas_z=10, nonzero_atlas_z=None)
    assert out[0] == 0
    assert out[-1] == 9


# ---------------------------------------------------------------------------
# _direct_z_mapping_fallback — leaf-region preservation
# ---------------------------------------------------------------------------


def _write_volume(path: Path, arr: np.ndarray) -> Path:
    img = nib.Nifti1Image(arr.astype(np.int32), affine=np.eye(4))
    nib.save(img, str(path))
    return path


def test_direct_z_mapping_preserves_leaf_regions(tmp_path: Path):
    """Every CCF-Z that the sampler lands on contributes its unique label.

    Build a synthetic 20-slice CCF annotation with a *different* region ID on
    each slice. Sample 5 slices proportionally across all 20. The resulting
    volume must contain exactly those 5 region IDs — the 3D-reslice path
    would collapse neighbouring Z onto single labels and drop most.
    """
    atlas_z, atlas_y, atlas_x = 20, 8, 10
    ann = np.zeros((atlas_z, atlas_y, atlas_x), dtype=np.int32)
    for z in range(atlas_z):
        ann[z, :, :] = 100 + z  # unique per slice

    ann_path = _write_volume(tmp_path / "annotation.nii.gz", ann)

    # Reference volume has 5 sample slices spanning the full atlas Z
    ref = np.ones((5, atlas_y, atlas_x), dtype=np.int32)
    ref_path = _write_volume(tmp_path / "reference.nii.gz", ref)

    # ANTs result volume covering the full atlas Z range (all slices non-zero)
    ants_result = np.ones((atlas_z, atlas_y, atlas_x), dtype=np.float32)
    ants_path = _write_volume(tmp_path / "ants_result.nii.gz", ants_result)

    out_path = tmp_path / "annotation_registered.nii.gz"
    ok = _direct_z_mapping_fallback(
        annotation_path=ann_path,
        reference_volume_path=ref_path,
        output_path=out_path,
        ants_result_path=ants_path,
    )
    assert ok is True

    out_vol = np.asarray(nib.load(str(out_path)).dataobj, dtype=np.int32)
    # Shape: (num_input_z=5, atlas_y=8, atlas_x=10) — keeps CCF native Y×X
    assert out_vol.shape == (5, atlas_y, atlas_x)

    # Unique non-zero labels: one per sample slice, all distinct
    unique_ids = set(np.unique(out_vol).tolist()) - {0}
    assert len(unique_ids) == 5
    # All IDs came from the input atlas, spaced across the Z range
    assert all(100 <= rid <= 119 for rid in unique_ids)


def test_direct_z_mapping_preserves_y_x_resolution(tmp_path: Path):
    """Output Y×X must equal CCF native Y×X, not the input reference shape."""
    atlas_shape = (10, 100, 120)
    ann = np.ones(atlas_shape, dtype=np.int32) * 42
    ann_path = _write_volume(tmp_path / "annotation.nii.gz", ann)

    # Reference has much smaller Y×X (e.g. a heavily downsampled sample volume)
    ref = np.ones((5, 20, 25), dtype=np.int32)
    ref_path = _write_volume(tmp_path / "reference.nii.gz", ref)

    ants_result = np.ones(atlas_shape, dtype=np.float32)
    ants_path = _write_volume(tmp_path / "ants.nii.gz", ants_result)

    out_path = tmp_path / "out.nii.gz"
    _direct_z_mapping_fallback(
        annotation_path=ann_path,
        reference_volume_path=ref_path,
        output_path=out_path,
        ants_result_path=ants_path,
    )
    out_vol = np.asarray(nib.load(str(out_path)).dataobj, dtype=np.int32)
    # Y×X inherited from CCF, not from the (small) reference volume
    assert out_vol.shape == (5, atlas_shape[1], atlas_shape[2])


# ---------------------------------------------------------------------------
# sidecar round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ["3d_reslice", "per_slice_native"])
def test_sidecar_write_and_read_round_trip(tmp_path: Path, mode: str):
    ann = tmp_path / "annotation_registered.nii.gz"
    ann.write_bytes(b"not-real-nifti")  # sidecar doesn't inspect contents
    write_annotation_sidecar(ann, mode)
    assert read_annotation_sampling_mode(ann) == mode


def test_sidecar_missing_defaults_to_3d_reslice(tmp_path: Path):
    ann = tmp_path / "missing_sidecar.nii.gz"
    assert read_annotation_sampling_mode(ann) == "3d_reslice"


def test_sidecar_unknown_mode_falls_back_to_default(tmp_path: Path):
    ann = tmp_path / "annotation.nii.gz"
    side = tmp_path / "annotation.meta.json"
    side.write_text('{"annotation_sampling_mode": "bogus"}', encoding="utf-8")
    assert read_annotation_sampling_mode(ann) == "3d_reslice"


def test_annotation_prewarped_flag_matches_mode():
    assert annotation_prewarped_for_mode("3d_reslice") is True
    assert annotation_prewarped_for_mode("per_slice_native") is False
    # Unknown modes → assume prewarped (matches default 3d_reslice semantics)
    assert annotation_prewarped_for_mode("bogus") is True
