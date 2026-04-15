from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from project.scripts.whole_brain_3d import (
    _extrapolate_annotation_to_tissue,
    run_whole_brain_3d,
)


def test_run_whole_brain_3d_emits_expected_stage_sequence(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    outputs_dir = tmp_path / "outputs"
    input_dir.mkdir()
    outputs_dir.mkdir()

    # Create dummy slice files so merged_slice_paths auto-population works
    for i in range(3):
        (input_dir / f"z{i:04d}.tif").write_bytes(b"")

    # Return a volume_path that differs from the requested output_path to
    # verify run_whole_brain_3d trusts the returned value (Task 4).
    actual_volume_path = outputs_dir / "actual_volume.nii.gz"
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d.build_volume_from_tiffs",
        lambda **kwargs: {
            "volume_path": actual_volume_path,
            "shape": [2, 3, 4],
        },
    )
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d.prepare_half_template_inputs",
        lambda **kwargs: {
            "template_path": outputs_dir / "template_half.nii.gz",
            "annotation_path": outputs_dir / "annotation_half.nii.gz",
        },
    )
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d.run_ants_registration",
        lambda **kwargs: {
            "registered_volume": outputs_dir / "ants_result.nii.gz",
            "metrics_csv": outputs_dir / "registration_metrics.csv",
            "summary_txt": outputs_dir / "registration_summary.txt",
        },
    )
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d.refine_registered_volume",
        lambda **kwargs: {
            "final_registered_path": outputs_dir / "final_registered.nii.gz",
            "field_path": outputs_dir / "laplacian_deformation_field.npy",
            "metrics_csv": outputs_dir / "refinement_metrics.csv",
        },
    )
    refine_calls: list[dict] = []
    warp_calls: list[dict] = []
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d._apply_refinement_field_to_annotation_volume",
        lambda **kwargs: refine_calls.append(kwargs) or (outputs_dir / "annotation_refined.nii.gz"),
        raising=False,
    )
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d._warp_annotation_volume_to_input_space",
        lambda **kwargs: (
            warp_calls.append(kwargs) or (outputs_dir / "annotation_registered.nii.gz")
        ),
        raising=False,
    )
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d._extrapolate_annotation_to_tissue",
        lambda **kwargs: kwargs.get(
            "annotation_path", outputs_dir / "annotation_registered.nii.gz"
        ),
        raising=False,
    )
    export_calls: list[dict] = []
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d.export_registered_truth_slices",
        lambda **kwargs: (
            export_calls.append(kwargs)
            or [
                {
                    "slice_id": 0,
                    "real_slice_path": str(outputs_dir / "slice_0000_real.tif"),
                    "registered_label_path": str(outputs_dir / "slice_0000_registered_label.tif"),
                    "overlay_path": str(outputs_dir / "slice_0000_overlay.png"),
                }
            ]
        ),
    )
    stages: list[tuple[str, int]] = []

    def fake_run_quantification_from_truth(**kwargs):
        assert not any(percent == 100 for _stage_name, percent in stages)
        return {"leaf_csv": str(outputs_dir / "cell_counts_leaf.csv")}

    monkeypatch.setattr(
        "project.scripts.whole_brain_3d.run_quantification_from_truth",
        fake_run_quantification_from_truth,
    )

    def _progress(stage_name, stage_index, stage_count, percent, message, artifacts):
        stages.append((stage_name, percent))

    result = run_whole_brain_3d(
        cfg={
            "input": {
                "pixel_size_um_xy": 5.0,
                "slice_spacing_um": 25.0,
                "slicing_plane": "coronal",
            },
            "quantify_fn": lambda **kwargs: {"cells_mapped_csv": "ignored.csv"},
            "registration": {
                "atlas_hemisphere": "left",
                "ml_flip": False,
            },
        },
        input_dir=input_dir,
        outputs_dir=outputs_dir,
        merged_slice_paths=[],
        progress_cb=_progress,
    )

    assert [stage_name for stage_name, _percent in stages] == [
        "Volume Build",
        "Template Prep",
        "ANTS Registration",
        "Laplacian Refinement",
        "Truth Export",
        "Quantification",
    ]
    assert stages[-1] == ("Quantification", 100)
    assert refine_calls[0]["annotation_path"] == outputs_dir / "annotation_half.nii.gz"
    assert refine_calls[0]["field_path"] == outputs_dir / "laplacian_deformation_field.npy"
    assert warp_calls[0]["annotation_path"] == outputs_dir / "annotation_refined.nii.gz"
    assert export_calls[0]["annotation_volume_path"] == outputs_dir / "annotation_registered.nii.gz"
    assert result["truth_source"] == "3d_registered_volume"


def test_run_whole_brain_3d_skip_laplacian(tmp_path, monkeypatch):
    """When skip_laplacian_refinement=True, refine_registered_volume is NOT called."""
    import nibabel as nib
    import numpy as np

    input_dir = tmp_path / "input"
    outputs_dir = tmp_path / "outputs"
    input_dir.mkdir()
    outputs_dir.mkdir()
    for i in range(3):
        (input_dir / f"z{i:04d}.tif").write_bytes(b"")

    # Create a real NIfTI file that the skip path will copy
    dummy_vol = np.zeros((4, 4, 4), dtype=np.float32)
    dummy_nii_path = outputs_dir / "ants_result.nii.gz"
    nib.save(nib.Nifti1Image(dummy_vol, np.eye(4)), str(dummy_nii_path))

    actual_volume_path = outputs_dir / "actual_volume.nii.gz"
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d.build_volume_from_tiffs",
        lambda **kwargs: {"volume_path": actual_volume_path, "shape": [4, 4, 4]},
    )
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d.prepare_half_template_inputs",
        lambda **kwargs: {
            "template_path": outputs_dir / "template_half.nii.gz",
            "annotation_path": outputs_dir / "annotation_half.nii.gz",
        },
    )
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d.run_ants_registration",
        lambda **kwargs: {
            "registered_volume": dummy_nii_path,
            "metrics_csv": outputs_dir / "registration_metrics.csv",
            "summary_txt": outputs_dir / "registration_summary.txt",
        },
    )

    refine_calls: list[dict] = []
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d.refine_registered_volume",
        lambda **kwargs: (
            refine_calls.append(kwargs)
            or {
                "final_registered_path": outputs_dir / "final_registered.nii.gz",
                "field_path": outputs_dir / "laplacian_deformation_field.npy",
                "metrics_csv": outputs_dir / "refinement_metrics.csv",
            }
        ),
    )
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d._apply_refinement_field_to_annotation_volume",
        lambda **kwargs: outputs_dir / "annotation_refined.nii.gz",
        raising=False,
    )
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d._warp_annotation_volume_to_input_space",
        lambda **kwargs: outputs_dir / "annotation_registered.nii.gz",
        raising=False,
    )
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d._extrapolate_annotation_to_tissue",
        lambda **kwargs: kwargs.get(
            "annotation_path", outputs_dir / "annotation_registered.nii.gz"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d.export_registered_truth_slices",
        lambda **kwargs: [{"slice_id": 0}],
    )
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d.run_quantification_from_truth",
        lambda **kwargs: {"cells_mapped_csv": "ignored.csv"},
    )

    run_whole_brain_3d(
        cfg={
            "input": {"pixel_size_um_xy": 5.0, "slice_spacing_um": 25.0},
            "quantify_fn": lambda **kwargs: {},
            "registration": {
                "atlas_hemisphere": "left",
                "ml_flip": False,
                "skip_laplacian_refinement": True,
            },
        },
        input_dir=input_dir,
        outputs_dir=outputs_dir,
        merged_slice_paths=[],
    )

    # refine_registered_volume should NOT have been called
    assert len(refine_calls) == 0, "refine_registered_volume should be skipped"
    # But the zero deformation field should have been created
    field_path = outputs_dir / "laplacian_refinement" / "laplacian_deformation_field.npy"
    assert field_path.exists(), "Zero field should still be created"


def test_run_whole_brain_3d_requires_callable_quantify_fn(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    outputs_dir = tmp_path / "outputs"
    input_dir.mkdir()
    outputs_dir.mkdir()

    with pytest.raises(ValueError, match="quantify_fn"):
        run_whole_brain_3d(
            cfg={
                "input": {
                    "pixel_size_um_xy": 5.0,
                    "slice_spacing_um": 25.0,
                    "slicing_plane": "coronal",
                },
                "registration": {
                    "atlas_hemisphere": "left",
                },
            },
            input_dir=input_dir,
            outputs_dir=outputs_dir,
            merged_slice_paths=[],
        )


# ---------------------------------------------------------------------------
# Tests for _extrapolate_annotation_to_tissue
# ---------------------------------------------------------------------------


def _make_nifti(path: Path, data: np.ndarray) -> Path:
    """Helper: save a numpy array as a NIfTI file and return the path."""
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))
    return path


def _make_tiff(path: Path, data: np.ndarray) -> Path:
    """Helper: save a numpy array as a TIFF file and return the path."""
    import tifffile

    tifffile.imwrite(str(path), data)
    return path


class TestExtrapolateAnnotationToTissue:
    """Tests for _extrapolate_annotation_to_tissue."""

    def test_fully_annotated_no_change(self, tmp_path):
        """When annotation already fully covers tissue, nothing changes."""
        # 2 slices, 4x8 each -- annotation covers everything
        ann = np.ones((2, 4, 8), dtype=np.int32) * 42
        ann_path = _make_nifti(tmp_path / "ann.nii.gz", ann)

        # Create TIFF slices with tissue everywhere
        tiff_dir = tmp_path / "slices"
        tiff_dir.mkdir()
        slice_paths = []
        for z in range(2):
            tif_data = np.full((4, 8), 200, dtype=np.uint16)
            slice_paths.append(_make_tiff(tiff_dir / f"z{z:04d}.tif", tif_data))

        result = _extrapolate_annotation_to_tissue(
            annotation_path=ann_path,
            input_volume_path=tmp_path / "dummy_vol.nii.gz",
            merged_slice_paths=slice_paths,
        )

        result_vol = nib.load(str(result)).get_fdata().astype(np.int32)
        np.testing.assert_array_equal(result_vol, ann)

    def test_gap_filled_with_nearest_label(self, tmp_path):
        """Annotation gap on one side is filled with the nearest label via EDT."""
        # 1 slice, 4x8: annotation covers left half (cols 0-3), gap on right (cols 4-7)
        ann = np.zeros((1, 4, 8), dtype=np.int32)
        ann[0, :, :4] = 7  # label 7 on left half
        ann_path = _make_nifti(tmp_path / "ann.nii.gz", ann)

        # Tissue covers the entire slice -- use a range [200, 1000] so the
        # 5th-percentile bg estimate is ~240 and all pixels exceed it.
        tiff_dir = tmp_path / "slices"
        tiff_dir.mkdir()
        tif_data = np.linspace(200, 1000, 4 * 8).reshape(4, 8).astype(np.uint16)
        slice_paths = [_make_tiff(tiff_dir / "z0000.tif", tif_data)]

        result = _extrapolate_annotation_to_tissue(
            annotation_path=ann_path,
            input_volume_path=tmp_path / "dummy_vol.nii.gz",
            merged_slice_paths=slice_paths,
        )

        result_vol = nib.load(str(result)).get_fdata().astype(np.int32)
        # Every tissue pixel in the gap should be filled with label 7
        gap = result_vol[0, :, 4:]
        assert (gap == 7).all(), f"Expected gap to be filled with label 7, got:\n{result_vol[0]}"

    def test_slice_index_beyond_merged_paths_no_crash(self, tmp_path):
        """If annotation volume has more Z-slices than merged_slice_paths, extras are skipped."""
        # 3 slices in annotation, but only 1 TIFF provided
        ann = np.zeros((3, 4, 8), dtype=np.int32)
        ann[0, :, :4] = 5
        ann[1, :, :4] = 5  # has gap, but no TIFF for z=1
        ann[2, :, :4] = 5  # has gap, but no TIFF for z=2
        ann_path = _make_nifti(tmp_path / "ann.nii.gz", ann)

        tiff_dir = tmp_path / "slices"
        tiff_dir.mkdir()
        tif_data = np.linspace(200, 1000, 4 * 8).reshape(4, 8).astype(np.uint16)
        slice_paths = [_make_tiff(tiff_dir / "z0000.tif", tif_data)]

        # Should not crash -- z=1 and z=2 are beyond len(merged_slice_paths)
        result = _extrapolate_annotation_to_tissue(
            annotation_path=ann_path,
            input_volume_path=tmp_path / "dummy_vol.nii.gz",
            merged_slice_paths=slice_paths,
        )

        result_vol = nib.load(str(result)).get_fdata().astype(np.int32)
        # z=0 gap should be filled
        assert (result_vol[0, :, 4:] == 5).all()
        # z=1, z=2 should remain unfilled (still 0 on right half)
        assert (result_vol[1, :, 4:] == 0).all()
        assert (result_vol[2, :, 4:] == 0).all()

    def test_empty_tissue_mask_no_filling(self, tmp_path):
        """When the TIFF slice is all-zero (no tissue), no filling happens."""
        ann = np.zeros((1, 4, 8), dtype=np.int32)
        ann[0, :, :4] = 3
        ann_path = _make_nifti(tmp_path / "ann.nii.gz", ann)

        tiff_dir = tmp_path / "slices"
        tiff_dir.mkdir()
        # All-zero TIFF -- no tissue
        tif_data = np.zeros((4, 8), dtype=np.uint16)
        slice_paths = [_make_tiff(tiff_dir / "z0000.tif", tif_data)]

        result = _extrapolate_annotation_to_tissue(
            annotation_path=ann_path,
            input_volume_path=tmp_path / "dummy_vol.nii.gz",
            merged_slice_paths=slice_paths,
        )

        result_vol = nib.load(str(result)).get_fdata().astype(np.int32)
        # Right half should remain 0 (no tissue to fill)
        assert (result_vol[0, :, 4:] == 0).all()

    def test_multi_label_nearest_neighbor(self, tmp_path):
        """Gap pixels should get the label from the nearest annotated pixel."""
        # 1 slice, 6x6: two labels separated by a gap
        ann = np.zeros((1, 6, 6), dtype=np.int32)
        ann[0, 0:3, 0:2] = 10  # top-left block
        ann[0, 3:6, 0:2] = 20  # bottom-left block
        # cols 2-5 are gaps
        ann_path = _make_nifti(tmp_path / "ann.nii.gz", ann)

        tiff_dir = tmp_path / "slices"
        tiff_dir.mkdir()
        tif_data = np.linspace(200, 1000, 6 * 6).reshape(6, 6).astype(np.uint16)
        slice_paths = [_make_tiff(tiff_dir / "z0000.tif", tif_data)]

        result = _extrapolate_annotation_to_tissue(
            annotation_path=ann_path,
            input_volume_path=tmp_path / "dummy_vol.nii.gz",
            merged_slice_paths=slice_paths,
        )

        result_vol = nib.load(str(result)).get_fdata().astype(np.int32)
        # Top rows should be filled with 10, bottom with 20
        assert (result_vol[0, 0:3, 2:] == 10).all()
        assert (result_vol[0, 3:6, 2:] == 20).all()
