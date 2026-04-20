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


def test_run_whole_brain_3d_passes_tuned_params_to_truth_export(tmp_path, monkeypatch):
    """Task 2 — the default whole-brain path must carry learned
    ``warp_params`` / ``fit_mode`` / ``edge_smooth_iter`` through to
    ``export_registered_truth_slices`` so the UI-triggered calibration
    learn result actually changes the exported rasters.
    """
    input_dir = tmp_path / "input"
    outputs_dir = tmp_path / "outputs"
    input_dir.mkdir()
    outputs_dir.mkdir()
    for i in range(3):
        (input_dir / f"z{i:04d}.tif").write_bytes(b"")

    # Stub every stage except Truth Export so we can assert on its kwargs
    for fn, ret in (
        ("build_volume_from_tiffs", {"volume_path": outputs_dir / "v.nii.gz", "shape": [2, 3, 4]}),
        ("prepare_half_template_inputs", {
            "template_path": outputs_dir / "t.nii.gz",
            "annotation_path": outputs_dir / "a.nii.gz",
        }),
        ("run_ants_registration", {
            "registered_volume": outputs_dir / "ants.nii.gz",
            "metrics_csv": outputs_dir / "m.csv",
            "summary_txt": outputs_dir / "s.txt",
        }),
        ("refine_registered_volume", {
            "final_registered_path": outputs_dir / "final.nii.gz",
            "field_path": outputs_dir / "f.npy",
            "metrics_csv": outputs_dir / "rm.csv",
        }),
    ):
        monkeypatch.setattr(
            f"project.scripts.whole_brain_3d.{fn}", lambda _ret=ret, **kw: _ret
        )
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d._apply_refinement_field_to_annotation_volume",
        lambda **kw: outputs_dir / "annotation_refined.nii.gz",
        raising=False,
    )
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d._warp_annotation_volume_to_input_space",
        lambda **kw: outputs_dir / "annotation_registered.nii.gz",
        raising=False,
    )
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d._extrapolate_annotation_to_tissue",
        lambda **kw: kw.get("annotation_path", outputs_dir / "annotation_registered.nii.gz"),
        raising=False,
    )
    export_calls: list[dict] = []
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d.export_registered_truth_slices",
        lambda **kw: export_calls.append(kw) or [],
    )
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d.run_quantification_from_truth",
        lambda **kw: {"leaf_csv": "ignored.csv"},
    )

    learned_warp = {"custom_knob": 0.42}
    run_whole_brain_3d(
        cfg={
            "input": {
                "pixel_size_um_xy": 5.0,
                "slice_spacing_um": 25.0,
                "slicing_plane": "coronal",
            },
            "quantify_fn": lambda **kw: {},
            "registration": {"atlas_hemisphere": "left", "ml_flip": False},
            # Learned calibration fields — these must land in truth export kwargs
            "truth_export": {
                "warp_params": learned_warp,
                "fit_mode": "contain",
                "edge_smooth_iter": 3,
            },
        },
        input_dir=input_dir,
        outputs_dir=outputs_dir,
        merged_slice_paths=[],
    )

    assert len(export_calls) == 1
    kw = export_calls[0]
    # Truth export must have received the learned parameters, not the old
    # hard-coded "cover" + 0 defaults.
    assert kw.get("fit_mode") == "contain"
    assert kw.get("edge_smooth_iter") == 3
    assert kw.get("warp_params", {}).get("custom_knob") == 0.42


def test_run_whole_brain_3d_reuses_registration_from_prior_dir(tmp_path, monkeypatch):
    """When ``registration.reuse_from_dir`` points at a prior channel's
    outputs_dir with all required artifacts, stages 1–5 are skipped and only
    Quantification (stage 6) runs. The *current* channel's merged_slice_paths
    still flow into quantification so cell detection uses C1 data.

    This is the dual-channel case: C0 already ran the full pipeline once and
    produced ants_registration/, laplacian_refinement/, truth_export/, etc.
    C1 only needs to re-detect + re-map against C0's truth slices.
    """
    prior_dir = tmp_path / "prior_c0"
    prior_dir.mkdir()
    (prior_dir / "ants_registration").mkdir()
    (prior_dir / "laplacian_refinement").mkdir()
    (prior_dir / "truth_export").mkdir()
    # Required artifact stubs — real files, not just presence markers
    (prior_dir / "ants_registration" / "annotation_registered.nii.gz").write_bytes(b"\x00")
    (prior_dir / "laplacian_refinement" / "annotation_refined.nii.gz").write_bytes(b"\x00")
    (prior_dir / "laplacian_refinement" / "laplacian_deformation_field.npy").write_bytes(b"\x00")
    # Fake 3 truth rows by leaving the dir empty; we monkeypatch the loader

    input_dir = tmp_path / "input_c1"
    outputs_dir = tmp_path / "outputs_c1"
    input_dir.mkdir()
    outputs_dir.mkdir()
    for i in range(3):
        (input_dir / f"z{i:04d}.tif").write_bytes(b"")

    # Fail the test if any stage-1-to-5 function is called. Quantification
    # must be the ONLY stage that runs.
    def _should_not_call(name):
        def _die(**kwargs):
            raise AssertionError(f"reuse mode must not call {name}")

        return _die

    for fn_name in (
        "build_volume_from_tiffs",
        "prepare_half_template_inputs",
        "run_ants_registration",
        "refine_registered_volume",
        "_apply_refinement_field_to_annotation_volume",
        "_warp_annotation_volume_to_input_space",
        "export_registered_truth_slices",
    ):
        monkeypatch.setattr(
            f"project.scripts.whole_brain_3d.{fn_name}",
            _should_not_call(fn_name),
            raising=False,
        )

    quant_calls: list[dict] = []

    def fake_quantify(**kwargs):
        quant_calls.append(kwargs)
        return {"leaf_csv": str(outputs_dir / "cell_counts_leaf.csv")}

    monkeypatch.setattr(
        "project.scripts.whole_brain_3d.run_quantification_from_truth",
        fake_quantify,
    )

    stages: list[tuple[str, int]] = []

    def _progress(stage_name, stage_index, stage_count, percent, message, artifacts):
        stages.append((stage_name, percent))

    result = run_whole_brain_3d(
        cfg={
            "input": {
                "pixel_size_um_xy": 5.0,
                "slice_spacing_um": 25.0,
                "slicing_plane": "coronal",
                "active_channel": "farred",
            },
            "quantify_fn": lambda **kwargs: {"cells_mapped_csv": "c1.csv"},
            "registration": {
                "atlas_hemisphere": "right_flipped",
                "ml_flip": False,
                "reuse_from_dir": str(prior_dir),
            },
        },
        input_dir=input_dir,
        outputs_dir=outputs_dir,
        merged_slice_paths=[],
        progress_cb=_progress,
    )

    # Only Quantification should have emitted progress
    stage_names = [s for s, _p in stages]
    assert "Quantification" in stage_names
    assert "ANTS Registration" not in stage_names
    assert "Laplacian Refinement" not in stage_names
    assert "Truth Export" not in stage_names

    # quantify_fn received C1's input_dir (merged_slice_paths from C1) even
    # though the annotation paths came from the prior C0 dir
    assert len(quant_calls) == 1
    assert quant_calls[0]["input_dir"] == input_dir
    # ants_meta / refine_meta should reference prior dir paths so truth
    # export artifacts are consistent with C0's registration
    reused_ants = quant_calls[0]["ants_meta"]
    assert Path(reused_ants["registered_volume"]).parent == (prior_dir / "ants_registration")
    assert result["truth_source"] == "3d_registered_volume"


def test_run_whole_brain_3d_reuse_requires_core_artifacts(tmp_path):
    """If reuse_from_dir is set but key artifacts are missing, fail loudly
    rather than silently skipping and producing garbage output.
    """
    prior_dir = tmp_path / "incomplete_prior"
    prior_dir.mkdir()
    # intentionally empty — no ants_registration/ etc.

    input_dir = tmp_path / "input_c1"
    outputs_dir = tmp_path / "outputs_c1"
    input_dir.mkdir()
    outputs_dir.mkdir()

    with pytest.raises((FileNotFoundError, ValueError), match="reuse"):
        run_whole_brain_3d(
            cfg={
                "input": {"pixel_size_um_xy": 5.0, "slice_spacing_um": 25.0},
                "quantify_fn": lambda **kwargs: {},
                "registration": {"reuse_from_dir": str(prior_dir)},
            },
            input_dir=input_dir,
            outputs_dir=outputs_dir,
            merged_slice_paths=[],
        )


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


def test_ap_auto_compute_uses_source_z_numbers_when_merged_names_lack_z(
    tmp_path, monkeypatch
):
    """AP auto-compute must read z-numbers from original input_dir files,
    not the sanitized merged_####.tif filenames.

    Regression: when merged files are named ``merged_0000.tif`` the regex
    ``z(\\d+)`` matches nothing, so `_z_nums` stays empty and ap_start/ap_end
    silently fall back to the full-atlas [0, 528] default. This forces ANTs to
    stretch a partial brain (~2.8 mm of slices) across the whole 12 mm atlas
    and destroys the registration (Dice ≈ 0.30 vs Miki-level 0.74).

    Expected: with source files z0050..z0600.tif, scale=0.2, offset=150, the
    AP window passed to ``prepare_half_template_inputs`` must be around
    [150, 280], not [0, 528].
    """
    input_dir = tmp_path / "input"
    outputs_dir = tmp_path / "outputs"
    merged_dir = tmp_path / "merged"
    input_dir.mkdir()
    outputs_dir.mkdir()
    merged_dir.mkdir()

    # Real microscope layout: z-numbered source slices, z = 50, 55, ..., 600.
    z_values = list(range(50, 601, 5))
    for z in z_values:
        (input_dir / f"z{z:04d}.tif").write_bytes(b"")

    # Merged sanitized filenames — this is what whole_brain_3d.py observes
    # after the merge stage. These file names do NOT contain "z<digits>".
    merged_paths = []
    for i, _ in enumerate(z_values):
        p = merged_dir / f"merged_{i:04d}.tif"
        p.write_bytes(b"")
        merged_paths.append(p)

    ap_capture: dict = {}

    monkeypatch.setattr(
        "project.scripts.whole_brain_3d.build_volume_from_tiffs",
        lambda **kwargs: {"volume_path": outputs_dir / "vol.nii.gz", "shape": [2, 2, 2]},
    )

    def capture_prepare(**kwargs):
        ap_capture["ap_start"] = kwargs.get("ap_start")
        ap_capture["ap_end"] = kwargs.get("ap_end")
        return {
            "template_path": outputs_dir / "template_half.nii.gz",
            "annotation_path": outputs_dir / "annotation_half.nii.gz",
        }

    monkeypatch.setattr(
        "project.scripts.whole_brain_3d.prepare_half_template_inputs", capture_prepare
    )
    # Raise immediately after template_prep so we do not need to mock ANTs etc.
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d.run_ants_registration",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("halt-after-prepare")),
    )

    with pytest.raises(RuntimeError, match="halt-after-prepare"):
        run_whole_brain_3d(
            cfg={
                "input": {
                    "pixel_size_um_xy": 5.0,
                    "slice_spacing_um": 25.0,
                    "slicing_plane": "coronal",
                    "slice_glob": "z*.tif",
                },
                "quantify_fn": lambda **kwargs: {},
                "registration": {
                    "atlas_hemisphere": "left",
                    "ml_flip": False,
                    "atlas_z_from_filename": True,
                    "atlas_z_z_scale": 0.2,
                    "atlas_z_offset": 150,
                },
            },
            input_dir=input_dir,
            outputs_dir=outputs_dir,
            merged_slice_paths=merged_paths,
        )

    # Expected with z_min=50, z_max=600, scale=0.2, offset=150:
    #   ap_start = max(0, int(50*0.2)+150-10)  = 150
    #   ap_end   = min(528, int(600*0.2)+150+10) = 280
    assert ap_capture["ap_start"] == 150, (
        f"AP auto-compute regressed: ap_start={ap_capture['ap_start']}, "
        "expected 150 (source z=50..600, scale=0.2, offset=150)."
    )
    assert ap_capture["ap_end"] == 280, (
        f"AP auto-compute regressed: ap_end={ap_capture['ap_end']}, "
        "expected 280 (source z=50..600, scale=0.2, offset=150)."
    )
