from __future__ import annotations

import pytest

from project.scripts.whole_brain_3d import run_whole_brain_3d


def test_run_whole_brain_3d_emits_expected_stage_sequence(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    outputs_dir = tmp_path / "outputs"
    input_dir.mkdir()
    outputs_dir.mkdir()

    monkeypatch.setattr(
        "project.scripts.whole_brain_3d.build_volume_from_tiffs",
        lambda **kwargs: {
            "volume_path": outputs_dir / "volume.nii.gz",
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
        lambda **kwargs: refine_calls.append(kwargs)
        or (outputs_dir / "annotation_refined.nii.gz"),
        raising=False,
    )
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d._warp_annotation_volume_to_input_space",
        lambda **kwargs: warp_calls.append(kwargs)
        or (outputs_dir / "annotation_registered.nii.gz"),
        raising=False,
    )
    export_calls: list[dict] = []
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d.export_registered_truth_slices",
        lambda **kwargs: export_calls.append(kwargs)
        or [
            {
                "slice_id": 0,
                "real_slice_path": str(outputs_dir / "slice_0000_real.tif"),
                "registered_label_path": str(outputs_dir / "slice_0000_registered_label.tif"),
                "overlay_path": str(outputs_dir / "slice_0000_overlay.png"),
            }
        ],
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
