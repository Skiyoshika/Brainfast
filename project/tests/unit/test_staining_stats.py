from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from project.scripts.staining_stats import compute_staining_stats


def _save_nifti(path: Path, data: np.ndarray) -> None:
    nib.save(nib.Nifti1Image(data, np.eye(4)), str(path))


def test_compute_staining_stats_reports_coverage_and_positive_fraction(tmp_path: Path) -> None:
    registered = np.zeros((5, 5, 5), dtype=np.float32)
    annotation = np.zeros((5, 5, 5), dtype=np.int16)

    annotation[1:4, 1:4, 1:4] = 1
    registered[1:4, 1:4, 1:4] = 20.0
    registered[2, 1:4, 1:4] = 200.0

    reg_path = tmp_path / "registered.nii.gz"
    ann_path = tmp_path / "annotation.nii.gz"
    _save_nifti(reg_path, registered)
    _save_nifti(ann_path, annotation)

    stats = compute_staining_stats(reg_path, ann_path)

    assert stats["atlas_voxels"] == 27.0
    assert stats["covered_voxels"] == 27.0
    assert stats["positive_voxels"] == 9.0
    assert stats["atlas_coverage"] == 1.0
    assert stats["staining_rate"] == 9.0 / 27.0
    assert stats["positive_fraction_of_atlas"] == 9.0 / 27.0
    assert 0.0 < stats["signal_threshold"] < 1.0
    assert 0.0 < stats["mean_signal"] < 1.0


def test_compute_staining_stats_handles_empty_annotation(tmp_path: Path) -> None:
    reg_path = tmp_path / "registered.nii.gz"
    ann_path = tmp_path / "annotation.nii.gz"
    _save_nifti(reg_path, np.ones((4, 4, 4), dtype=np.float32))
    _save_nifti(ann_path, np.zeros((4, 4, 4), dtype=np.int16))

    stats = compute_staining_stats(reg_path, ann_path)

    assert stats["atlas_voxels"] == 0.0
    assert stats["covered_voxels"] == 0.0
    assert stats["positive_voxels"] == 0.0
    assert stats["atlas_coverage"] == 0.0
    assert stats["staining_rate"] == 0.0
