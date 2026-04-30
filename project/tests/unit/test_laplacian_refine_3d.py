from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from project.scripts.registration_3d_ants import compute_registration_metrics


def test_refine_registered_volume_writes_field_and_final_volume(tmp_path):
    # refine_registered_volume calls Xu Lab sliceToSlice3DLaplacian which
    # imports joblib + tqdm + skimage.feature at module level.  The Py 3.10
    # CI lane installs the lighter dep set, so the underlying machinery
    # isn't available there — skip rather than error.
    pytest.importorskip("joblib")
    pytest.importorskip("skimage.feature")
    from project.scripts.laplacian_refine_3d import refine_registered_volume
    fixed_path = tmp_path / "fixed.nii.gz"
    moving_path = tmp_path / "moving.nii.gz"
    out_dir = tmp_path / "refined"

    fixed_data = np.zeros((5, 5, 5), dtype=np.float32)
    fixed_data[2, 2, 2] = 1.0
    moving_data = np.zeros((5, 5, 5), dtype=np.float32)
    moving_data[1, 1, 1] = 1.0

    affine = np.eye(4)
    nib.save(nib.Nifti1Image(fixed_data, affine), str(fixed_path))
    nib.save(nib.Nifti1Image(moving_data, affine), str(moving_path))

    result = refine_registered_volume(
        fixed_path,
        moving_path,
        out_dir,
        iterations=5,
        lambda_=0.15,
    )

    final_registered_path = Path(result["final_registered_path"])
    field_path = Path(result["field_path"])
    metrics_csv = Path(result["metrics_csv"])

    assert final_registered_path.exists()
    assert field_path.exists()
    assert metrics_csv.exists()

    final_img = nib.load(str(final_registered_path))
    assert final_img.shape == (5, 5, 5)

    field = np.load(str(field_path))
    assert field.shape == (3, 5, 5, 5)
    assert field.dtype == np.float32

    with metrics_csv.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)

    assert rows
    assert reader.fieldnames == ["metric", "before", "after", "change"]
    assert {row["metric"] for row in rows} >= set(
        compute_registration_metrics(fixed_data, moving_data)
    )
