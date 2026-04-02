from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np

from project.scripts.laplacian_refine_3d import refine_registered_volume


def test_refine_registered_volume_writes_field_and_final_volume(tmp_path):
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
