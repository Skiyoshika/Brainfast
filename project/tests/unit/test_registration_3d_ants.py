from __future__ import annotations

import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

from project.scripts.registration_3d_ants import (
    compute_registration_metrics,
    run_ants_registration,
)


class _FakeAntsModule:
    def image_read(self, path):
        return nib.load(str(path)).get_fdata().astype(np.float32)

    def image_write(self, arr, path):
        img = nib.Nifti1Image(np.asarray(arr, dtype=np.float32), np.eye(4))
        nib.save(img, str(path))

    def registration(self, fixed, moving, type_of_transform, random_seed):
        warped = np.asarray(moving, dtype=np.float32) * 0.5 + np.asarray(fixed, dtype=np.float32) * 0.5
        return {
            "warpedmovout": warped,
            "fwdtransforms": [f"{type_of_transform}_{random_seed}_fwd.mat"],
            "invtransforms": [f"{type_of_transform}_{random_seed}_inv.mat"],
        }


def test_run_ants_registration_writes_metrics_and_summary(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "ants", _FakeAntsModule())

    fixed_path = tmp_path / "fixed.nii.gz"
    moving_path = tmp_path / "moving.nii.gz"
    out_dir = tmp_path / "ants_out"

    fixed_data = np.zeros((4, 4, 4), dtype=np.float32)
    fixed_data[1:3, 1:3, 1:3] = 1.0
    moving_data = np.zeros((4, 4, 4), dtype=np.float32)
    moving_data[2:, 2:, 2:] = 1.0

    nib.save(nib.Nifti1Image(fixed_data, np.eye(4)), str(fixed_path))
    nib.save(nib.Nifti1Image(moving_data, np.eye(4)), str(moving_path))

    result = run_ants_registration(
        fixed_path,
        moving_path,
        out_dir,
        transform="SyN",
        random_seed=42,
    )

    registered_volume = Path(result["registered_volume"])
    metrics_csv = Path(result["metrics_csv"])
    summary_txt = Path(result["summary_txt"])

    assert registered_volume.exists()
    assert metrics_csv.exists()
    assert summary_txt.exists()

    with metrics_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert rows
    assert rows[0]["metric"] == "NCC"


def test_compute_registration_metrics_returns_expected_metric_names():
    fixed_arr = np.zeros((3, 3, 3), dtype=np.float32)
    fixed_arr[1, 1, 1] = 1.0
    moving_arr = np.zeros((3, 3, 3), dtype=np.float32)
    moving_arr[1, 1, 1] = 1.0

    metrics = compute_registration_metrics(fixed_arr, moving_arr)

    assert {"NCC", "NMI", "SSIM", "Dice", "MSE"} <= set(metrics)
