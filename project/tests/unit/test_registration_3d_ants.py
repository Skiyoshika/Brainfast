from __future__ import annotations

import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from project.scripts.registration_3d_ants import (
    _norm,
    compute_registration_metrics,
    run_ants_registration,
)


class _FakeAntsModule:
    def __init__(self):
        self.registration_calls: list[dict[str, object]] = []

    def image_read(self, path):
        return nib.load(str(path)).get_fdata().astype(np.float32)

    def image_write(self, arr, path):
        img = nib.Nifti1Image(np.asarray(arr, dtype=np.float32), np.eye(4))
        nib.save(img, str(path))

    def registration(self, fixed, moving, type_of_transform, random_seed):
        self.registration_calls.append(
            {
                "type_of_transform": type_of_transform,
                "random_seed": random_seed,
                "fixed_shape": np.shape(fixed),
                "moving_shape": np.shape(moving),
            }
        )
        warped = np.asarray(moving, dtype=np.float32) * 0.5 + np.asarray(fixed, dtype=np.float32) * 0.5
        return {
            "warpedmovout": warped,
            "fwdtransforms": [f"{type_of_transform}_{random_seed}_fwd.mat"],
            "invtransforms": [f"{type_of_transform}_{random_seed}_inv.mat"],
        }


def test_norm_returns_float32_and_clips_to_unit_interval():
    arr = np.array([-100.0, 0.0, 1.0, 2.0, 100.0], dtype=np.float64)

    normed = _norm(arr)

    assert normed.dtype == np.float32
    assert normed.shape == arr.shape
    assert np.all(normed >= 0.0)
    assert np.all(normed <= 1.0)
    assert normed.min() == pytest.approx(0.0)
    assert normed.max() == pytest.approx(1.0)


def test_run_ants_registration_writes_metrics_and_summary(tmp_path, monkeypatch):
    fake_ants = _FakeAntsModule()
    monkeypatch.setitem(sys.modules, "ants", fake_ants)

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
    assert result["forward_transforms"] == ["SyN_42_fwd.mat"]
    assert result["inverse_transforms"] == ["SyN_42_inv.mat"]
    assert fake_ants.registration_calls == [
        {
            "type_of_transform": "SyN",
            "random_seed": 42,
            "fixed_shape": (4, 4, 4),
            "moving_shape": (4, 4, 4),
        }
    ]

    with metrics_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert [row["metric"] for row in rows] == ["NCC", "NMI", "SSIM", "Dice", "MSE"]
    assert all(np.isfinite(float(row["value"])) for row in rows)

    assert summary_txt.read_text(encoding="utf-8").splitlines() == [
        f"fixed image: {fixed_path}",
        f"moving image: {moving_path}",
        "transform: SyN",
        f"registered output: {registered_volume}",
    ]


def test_compute_registration_metrics_constant_path_returns_finite_values():
    fixed_arr = np.zeros((4, 4, 4), dtype=np.float32)
    moving_arr = np.zeros((4, 4, 4), dtype=np.float32)

    metrics = compute_registration_metrics(fixed_arr, moving_arr)

    assert set(metrics) == {"NCC", "NMI", "SSIM", "Dice", "MSE"}
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["NCC"] == pytest.approx(0.0)
    assert metrics["Dice"] == pytest.approx(1.0)
    assert metrics["MSE"] == pytest.approx(0.0)


def test_compute_registration_metrics_tiny_slice_path_returns_finite_values():
    fixed_arr = np.zeros((1, 2, 2), dtype=np.float32)
    fixed_arr[0, 0, 0] = 1.0
    moving_arr = fixed_arr.copy()

    metrics = compute_registration_metrics(fixed_arr, moving_arr)

    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["SSIM"] == pytest.approx(1.0)
    assert metrics["Dice"] == pytest.approx(1.0)


def test_compute_registration_metrics_normal_path_returns_finite_values():
    fixed_arr = np.zeros((4, 4, 4), dtype=np.float32)
    fixed_arr[1:3, 1:3, 1:3] = 1.0
    moving_arr = np.zeros((4, 4, 4), dtype=np.float32)
    moving_arr[2:, 2:, 2:] = 1.0

    metrics = compute_registration_metrics(fixed_arr, moving_arr)

    assert all(np.isfinite(value) for value in metrics.values())
    assert 0.0 <= metrics["Dice"] < 1.0
    assert metrics["MSE"] > 0.0
