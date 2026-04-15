from __future__ import annotations

import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from project.scripts.registration_3d_ants import (
    _nmi_histogram,
    _norm,
    _tissue_mask,
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

    def registration(self, fixed, moving, type_of_transform="SyN", random_seed=42, **kwargs):
        self.registration_calls.append(
            {
                "type_of_transform": type_of_transform,
                "random_seed": random_seed,
                "fixed_shape": np.shape(fixed),
                "moving_shape": np.shape(moving),
                **kwargs,
            }
        )
        warped = (
            np.asarray(moving, dtype=np.float32) * 0.5 + np.asarray(fixed, dtype=np.float32) * 0.5
        )
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
    # SyN/SyNRA both accepted — code upgrades SyN to SyNRA with CC metric
    assert len(result["forward_transforms"]) == 1
    assert len(result["inverse_transforms"]) == 1
    assert len(fake_ants.registration_calls) == 1
    call = fake_ants.registration_calls[0]
    assert call["type_of_transform"] in ("SyN", "SyNRA")
    assert call["random_seed"] == 42
    assert call["fixed_shape"] == (4, 4, 4)
    assert call["moving_shape"] == (4, 4, 4)

    with metrics_csv.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    assert [row["metric"] for row in rows] == ["NCC", "NMI", "SSIM", "Dice", "MSE", "PSNR"]
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

    assert set(metrics) == {"NCC", "NMI", "SSIM", "Dice", "MSE", "PSNR"}
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


def test_run_ants_registration_succeeds_when_transform_copy_fails(tmp_path, monkeypatch):
    """Task 5: transform persistence is best-effort — registration succeeds
    even when transform files cannot be copied."""

    class _FakeAntsNoCopy:
        def image_read(self, path):
            return nib.load(str(path)).get_fdata().astype(np.float32)

        def image_write(self, arr, path):
            img = nib.Nifti1Image(np.asarray(arr, dtype=np.float32), np.eye(4))
            nib.save(img, str(path))

        def registration(self, fixed, moving, type_of_transform="SyN", random_seed=42, **kwargs):
            warped = (
                np.asarray(moving, dtype=np.float32) * 0.5
                + np.asarray(fixed, dtype=np.float32) * 0.5
            )
            # Return paths that do NOT exist on disk — simulates temp cleanup
            return {
                "warpedmovout": warped,
                "fwdtransforms": ["/nonexistent/fwd.mat"],
                "invtransforms": ["/nonexistent/inv.mat"],
            }

    fake_ants = _FakeAntsNoCopy()
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

    # Should NOT raise despite transform files not existing
    result = run_ants_registration(
        fixed_path,
        moving_path,
        out_dir,
        transform="SyN",
        random_seed=42,
    )

    # Registration output itself should still be written
    assert Path(result["registered_volume"]).exists()
    assert Path(result["metrics_csv"]).exists()
    assert Path(result["summary_txt"]).exists()
    # Transform metadata falls back to original paths instead of crashing
    assert result["forward_transforms"] == ["/nonexistent/fwd.mat"]
    assert result["inverse_transforms"] == ["/nonexistent/inv.mat"]


# ---------------------------------------------------------------------------
# Tests for _tissue_mask
# ---------------------------------------------------------------------------


class TestTissueMask:
    """Tests for _tissue_mask."""

    def test_above_threshold_is_true(self):
        arr = np.array([0.0, 0.05, 0.1, 0.2, 0.5, 1.0], dtype=np.float32)
        mask = _tissue_mask(arr, threshold=0.1)
        expected = np.array([False, False, False, True, True, True])
        np.testing.assert_array_equal(mask, expected)

    def test_default_threshold(self):
        arr = np.array([0.0, 0.1, 0.11, 1.0], dtype=np.float32)
        mask = _tissue_mask(arr)
        expected = np.array([False, False, True, True])
        np.testing.assert_array_equal(mask, expected)

    def test_custom_threshold(self):
        arr = np.array([0.0, 0.5, 0.6, 1.0], dtype=np.float32)
        mask = _tissue_mask(arr, threshold=0.5)
        expected = np.array([False, False, True, True])
        np.testing.assert_array_equal(mask, expected)

    def test_all_zeros(self):
        arr = np.zeros((4, 4), dtype=np.float32)
        mask = _tissue_mask(arr)
        assert not mask.any()

    def test_3d_array(self):
        arr = np.zeros((2, 3, 4), dtype=np.float32)
        arr[0, 1, 2] = 0.5
        mask = _tissue_mask(arr)
        assert mask.sum() == 1
        assert mask[0, 1, 2]


# ---------------------------------------------------------------------------
# Tests for _nmi_histogram
# ---------------------------------------------------------------------------


class TestNmiHistogram:
    """Tests for _nmi_histogram."""

    def test_identical_arrays_nmi_near_two(self):
        """Identical images should give NMI close to 2.0."""
        rng = np.random.RandomState(42)
        arr = rng.rand(20, 20).astype(np.float32) * 0.8 + 0.15  # all above threshold
        nmi = _nmi_histogram(arr, arr)
        assert nmi > 1.8, f"NMI for identical arrays should be near 2.0, got {nmi}"

    def test_independent_random_nmi_near_one(self):
        """Independent random arrays should give NMI close to 1.0."""
        rng = np.random.RandomState(42)
        a = rng.rand(30, 30).astype(np.float32) * 0.8 + 0.15
        b = rng.rand(30, 30).astype(np.float32) * 0.8 + 0.15
        nmi = _nmi_histogram(a, b)
        assert 0.9 < nmi < 1.3, f"NMI for independent arrays should be near 1.0, got {nmi}"

    def test_all_zero_returns_one(self):
        """All-zero inputs have <100 tissue pixels, so returns 1.0."""
        a = np.zeros((10, 10), dtype=np.float32)
        b = np.zeros((10, 10), dtype=np.float32)
        nmi = _nmi_histogram(a, b)
        assert nmi == 1.0

    def test_too_few_tissue_pixels_returns_one(self):
        """Fewer than 100 tissue pixels should return 1.0."""
        a = np.zeros((10, 10), dtype=np.float32)
        b = np.zeros((10, 10), dtype=np.float32)
        # Set only 50 pixels above threshold
        a.ravel()[:50] = 0.5
        b.ravel()[:50] = 0.5
        nmi = _nmi_histogram(a, b)
        assert nmi == 1.0

    def test_sufficient_tissue_pixels_returns_finite(self):
        """With enough tissue pixels, NMI should be a finite value >= 1.0."""
        rng = np.random.RandomState(99)
        a = rng.rand(15, 15).astype(np.float32) * 0.8 + 0.15  # 225 pixels, all tissue
        b = rng.rand(15, 15).astype(np.float32) * 0.8 + 0.15
        nmi = _nmi_histogram(a, b)
        assert np.isfinite(nmi)
        assert nmi >= 1.0
