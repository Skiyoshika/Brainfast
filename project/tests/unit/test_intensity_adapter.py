"""Unit tests for Phase α intensity adaptation.

Phase α's goal is to reduce the cross-modality ANTs gap (fluorescence moving
vs Allen-average template) by pre-aligning the moving volume's intensity
distribution and local contrast to the template's.

These tests exercise the three building blocks plus the mode-routing that
registration_3d_volume.py wires up.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_histogram_match_reshapes_distribution_toward_reference():
    """After histogram matching the moving volume's percentile summary should
    be much closer to the reference than before.
    """
    from project.scripts.intensity_adapter import histogram_match_to_template

    rng = np.random.default_rng(0)
    # Moving: bright low-entropy fluorescence-like (mostly dark, sparse bright)
    moving = np.clip(rng.exponential(2000.0, size=(10, 32, 32)), 0, 65535).astype(np.uint16)
    # Template: broad MRI-like distribution centred mid-range
    template = np.clip(rng.normal(30000.0, 8000.0, size=(10, 32, 32)), 0, 65535).astype(
        np.uint16
    )

    before_gap = abs(float(np.percentile(moving, 50)) - float(np.percentile(template, 50)))
    adapted = histogram_match_to_template(moving, template)
    after_gap = abs(float(np.percentile(adapted, 50)) - float(np.percentile(template, 50)))

    assert adapted.shape == moving.shape
    assert adapted.dtype == moving.dtype
    assert after_gap < before_gap * 0.2, (
        f"histogram match did not pull p50 toward reference: before={before_gap:.0f}, "
        f"after={after_gap:.0f}"
    )


def test_clahe_3d_preserves_shape_and_lifts_low_contrast_slices():
    """Per-slice CLAHE stacked as 3D must keep shape and measurably raise
    per-slice standard deviation for low-contrast inputs.
    """
    from project.scripts.intensity_adapter import clahe_3d

    # Low-contrast input: gradient-like with small dynamic range
    shape = (8, 64, 64)
    base = np.linspace(1000, 1200, shape[1] * shape[2]).reshape(shape[1], shape[2])
    vol = np.broadcast_to(base[None, ...], shape).astype(np.uint16)

    out = clahe_3d(vol, kernel_size=16, clip_limit=0.03)
    assert out.shape == vol.shape
    assert out.dtype == np.uint16

    # Std across pixels per slice should increase after CLAHE
    before_std = float(np.std(vol[0]))
    after_std = float(np.std(out[0]))
    assert after_std > before_std * 2.0, (
        f"CLAHE did not increase contrast: before_std={before_std:.1f}, "
        f"after_std={after_std:.1f}"
    )


def test_adapt_intensity_off_is_noop():
    """With mode='off' the adapter returns the moving volume unchanged."""
    from project.scripts.intensity_adapter import adapt_intensity

    moving = np.full((4, 8, 8), 30000, dtype=np.uint16)
    template = np.full((4, 8, 8), 50000, dtype=np.uint16)
    out = adapt_intensity(moving, template, mode="off")

    assert out is moving or np.array_equal(out, moving), "mode='off' must not modify data"


def test_adapt_intensity_hist_match_changes_distribution():
    """mode='hist_match' must produce a different distribution than input."""
    from project.scripts.intensity_adapter import adapt_intensity

    rng = np.random.default_rng(7)
    moving = np.clip(rng.exponential(1500.0, size=(6, 24, 24)), 0, 65535).astype(np.uint16)
    template = np.clip(rng.normal(40000.0, 6000.0, size=(6, 24, 24)), 0, 65535).astype(
        np.uint16
    )

    out = adapt_intensity(moving, template, mode="hist_match")
    assert out.shape == moving.shape
    assert out.dtype == np.uint16
    assert not np.array_equal(out, moving)


def test_adapt_intensity_invalid_mode_raises():
    from project.scripts.intensity_adapter import adapt_intensity

    moving = np.zeros((2, 4, 4), dtype=np.uint16)
    template = np.zeros((2, 4, 4), dtype=np.uint16)
    with pytest.raises(ValueError, match="unsupported intensity_adapt.mode"):
        adapt_intensity(moving, template, mode="nonsense")


def test_adapt_intensity_combined_mode_chains_operations():
    """hist_match+clahe should differ from either alone on a non-trivial input."""
    from project.scripts.intensity_adapter import adapt_intensity

    rng = np.random.default_rng(11)
    moving = np.clip(rng.exponential(3000.0, size=(6, 32, 32)), 0, 65535).astype(np.uint16)
    template = np.clip(rng.normal(35000.0, 5000.0, size=(6, 32, 32)), 0, 65535).astype(
        np.uint16
    )

    only_hist = adapt_intensity(moving, template, mode="hist_match")
    only_clahe = adapt_intensity(moving, template, mode="clahe")
    both = adapt_intensity(moving, template, mode="hist_match+clahe")

    assert both.shape == moving.shape
    assert both.dtype == np.uint16
    # Combined output should differ from either single-stage output
    assert not np.array_equal(both, only_hist)
    assert not np.array_equal(both, only_clahe)
