"""Phase α intensity adaptation for cross-modality registration.

Purpose: reduce the ANTs MI/CC gap when registering cleared-tissue
fluorescence volumes against the Allen average template, which has an
MRI-like intensity profile. Pre-aligning the moving volume's histogram
and local contrast to the template gives ANTs SyN a better starting
optimum than raw fluorescence-vs-template intensity comparison.

Three building blocks:
    * ``histogram_match_to_template`` — rescales moving intensity CDF onto
      the template's CDF (skimage.exposure.match_histograms).
    * ``clahe_3d`` — contrast-limited adaptive histogram equalisation,
      applied per axial slice and stacked.
    * ``adapt_intensity`` — mode router wrapping the two primitives; the
      pipeline calls this single entry point from whole_brain_3d.py.

See docs/.../2026-04-16-internal-alignment-closed-loop-plan.md (Phase α)
for exit criteria and the closed-loop architecture this slots into.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from skimage import exposure

AdaptMode = Literal["off", "hist_match", "clahe", "hist_match+clahe"]


def histogram_match_to_template(moving: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Return a copy of *moving* whose intensity CDF matches *template*.

    Works on any ndarray shape; uses skimage's reference-CDF matching.
    Output dtype tracks the input dtype.
    """
    if moving.dtype != template.dtype:
        # match_histograms requires matching dtypes for a clean result
        template = template.astype(moving.dtype, copy=False)
    matched = exposure.match_histograms(moving, template, channel_axis=None)
    # skimage may up-cast to float64; cast back to input dtype, clipping to range.
    if np.issubdtype(moving.dtype, np.integer):
        info = np.iinfo(moving.dtype)
        matched = np.clip(matched, info.min, info.max)
    return matched.astype(moving.dtype, copy=False)


def clahe_3d(
    vol: np.ndarray,
    kernel_size: int = 32,
    clip_limit: float = 0.01,
) -> np.ndarray:
    """Per-slice CLAHE stacked as a 3D output.

    Parameters
    ----------
    vol
        3D array ``(Z, Y, X)``. Integer or floating dtype accepted.
    kernel_size
        Square kernel side (pixels) for the adaptive histogram.
    clip_limit
        skimage's normalised clip limit in [0, 1].

    Notes
    -----
    skimage.exposure.equalize_adapthist returns float in [0, 1]. We rescale
    back to the input dtype's full range so downstream stages see consistent
    numeric scale.
    """
    if vol.ndim != 3:
        raise ValueError(f"clahe_3d expects 3D volume, got shape {vol.shape}")

    if np.issubdtype(vol.dtype, np.integer):
        info = np.iinfo(vol.dtype)
        scale_max = float(info.max)
    else:
        scale_max = float(vol.max()) or 1.0

    out = np.empty_like(vol)
    for z in range(vol.shape[0]):
        slc = vol[z]
        # equalize_adapthist wants float in [-1, 1] or uint8/uint16. Normalise
        # to [0, 1] float then pass through; clip_limit + kernel_size handed
        # straight to skimage.
        slc_f = slc.astype(np.float64, copy=False)
        slc_max = float(slc_f.max()) or 1.0
        slc_norm = slc_f / slc_max
        eq = exposure.equalize_adapthist(
            slc_norm,
            kernel_size=kernel_size,
            clip_limit=clip_limit,
        )
        # Rescale back to original dtype range using the slice's own max to
        # preserve dark/bright dynamics without clipping bright structures.
        rescaled = eq * scale_max
        if np.issubdtype(vol.dtype, np.integer):
            rescaled = np.clip(rescaled, info.min, info.max)
        out[z] = rescaled.astype(vol.dtype, copy=False)

    return out


def adapt_intensity(
    moving: np.ndarray,
    template: np.ndarray,
    mode: AdaptMode = "off",
    *,
    clahe_kernel_size: int = 32,
    clahe_clip_limit: float = 0.01,
) -> np.ndarray:
    """Route the moving volume through the selected adaptation mode.

    Modes
    -----
    ``off``
        Identity; returns the input unchanged.
    ``hist_match``
        Match moving's CDF to template's CDF.
    ``clahe``
        Per-slice CLAHE only.
    ``hist_match+clahe``
        Histogram match first, then CLAHE. Useful when the fluorescence
        volume's global dynamic range differs from the template AND local
        contrast is low (typical for cleared tissue).
    """
    if mode == "off":
        return moving
    if mode == "hist_match":
        return histogram_match_to_template(moving, template)
    if mode == "clahe":
        return clahe_3d(moving, kernel_size=clahe_kernel_size, clip_limit=clahe_clip_limit)
    if mode == "hist_match+clahe":
        stage_a = histogram_match_to_template(moving, template)
        return clahe_3d(stage_a, kernel_size=clahe_kernel_size, clip_limit=clahe_clip_limit)
    raise ValueError(
        f"unsupported intensity_adapt.mode: {mode!r}. "
        "Expected one of: off, hist_match, clahe, hist_match+clahe."
    )
