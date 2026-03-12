from __future__ import annotations

from pathlib import Path
import numpy as np
from skimage.registration import phase_cross_correlation
from scipy.ndimage import shift as ndi_shift
from tifffile import imread, imwrite


def estimate_translation(fixed_img: np.ndarray, moving_img: np.ndarray) -> tuple[float, float]:
    shift, _, _ = phase_cross_correlation(fixed_img.astype(np.float32), moving_img.astype(np.float32), upsample_factor=10)
    return float(shift[0]), float(shift[1])


def apply_translation(moving_img: np.ndarray, dy: float, dx: float) -> np.ndarray:
    return ndi_shift(moving_img.astype(np.float32), shift=(dy, dx), order=1, mode="nearest")


def run_baseline_alignment(fixed_path: Path, moving_path: Path, out_path: Path) -> dict:
    fixed = imread(str(fixed_path))
    moving = imread(str(moving_path))
    if fixed.ndim == 3:
        fixed = fixed[..., 0]
    if moving.ndim == 3:
        moving = moving[..., 0]

    dy, dx = estimate_translation(fixed, moving)
    aligned = apply_translation(moving, dy, dx)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imwrite(str(out_path), np.clip(aligned, 0, 65535).astype(np.uint16))
    return {"dy": dy, "dx": dx, "aligned_path": str(out_path)}
