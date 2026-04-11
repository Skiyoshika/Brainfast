from __future__ import annotations

from pathlib import Path

import numpy as np
from tifffile import imread, imwrite


def _conform_shapes(imgs: list[np.ndarray]) -> list[np.ndarray]:
    """Pad or crop images so they all share the same shape (use the max per axis)."""
    if not imgs:
        return imgs
    # Normalise to 2-D (take first channel / page when multi-channel)
    norm: list[np.ndarray] = []
    for im in imgs:
        if im.ndim == 3:
            # (C, H, W) or (H, W, C) — pick the first channel
            if im.shape[0] <= 4:  # likely (C, H, W)
                im = im[0]
            else:  # likely (H, W, C)
                im = im[..., 0]
        norm.append(im)

    target_shape = tuple(max(im.shape[ax] for im in norm) for ax in range(norm[0].ndim))
    out: list[np.ndarray] = []
    for im in norm:
        if im.shape == target_shape:
            out.append(im)
        else:
            padded = np.zeros(target_shape, dtype=im.dtype)
            slices = tuple(slice(0, min(s, t)) for s, t in zip(im.shape, target_shape, strict=True))
            padded[slices] = im[slices]
            out.append(padded)
    return out


def merge_every_n_slices(input_files: list[Path], out_dir: Path, n: int = 5) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for i in range(0, len(input_files), n):
        chunk = input_files[i : i + n]
        if not chunk:
            continue
        imgs = [imread(str(p)).astype(np.float32) for p in chunk]
        imgs = _conform_shapes(imgs)
        merged = np.mean(np.stack(imgs, axis=0), axis=0)
        out = out_dir / f"merged_{i // n:04d}.tif"
        imwrite(str(out), merged.astype(np.uint16))
        outputs.append(out)
    return outputs
