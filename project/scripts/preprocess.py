from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
from tifffile import imread, imwrite


def merge_every_n_slices(input_files: List[Path], out_dir: Path, n: int = 5) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: List[Path] = []
    for i in range(0, len(input_files), n):
        chunk = input_files[i : i + n]
        if not chunk:
            continue
        imgs = [imread(str(p)).astype(np.float32) for p in chunk]
        merged = np.mean(np.stack(imgs, axis=0), axis=0)
        out = out_dir / f"merged_{i//n:04d}.tif"
        imwrite(str(out), merged.astype(np.uint16))
        outputs.append(out)
    return outputs
