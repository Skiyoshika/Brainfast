from __future__ import annotations

from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd
from tifffile import imread
from skimage.feature import peak_local_max


def detect_cells_fallback(slice_path: Path, min_distance: int = 8, threshold_abs: float = 200.0) -> pd.DataFrame:
    img = imread(str(slice_path))
    if img.ndim == 3:
        img = img[..., 0]
    coords = peak_local_max(img.astype(np.float32), min_distance=min_distance, threshold_abs=threshold_abs)
    rows = []
    for i, (y, x) in enumerate(coords, 1):
        rows.append({"cell_id": i, "x": float(x), "y": float(y), "score": float(img[y, x]), "detector": "fallback"})
    return pd.DataFrame(rows)


def detect_cells_cellpose(slice_path: Path, model_type: str = "cyto2", diameter: float | None = None) -> pd.DataFrame:
    try:
        from cellpose import models
    except Exception:
        return pd.DataFrame()

    img = imread(str(slice_path))
    if img.ndim == 3:
        img = img[..., 0]
    imgf = img.astype(np.float32)

    model = models.Cellpose(model_type=model_type)
    masks, flows, styles, diams = model.eval(imgf, diameter=diameter, channels=[0, 0])

    ys, xs = np.where(masks > 0)
    if len(xs) == 0:
        return pd.DataFrame(columns=["cell_id", "x", "y", "score", "detector"])

    lab_ids = masks[ys, xs]
    df = pd.DataFrame({"label": lab_ids, "x": xs.astype(float), "y": ys.astype(float)})
    cent = df.groupby("label", as_index=False).mean(numeric_only=True)
    cent["score"] = 1.0
    cent["cell_id"] = range(1, len(cent) + 1)
    cent["detector"] = f"cellpose_{model_type}"
    return cent[["cell_id", "x", "y", "score", "detector"]]


def detect_cells(slice_path: Path, cfg: Dict[str, Any]) -> pd.DataFrame:
    det_cfg = cfg.get("detection", {})
    primary = det_cfg.get("primary_model", "cellpose_cyto2")
    if primary.startswith("cellpose"):
        model_type = "cyto2" if "cyto" in primary else "nuclei"
        diameter = det_cfg.get("cellpose_diameter_um", None)
        cp = detect_cells_cellpose(slice_path, model_type=model_type, diameter=diameter)
        if not cp.empty:
            return cp

    # fallback
    thr = det_cfg.get("fallback_threshold", 200.0)
    return detect_cells_fallback(slice_path, threshold_abs=float(thr))
