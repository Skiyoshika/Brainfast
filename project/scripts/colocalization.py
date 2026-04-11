"""colocalization.py — Reporter / marker co-localization for AAV toolbox analysis.

Computes per-cell and per-region co-localization statistics:
  - dTom+ (reporter-positive):  all cells in cells_mapped.csv
  - marker+:                    marker channel pixel intensity > threshold at cell centroid
  - double+ (dTom+ & marker+):  intersection of above

Per-region aggregations:
  specificity = double_pos_count / dtom_pos_count   (how specific is the AAV to the target type)
  sensitivity = double_pos_count / marker_pos_count (what fraction of target cells are hit)

Outputs:
  cells_colocalization.csv       — per-cell table with dtom_pos, marker_pos, double_pos columns
  colocalization_summary.csv     — per-region aggregation with specificity / sensitivity
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from tifffile import imread


def _read_gray_slice(path: Path) -> np.ndarray:
    img = imread(str(path))
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.float32, copy=False)


def _intensity_threshold(img: np.ndarray, threshold_pct: float) -> float:
    """Return absolute intensity at the given percentile of the image."""
    return float(np.percentile(img, float(threshold_pct)))


def run_colocalization(
    cells_mapped_csv: Path,
    marker_tifs: Mapping[int, Path],
    out_dir: Path,
    marker_intensity_threshold_pct: float = 95.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run colocalization analysis.

    Parameters
    ----------
    cells_mapped_csv:
        Path to cells_mapped.csv.  Required columns: cell_id, slice_id, x, y, region_id.
    marker_tifs:
        Mapping from slice_id (int) to the corresponding marker-channel TIF path.
    out_dir:
        Directory to write output CSVs.
    marker_intensity_threshold_pct:
        Percentile of the marker image used as the positivity threshold (default 95).

    Returns
    -------
    (cell_df, summary_df): per-cell table and per-region summary DataFrame.
    """
    cells = pd.read_csv(cells_mapped_csv)

    # Validate minimum columns
    for col in ("cell_id", "slice_id", "x", "y", "region_id"):
        if col not in cells.columns:
            raise ValueError(f"cells_mapped_csv missing required column: {col}")

    # --- Per-cell marker classification ---
    marker_pos_flags: list[bool] = []
    # Cache loaded images to avoid re-reading the same slice repeatedly.
    _img_cache: dict[int, np.ndarray] = {}
    _thr_cache: dict[int, float] = {}

    for _, row in cells.iterrows():
        sid = int(row["slice_id"])
        tif_path = marker_tifs.get(sid)
        if tif_path is None or not Path(tif_path).exists():
            marker_pos_flags.append(False)
            continue

        if sid not in _img_cache:
            img = _read_gray_slice(Path(tif_path))
            _img_cache[sid] = img
            _thr_cache[sid] = _intensity_threshold(img, marker_intensity_threshold_pct)

        img = _img_cache[sid]
        thr = _thr_cache[sid]
        h, w = img.shape[:2]
        px = int(round(float(row["x"])))
        py = int(round(float(row["y"])))
        if 0 <= px < w and 0 <= py < h:
            marker_pos_flags.append(bool(img[py, px] >= thr))
        else:
            marker_pos_flags.append(False)

    cell_df = cells.copy()
    cell_df["dtom_pos"] = True  # all cells in cells_mapped are reporter-positive by definition
    cell_df["marker_pos"] = marker_pos_flags
    cell_df["double_pos"] = cell_df["marker_pos"]  # dtom_pos & marker_pos

    # --- Per-region summary ---
    # Group by region; also count marker+ cells (those in region regardless of dtom status).
    # Note: cells_mapped only has dtom+ cells, so marker+ total in the *region* cannot be
    # derived from this table alone.  We compute what we can and flag the limitation.
    group_cols = ["region_id"]
    meta_cols = [c for c in ("region_name", "acronym", "hemisphere") if c in cell_df.columns]

    agg = cell_df.groupby(group_cols, as_index=False).agg(
        dtom_pos_count=("dtom_pos", "sum"),
        marker_pos_count=("marker_pos", "sum"),
        double_pos_count=("double_pos", "sum"),
    )
    if meta_cols:
        meta = cell_df.groupby(group_cols, as_index=False)[meta_cols].first()
        agg = agg.merge(meta, on=group_cols, how="left")

    agg["specificity"] = agg.apply(
        lambda r: (
            (float(r["double_pos_count"]) / float(r["dtom_pos_count"]))
            if float(r["dtom_pos_count"]) > 0
            else float("nan")
        ),
        axis=1,
    )
    # sensitivity requires total marker+ in region (including non-dtom cells).
    # With only the dtom channel analysed here, marker_pos_count == double_pos_count,
    # so sensitivity cannot be fully computed from this table.  We leave it NaN and note it.
    agg["sensitivity"] = float("nan")
    agg["sensitivity_note"] = (
        "requires marker+ count from all cells in region (not only dTom+ cells)"
    )

    summary_cols = [
        "region_id",
        *(c for c in ("region_name", "acronym", "hemisphere") if c in agg.columns),
        "dtom_pos_count",
        "marker_pos_count",
        "double_pos_count",
        "specificity",
        "sensitivity",
        "sensitivity_note",
    ]
    summary_df = agg[[c for c in summary_cols if c in agg.columns]].copy()
    summary_df = summary_df.sort_values("dtom_pos_count", ascending=False).reset_index(drop=True)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cell_out_cols = [
        "cell_id",
        "slice_id",
        "x",
        "y",
        "region_id",
        *(c for c in ("region_name", "acronym", "hemisphere") if c in cell_df.columns),
        "dtom_pos",
        "marker_pos",
        "double_pos",
    ]
    cell_df[[c for c in cell_out_cols if c in cell_df.columns]].to_csv(
        out_dir / "cells_colocalization.csv", index=False
    )
    summary_df.to_csv(out_dir / "colocalization_summary.csv", index=False)

    return cell_df, summary_df
