"""paper_aav_summary.py — Paper-style AAV toolbox region summary.

Computes per-region statistics aligned to the AAV_toolbox paper methodology:
  - Representative slice: the slice with the highest cell count for each region
  - Region area (mm²) from the registered label TIF
  - Cell density (cells / mm²)

Output: paper_aav_region_summary.csv
Columns: region_id, region_name, acronym, hemisphere,
         representative_slice_id, count, area_mm2, density_cells_per_mm2
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def generate_paper_aav_summary(
    cells_mapped_csv: Path,
    region_areas_csv: Path,
    out_csv: Path,
) -> pd.DataFrame:
    """Generate paper-style region summary with representative slices and density.

    Parameters
    ----------
    cells_mapped_csv:
        Path to cells_mapped.csv (output of main pipeline dedup step).
        Required columns: slice_id, region_id, region_name, acronym, hemisphere
    region_areas_csv:
        Path to region_areas.csv (output of compute_region_areas_from_label_tif).
        Required columns: slice_id, region_id, area_mm2
    out_csv:
        Destination path for paper_aav_region_summary.csv.

    Returns
    -------
    pd.DataFrame with the summary table (also written to out_csv).
    """
    cells = pd.read_csv(cells_mapped_csv)
    areas = pd.read_csv(region_areas_csv)

    # Require minimum columns
    for col in ("slice_id", "region_id"):
        if col not in cells.columns:
            raise ValueError(f"cells_mapped_csv is missing column: {col}")
    for col in ("slice_id", "region_id", "area_mm2"):
        if col not in areas.columns:
            raise ValueError(f"region_areas_csv is missing column: {col}")

    # --- cell count per (region_id, hemisphere, slice_id) ---
    group_cols = ["region_id", "slice_id"]
    meta_cols = [c for c in ("region_name", "acronym", "hemisphere") if c in cells.columns]
    if meta_cols:
        # carry forward one representative metadata value per group
        meta = cells.groupby(group_cols, as_index=False)[meta_cols].first()
        counts = cells.groupby(group_cols, as_index=False).size().rename(columns={"size": "count"})
        slice_counts = counts.merge(meta, on=group_cols, how="left")
    else:
        slice_counts = (
            cells.groupby(group_cols, as_index=False).size().rename(columns={"size": "count"})
        )

    # --- representative slice: the slice with max count for each region ---
    # For ties, take the smallest slice_id for reproducibility.
    idx = (
        slice_counts.sort_values(["count", "slice_id"], ascending=[False, True])
        .groupby("region_id", as_index=False)
        .first()
    )
    rep = idx.rename(columns={"slice_id": "representative_slice_id"})

    # --- merge area for representative slice ---
    area_rep = areas.rename(columns={"slice_id": "representative_slice_id"})
    rep = rep.merge(
        area_rep[["representative_slice_id", "region_id", "area_mm2"]],
        on=["representative_slice_id", "region_id"],
        how="left",
    )

    # --- density ---
    rep["density_cells_per_mm2"] = rep.apply(
        lambda r: (
            (float(r["count"]) / float(r["area_mm2"]))
            if pd.notna(r.get("area_mm2")) and float(r.get("area_mm2", 0)) > 0
            else float("nan")
        ),
        axis=1,
    )

    # --- output column order ---
    out_cols = [
        "region_id",
        *(c for c in ("region_name", "acronym", "hemisphere") if c in rep.columns),
        "representative_slice_id",
        "count",
        "area_mm2",
        "density_cells_per_mm2",
    ]
    result = rep[[c for c in out_cols if c in rep.columns]].copy()
    result = result.sort_values("count", ascending=False).reset_index(drop=True)

    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_csv, index=False)
    return result
