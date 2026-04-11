from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.structure_tree import load_structure_table, parse_structure_id_path
except Exception:
    from structure_tree import load_structure_table, parse_structure_id_path


def map_cells_to_regions(
    cells_csv: Path,
    atlas_map_csv: Path,
    label_dir: Path | None = None,
    structure_csv: Path | None = None,
) -> pd.DataFrame:
    """Map cells to atlas regions.

    If label_dir is provided and contains per-slice registered label TIFs
    (slice_NNNN_label.tif), uses per-pixel coordinate lookup for precise mapping.
    Otherwise falls back to slice-level join via atlas_map_csv.
    """
    cells = pd.read_csv(cells_csv)

    # Try per-pixel mapping if label directory is available
    if label_dir and Path(label_dir).is_dir():
        mapped = _map_cells_per_pixel(cells, Path(label_dir), structure_csv)
        if mapped is not None:
            return mapped

    # Fallback: slice-level join
    atlas = pd.read_csv(atlas_map_csv)
    merged = cells.merge(atlas, on="slice_id", how="left")
    merged["region_id"] = merged["region_id"].fillna(0).astype(int)
    merged["region_name"] = merged["region_name"].fillna("OUTSIDE")
    merged["hemisphere"] = merged["hemisphere"].fillna("unknown")
    return merged


def _map_cells_per_pixel(
    cells: pd.DataFrame,
    label_dir: Path,
    structure_csv: Path | None,
) -> pd.DataFrame | None:
    """Look up each cell's (x, y) in its slice's registered label TIF."""
    from functools import lru_cache

    from tifffile import imread as _tiff_read

    @lru_cache(maxsize=32)
    def _load_label(path_str: str) -> np.ndarray:
        label = _tiff_read(path_str)
        if label.ndim == 3:
            label = label[..., 0]
        return label.astype(np.int32, copy=False)

    # Check if any label files exist
    label_files = list(label_dir.glob("slice_*_label.tif"))
    if not label_files:
        return None

    region_ids = []
    for _, row in cells.iterrows():
        sid = int(row.get("slice_id", -1))
        label_path = label_dir / f"slice_{sid:04d}_label.tif"
        if label_path.exists():
            label = _load_label(str(label_path))
            x = int(round(float(row["x"])))
            y = int(round(float(row["y"])))
            h, w = label.shape[:2]
            if 0 <= y < h and 0 <= x < w:
                region_ids.append(int(label[y, x]))
            else:
                region_ids.append(0)
        else:
            region_ids.append(0)

    cells = cells.copy()
    cells["region_id"] = region_ids

    # Attach structure metadata if available
    if structure_csv and Path(structure_csv).exists():
        struct = load_structure_table(Path(structure_csv))
        meta_cols = [
            "id",
            "name",
            "acronym",
            "parent_structure_id",
            "depth",
            "graph_order",
            "structure_id_path",
        ]
        available = [c for c in meta_cols if c in struct.columns]
        meta = struct[available].rename(columns={"id": "region_id", "name": "region_name"})
        cells = cells.merge(meta, on="region_id", how="left")
        cells["region_name"] = cells["region_name"].fillna("OUTSIDE")
    else:
        cells["region_name"] = "OUTSIDE"

    cells["hemisphere"] = cells.get("hemisphere", "unknown")
    if "hemisphere" not in cells.columns:
        cells["hemisphere"] = "unknown"
    return cells


def aggregate_by_region(mapped: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if mapped.empty:
        empty_leaf = pd.DataFrame(
            columns=["region_id", "region_name", "acronym", "hemisphere", "count", "confidence"]
        )
        empty_hierarchy = pd.DataFrame(
            columns=[
                "region_id",
                "region_name",
                "acronym",
                "parent_structure_id",
                "hemisphere",
                "depth",
                "count",
                "confidence",
            ]
        )
        return empty_leaf, empty_hierarchy

    total_cells = max(int(len(mapped)), 1)
    leaf = (
        mapped.groupby(["region_id", "region_name", "acronym", "hemisphere"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )
    leaf["confidence"] = leaf["count"].astype(float) / float(total_cells)

    if "structure_source" not in mapped.columns:
        empty_hierarchy = pd.DataFrame(
            columns=[
                "region_id",
                "region_name",
                "acronym",
                "parent_structure_id",
                "hemisphere",
                "depth",
                "count",
                "confidence",
            ]
        )
        return leaf, empty_hierarchy
    structure_sources = [
        x for x in mapped["structure_source"].dropna().astype(str).unique().tolist() if x
    ]
    if not structure_sources:
        empty_hierarchy = pd.DataFrame(
            columns=[
                "region_id",
                "region_name",
                "acronym",
                "parent_structure_id",
                "hemisphere",
                "depth",
                "count",
                "confidence",
            ]
        )
        return leaf, empty_hierarchy

    structure_df = load_structure_table(Path(structure_sources[0]))
    structure_meta = structure_df[
        ["id", "name", "acronym", "parent_structure_id", "depth", "graph_order"]
    ].rename(columns={"id": "region_id", "name": "region_name"})

    # If the loaded structure_df has all-zero depth (flat format), try to use a richer source.
    # Priority: (1) mapped cells' own metadata if valid, merged with structure_df for ancestors;
    #           (2) structure_df as-is.
    _struct_depth_max = int(structure_meta["depth"].max()) if len(structure_meta) else 0
    if _struct_depth_max == 0 and "depth" in mapped.columns and mapped["depth"].notna().any():
        _mapped_depth_max = float(mapped["depth"].dropna().max())
    else:
        _mapped_depth_max = 0.0
    _has_rich_depth = _struct_depth_max == 0 and _mapped_depth_max > 0

    if _has_rich_depth:
        # Build structure_meta from cells (covers leaf regions) + structure_df (covers ancestors)
        meta_cols = [
            "region_id",
            "region_name",
            "acronym",
            "parent_structure_id",
            "depth",
            "graph_order",
            "structure_id_path",
        ]
        meta_cols_present = [c for c in meta_cols if c in mapped.columns]
        from_cells = (
            mapped[mapped["region_id"].notna() & (mapped["region_id"] != 0)][meta_cols_present]
            .drop_duplicates("region_id")
            .copy()
        )
        if "graph_order" not in from_cells.columns:
            from_cells["graph_order"] = from_cells["region_id"]
        if "structure_id_path" not in from_cells.columns:
            from_cells["structure_id_path"] = from_cells["region_id"].map(lambda x: f"/{x}/")
        # structure_df may have ancestor names even if depths are wrong — use its names for merge
        anc_meta = structure_df[
            ["id", "name", "acronym", "parent_structure_id", "graph_order", "structure_id_path"]
        ].rename(columns={"id": "region_id", "name": "region_name"})
        # Combine: prefer from_cells (has correct depth), fill ancestors from anc_meta
        combined = pd.concat([from_cells, anc_meta], ignore_index=True)
        # Drop duplicates keeping from_cells entries (which come first)
        combined = combined.drop_duplicates(subset=["region_id"], keep="first")
        structure_meta = combined

    hierarchy_rows: list[dict] = []
    for row in mapped[["structure_id_path", "hemisphere"]].itertuples(index=False):
        for region_id in parse_structure_id_path(row.structure_id_path):
            hierarchy_rows.append({"region_id": int(region_id), "hemisphere": row.hemisphere})

    hierarchy = pd.DataFrame(hierarchy_rows)
    if hierarchy.empty:
        hierarchy = pd.DataFrame(
            columns=[
                "region_id",
                "region_name",
                "acronym",
                "parent_structure_id",
                "hemisphere",
                "depth",
                "count",
                "confidence",
            ]
        )
    else:
        hierarchy = (
            hierarchy.groupby(["region_id", "hemisphere"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
        )
        merge_cols = [
            c
            for c in [
                "region_id",
                "region_name",
                "acronym",
                "parent_structure_id",
                "depth",
                "graph_order",
                "structure_id_path",
            ]
            if c in structure_meta.columns
        ]
        hierarchy = hierarchy.merge(structure_meta[merge_cols], on="region_id", how="left")
        hierarchy["confidence"] = hierarchy["count"].astype(float) / float(total_cells)
        hierarchy = hierarchy.sort_values(["depth", "region_id", "hemisphere"]).reset_index(
            drop=True
        )

    leaf_merge_cols = [
        c
        for c in ["region_id", "parent_structure_id", "depth", "graph_order"]
        if c in structure_meta.columns
    ]
    leaf = (
        leaf.merge(
            structure_meta[leaf_merge_cols],
            on="region_id",
            how="left",
        )
        .sort_values(["graph_order", "region_id", "hemisphere"])
        .reset_index(drop=True)
    )

    return leaf, hierarchy


def compute_region_areas_from_label_tif(
    label_tif_path: Path,
    slice_id: int,
    pixel_size_um: float,
) -> pd.DataFrame:
    """Count atlas label pixels per region for one registered label slice.

    Returns a DataFrame with columns:
      slice_id, region_id, area_px, area_mm2
    """
    from tifffile import imread

    label = imread(str(label_tif_path))
    if label.ndim == 3:
        label = label[..., 0]
    label = label.astype(np.int32, copy=False)

    px_mm = float(pixel_size_um) / 1000.0
    area_per_px_mm2 = px_mm * px_mm

    unique_ids, counts = np.unique(label, return_counts=True)
    rows = []
    for rid, cnt in zip(unique_ids, counts, strict=True):
        if int(rid) == 0:
            continue
        rows.append(
            {
                "slice_id": int(slice_id),
                "region_id": int(rid),
                "area_px": int(cnt),
                "area_mm2": float(cnt) * area_per_px_mm2,
            }
        )
    return pd.DataFrame(rows, columns=["slice_id", "region_id", "area_px", "area_mm2"])


def write_outputs(leaf: pd.DataFrame, hierarchy: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    leaf.to_csv(out_dir / "cell_counts_leaf.csv", index=False)
    hierarchy.to_csv(out_dir / "cell_counts_hierarchy.csv", index=False)
