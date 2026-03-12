from __future__ import annotations

from pathlib import Path
import pandas as pd


def map_cells_to_regions(cells_csv: Path, atlas_map_csv: Path) -> pd.DataFrame:
    """
    MVP placeholder mapping by nearest slice-level lookup.
    Expected cells_csv cols: cell_id,slice_id,x,y,score
    Expected atlas_map_csv cols: slice_id,region_id,region_name,hemisphere
    """
    cells = pd.read_csv(cells_csv)
    atlas = pd.read_csv(atlas_map_csv)
    merged = cells.merge(atlas, on="slice_id", how="left")
    merged["region_id"] = merged["region_id"].fillna(0).astype(int)
    merged["region_name"] = merged["region_name"].fillna("OUTSIDE")
    merged["hemisphere"] = merged["hemisphere"].fillna("unknown")
    return merged


def aggregate_by_region(mapped: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    leaf = (
        mapped.groupby(["region_id", "region_name", "hemisphere"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    # Simplified hierarchy placeholder: aggregate by region_name prefix
    mapped = mapped.copy()
    mapped["parent_region"] = mapped["region_name"].map(
        lambda x: x.split("/")[0] if isinstance(x, str) and "/" in x else x
    )
    hierarchy = (
        mapped.groupby(["parent_region", "hemisphere"], as_index=False)
        .size()
        .rename(columns={"size": "count"})
    )

    def confidence(n: int) -> str:
        return "high" if n > 100 else ("medium" if n > 20 else "low")

    leaf["confidence"] = leaf["count"].map(confidence)
    hierarchy["confidence"] = hierarchy["count"].map(confidence)
    return leaf, hierarchy


def write_outputs(leaf: pd.DataFrame, hierarchy: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    leaf.to_csv(out_dir / "cell_counts_leaf.csv", index=False)
    hierarchy.to_csv(out_dir / "cell_counts_hierarchy.csv", index=False)
