"""Phase β "close the loop": push user landmark corrections downstream.

After the user hits **Apply 3D warp** in the 3D Liquify tab Brainfast writes
``annotation_refined_liquify3d.nii.gz`` — but the final cell-count-per-region
bar chart continues to read the *pre-liquify* annotation until we explicitly
re-map cells against the refined labels.

``finalize_liquify_to_cell_counts`` does exactly that:

1. Re-exports per-slice registered-label TIFs from the refined annotation
   into ``<outputs_dir>/truth_export_liquify3d/``.
2. Re-maps every deduped cell's ``(slice_id, x, y)`` to a region id by
   reading the new per-slice label tif.
3. Re-aggregates to leaf + hierarchy CSVs suffixed ``_liquify3d`` so the
   original quantification (which the Results tab currently displays) is
   preserved for diffing.

Everything runs on the existing ``map_cells_with_registered_label_slice`` +
``aggregate_by_region`` helpers — we do **not** invent a new mapping
algorithm. The only change is the annotation source and output filenames.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from scripts.atlas_mapper import map_cells_with_registered_label_slice
    from scripts.map_and_aggregate import aggregate_by_region
    from scripts.truth_export_3d import export_registered_truth_slices
except ImportError:  # pragma: no cover — pipeline context
    from atlas_mapper import map_cells_with_registered_label_slice
    from map_and_aggregate import aggregate_by_region
    from truth_export_3d import export_registered_truth_slices


_REFINED_NAME = "annotation_refined_liquify3d.nii.gz"
_TRUTH_DIR_NAME = "truth_export_liquify3d"
_CELLS_OUT = "cells_mapped_liquify3d.csv"
_LEAF_OUT = "cell_counts_leaf_liquify3d.csv"
_HIERARCHY_OUT = "cell_counts_hierarchy_liquify3d.csv"


def _require_path(p: Path, kind: str) -> Path:
    if not p.exists():
        raise FileNotFoundError(f"{kind} not found: {p}")
    return p


def finalize_liquify_to_cell_counts(
    outputs_dir: Path | str,
    real_slice_paths: list[Path],
    cells_csv: Path | str,
    *,
    pixel_size_um: float,
    structure_csv: Path | str | None = None,
    slicing_plane: str = "coronal",
    atlas_hemisphere: str = "",
    warp_params: dict | None = None,
    fit_mode: str = "cover",
    edge_smooth_iter: int = 0,
    progress_cb=None,
) -> dict:
    """Re-run cell→region mapping + aggregation against the refined annotation.

    Parameters
    ----------
    outputs_dir
        Pipeline run output directory. Must contain
        ``annotation_refined_liquify3d.nii.gz`` in its root.
    real_slice_paths
        The same list of moving-volume slices the original truth-export was
        given, in source z order. Used to re-export per-slice label TIFs.
    cells_csv
        Deduped cells CSV with at least ``slice_id``, ``x``, ``y`` columns.
        Typical inputs: ``cells_dedup.csv`` or the existing
        ``cells_mapped.csv``.
    pixel_size_um
        XY pixel size (µm) — passed through to ``export_registered_truth_slices``.
    structure_csv
        Optional path to the Allen CCF structure table (CSV or JSON).
        Without this the hierarchy CSV will still be written but will not
        contain parent-depth metadata.
    slicing_plane
        ``"coronal"`` (default), ``"sagittal"``, or ``"horizontal"``.
    atlas_hemisphere
        Passed through to the truth-exporter.

    Returns
    -------
    dict
        ``{
            ok, refined_annotation_path, truth_export_dir,
            cells_mapped_csv, cell_counts_leaf_csv, cell_counts_hierarchy_csv,
            mapped_count,
        }``

    Raises
    ------
    FileNotFoundError
        If the refined annotation or cells CSV is missing.
    """
    outputs_dir = Path(outputs_dir)
    cells_csv = Path(cells_csv)

    def _emit(stage: str, idx: int, pct: int, msg: str) -> None:
        if progress_cb is not None:
            try:
                progress_cb(stage, idx, 3, pct, msg)
            except Exception:  # noqa: BLE001 — best-effort
                pass

    _emit("load_inputs", 0, 2, "Locating refined annotation + cells CSV")
    refined_path = _require_path(outputs_dir / _REFINED_NAME, "annotation_refined_liquify3d")
    _require_path(cells_csv, "cells CSV")

    truth_dir = outputs_dir / _TRUTH_DIR_NAME
    truth_dir.mkdir(parents=True, exist_ok=True)

    _emit(
        "export_truth",
        1,
        10,
        f"Re-exporting {len(real_slice_paths)} truth slice(s) from refined annotation",
    )
    truth_rows = export_registered_truth_slices(
        real_slice_paths=real_slice_paths,
        annotation_volume_path=refined_path,
        out_dir=truth_dir,
        pixel_size_um=float(pixel_size_um),
        slicing_plane=slicing_plane,
        atlas_hemisphere=atlas_hemisphere,
        warp_params=dict(warp_params or {}),
        fit_mode=str(fit_mode),
        edge_smooth_iter=int(edge_smooth_iter),
    )

    # Build a slice_id → registered_label_path lookup so we can map each cell
    # to its correct slice's TIF without re-deriving filenames.
    label_by_slice: dict[int, Path] = {}
    for row in truth_rows:
        label_by_slice[int(row["slice_id"])] = Path(row["registered_label_path"])

    # 2. Re-map every cell against the refined label tifs.
    cells = pd.read_csv(cells_csv)
    # Drop stale mapping columns — we re-derive from the refined labels.
    for col in (
        "region_id",
        "region_name",
        "acronym",
        "parent_structure_id",
        "parent_name",
        "depth",
        "graph_order",
        "structure_id_path",
        "color_hex_triplet",
        "structure_source",
        "hemisphere",
        "hemisphere_id",
        "mapping_status",
        "registered_label_path",
        "atlas_slice_index",
        "registration_method",
        "registration_score",
    ):
        if col in cells.columns:
            cells = cells.drop(columns=[col])

    _emit("map_cells", 2, 60, f"Re-mapping {len(cells)} cells onto refined labels")
    if cells.empty:
        mapped = cells.assign(region_id=[]).copy()
    else:
        per_slice_mapped: list[pd.DataFrame] = []
        for sid, group in cells.groupby("slice_id", sort=True):
            label_path = label_by_slice.get(int(sid))
            if label_path is None or not label_path.exists():
                # Slice missing from the refined truth export — leave cells
                # with region_id=0 so they show up as OUTSIDE_ATLAS rather
                # than silently discarded.
                g = group.copy()
                g["region_id"] = 0
                g["mapping_status"] = "missing_refined_label"
                per_slice_mapped.append(g)
                continue
            mapped_group = (
                map_cells_with_registered_label_slice(
                    group,
                    registered_label_tif=label_path,
                    structure_csv=Path(structure_csv)
                    if structure_csv
                    else Path("/nonexistent.csv"),
                    atlas_slice_index=int(sid),
                    slicing_plane=slicing_plane,
                    registration_method="liquify3d_finalize",
                )
                if structure_csv
                else _map_without_structure(group, label_path, sid)
            )
            per_slice_mapped.append(mapped_group)
        mapped = pd.concat(per_slice_mapped, ignore_index=True)

    cells_out = outputs_dir / _CELLS_OUT
    mapped.to_csv(cells_out, index=False)

    # 3. Re-aggregate.
    _emit("aggregate", 3, 90, "Aggregating region counts")
    leaf, hierarchy = aggregate_by_region(mapped)
    leaf_out = outputs_dir / _LEAF_OUT
    hierarchy_out = outputs_dir / _HIERARCHY_OUT
    leaf.to_csv(leaf_out, index=False)
    hierarchy.to_csv(hierarchy_out, index=False)
    _emit("done", 3, 100, f"Hierarchy written: {_HIERARCHY_OUT}")

    return {
        "ok": True,
        "refined_annotation_path": str(refined_path),
        "truth_export_dir": str(truth_dir),
        "cells_mapped_csv": str(cells_out),
        "cell_counts_leaf_csv": str(leaf_out),
        "cell_counts_hierarchy_csv": str(hierarchy_out),
        "mapped_count": int(len(mapped)),
    }


def _map_without_structure(cells: pd.DataFrame, label_path: Path, slice_id: int) -> pd.DataFrame:
    """Minimal region-id lookup when no structure CSV is available.

    Used by tests and by pipelines that haven't shipped the Allen structure
    ontology. We still need region_id for aggregation; downstream Results
    rendering will show ``region_id`` numerics without region names.
    """
    import numpy as _np
    from tifffile import imread as _tiff_read

    label = _tiff_read(str(label_path))
    if label.ndim == 3:
        label = label[..., 0]
    label = label.astype(_np.int32, copy=False)
    h, w = label.shape[:2]

    out = cells.copy()
    region_ids: list[int] = []
    for _, row in out.iterrows():
        x = int(round(float(row["x"])))
        y = int(round(float(row["y"])))
        if 0 <= x < w and 0 <= y < h:
            region_ids.append(int(label[y, x]))
        else:
            region_ids.append(0)
    out["region_id"] = region_ids
    out["region_name"] = out["region_id"].apply(lambda r: "OUTSIDE_ATLAS" if r == 0 else f"RID_{r}")
    out["acronym"] = out["region_id"].apply(lambda r: "OUT" if r == 0 else f"RID{r}")
    out["hemisphere"] = "unknown"
    out["mapping_status"] = ["ok" if r > 0 else "outside_slice_bounds" for r in region_ids]
    out["atlas_slice_index"] = int(slice_id)
    out["registration_method"] = "liquify3d_finalize_no_structure"
    out["registered_label_path"] = str(Path(label_path).resolve())
    return out
