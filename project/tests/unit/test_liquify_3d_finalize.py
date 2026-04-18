"""Unit tests for Phase β "close the loop" — propagating user landmark
corrections into the final cell-count-per-region hierarchy.

The finalize step re-exports truth slices from the user-refined annotation
and re-runs cell→region mapping + aggregation so the bar chart downstream
reflects the corrections, rather than silently using the pre-liquify
annotation as it does today.
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from tifffile import imwrite


def _seed_run_dir(tmp_path: Path) -> dict:
    """Build a fake pipeline outputs dir with everything finalize needs:

    * ``ants_registration/annotation_registered.nii.gz`` (original, all label=1)
    * ``annotation_refined_liquify3d.nii.gz`` (refined, all label=2)
    * ``tmp_merged/`` real slices referenced by truth-export
    * ``cells_dedup.csv`` with 3 cells (slice_id + x + y)
    * ``configs`` dummy structure CSV so mapping merges without crashing
    """
    outputs_dir = tmp_path / "run"
    outputs_dir.mkdir()

    shape = (4, 6, 6)  # small enough to finish in milliseconds
    ann_orig = np.ones(shape, dtype=np.int32)
    ann_refined = np.full(shape, 2, dtype=np.int32)

    ants_dir = outputs_dir / "ants_registration"
    ants_dir.mkdir()
    nib.save(
        nib.Nifti1Image(ann_orig, np.eye(4)),
        str(ants_dir / "annotation_registered.nii.gz"),
    )
    nib.save(
        nib.Nifti1Image(ann_refined, np.eye(4)),
        str(outputs_dir / "annotation_refined_liquify3d.nii.gz"),
    )

    # Real slice TIFs matching the annotation z dimension
    merged_dir = outputs_dir / "tmp_merged"
    merged_dir.mkdir()
    real_slice_paths = []
    for z in range(shape[0]):
        p = merged_dir / f"merged_{z:04d}.tif"
        imwrite(str(p), np.full((6, 6), 100, dtype=np.uint16))
        real_slice_paths.append(p)

    # Deduped cells with 3 rows — all in slice 1 for determinism
    cells = pd.DataFrame(
        {
            "cell_id": [0, 1, 2],
            "slice_id": [1, 1, 1],
            "x": [2.0, 3.0, 4.0],
            "y": [2.0, 3.0, 4.0],
        }
    )
    cells.to_csv(outputs_dir / "cells_dedup.csv", index=False)

    # Minimal atlas map CSV for the fallback branch (not required when
    # per-pixel mapping succeeds, but finalize() passes it through anyway).
    atlas_map_csv = outputs_dir / "atlas_map.csv"
    pd.DataFrame({"slice_id": [0, 1, 2, 3]}).to_csv(atlas_map_csv, index=False)

    # Minimal Allen-style structure CSV so aggregate_by_region can populate
    # the hierarchy CSV with a row for our synthetic label_id=2. Without
    # this the hierarchy comes back empty and downstream tests cannot
    # verify the counts propagated.
    structure_csv = outputs_dir / "structure.csv"
    pd.DataFrame(
        [
            {
                "id": 2,
                "name": "FakeRegion2",
                "acronym": "FAKE2",
                "parent_structure_id": 0,
                "structure_id_path": "/0/2/",
                "color_hex_triplet": "ff0000",
                "depth": 1,
                "graph_order": 1,
                "hemisphere_id": 3,
            },
            {
                "id": 1,
                "name": "FakeRegion1",
                "acronym": "FAKE1",
                "parent_structure_id": 0,
                "structure_id_path": "/0/1/",
                "color_hex_triplet": "00ff00",
                "depth": 1,
                "graph_order": 2,
                "hemisphere_id": 3,
            },
        ]
    ).to_csv(structure_csv, index=False)

    return {
        "outputs_dir": outputs_dir,
        "real_slice_paths": real_slice_paths,
        "cells_csv": outputs_dir / "cells_dedup.csv",
        "atlas_map_csv": atlas_map_csv,
        "structure_csv": structure_csv,
    }


def test_finalize_writes_liquify3d_artifacts(tmp_path):
    from project.scripts.liquify_3d_finalize import finalize_liquify_to_cell_counts

    seed = _seed_run_dir(tmp_path)
    result = finalize_liquify_to_cell_counts(
        outputs_dir=seed["outputs_dir"],
        real_slice_paths=seed["real_slice_paths"],
        cells_csv=seed["cells_csv"],
        structure_csv=seed["structure_csv"],
        pixel_size_um=5.0,
    )

    assert result["ok"] is True
    expected_keys = {
        "refined_annotation_path",
        "truth_export_dir",
        "cells_mapped_csv",
        "cell_counts_leaf_csv",
        "cell_counts_hierarchy_csv",
        "mapped_count",
    }
    assert expected_keys <= set(result.keys())

    for key in (
        "truth_export_dir",
        "cells_mapped_csv",
        "cell_counts_leaf_csv",
        "cell_counts_hierarchy_csv",
    ):
        assert Path(result[key]).exists(), f"missing output: {key}"


def test_finalize_uses_refined_annotation_not_original(tmp_path):
    """All seeded cells should end up mapped to label 2 (refined) — **not**
    label 1 (original). If this assertion fails, finalize is silently using
    the pre-liquify annotation and the closed loop is broken."""
    from project.scripts.liquify_3d_finalize import finalize_liquify_to_cell_counts

    seed = _seed_run_dir(tmp_path)
    result = finalize_liquify_to_cell_counts(
        outputs_dir=seed["outputs_dir"],
        real_slice_paths=seed["real_slice_paths"],
        cells_csv=seed["cells_csv"],
        structure_csv=seed["structure_csv"],
        pixel_size_um=5.0,
    )
    mapped = pd.read_csv(result["cells_mapped_csv"])
    # Every cell must have region_id == 2 (the refined annotation's label)
    assert set(mapped["region_id"].tolist()) == {2}, (
        f"finalize did not use refined annotation; got region_ids {set(mapped['region_id'])}"
    )


def test_finalize_hierarchy_counts_reflect_refined_labels(tmp_path):
    from project.scripts.liquify_3d_finalize import finalize_liquify_to_cell_counts

    seed = _seed_run_dir(tmp_path)
    result = finalize_liquify_to_cell_counts(
        outputs_dir=seed["outputs_dir"],
        real_slice_paths=seed["real_slice_paths"],
        cells_csv=seed["cells_csv"],
        structure_csv=seed["structure_csv"],
        pixel_size_um=5.0,
    )
    hierarchy = pd.read_csv(result["cell_counts_hierarchy_csv"])
    # Hierarchy should have a row for region_id=2 with count=3 (all seeded cells)
    match = hierarchy[hierarchy["region_id"] == 2]
    assert not match.empty
    assert int(match["count"].iloc[0]) == 3


def test_finalize_missing_refined_annotation_returns_error(tmp_path):
    from project.scripts.liquify_3d_finalize import finalize_liquify_to_cell_counts

    seed = _seed_run_dir(tmp_path)
    # Remove the refined annotation to trigger the error path
    (seed["outputs_dir"] / "annotation_refined_liquify3d.nii.gz").unlink()

    with pytest.raises(FileNotFoundError, match="annotation_refined_liquify3d"):
        finalize_liquify_to_cell_counts(
            outputs_dir=seed["outputs_dir"],
            real_slice_paths=seed["real_slice_paths"],
            cells_csv=seed["cells_csv"],
            pixel_size_um=5.0,
        )


def test_finalize_missing_cells_csv_returns_error(tmp_path):
    from project.scripts.liquify_3d_finalize import finalize_liquify_to_cell_counts

    seed = _seed_run_dir(tmp_path)
    seed["cells_csv"].unlink()

    with pytest.raises(FileNotFoundError, match="cells"):
        finalize_liquify_to_cell_counts(
            outputs_dir=seed["outputs_dir"],
            real_slice_paths=seed["real_slice_paths"],
            cells_csv=seed["cells_csv"],
            pixel_size_um=5.0,
        )
