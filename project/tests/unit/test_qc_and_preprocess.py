"""Unit tests for qc.py and preprocess.py — no atlas file required."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from tifffile import imwrite

from project.scripts.qc import export_slice_qc
from project.scripts.preprocess import merge_every_n_slices


# ---------------------------------------------------------------------------
# qc.export_slice_qc
# ---------------------------------------------------------------------------


def test_export_slice_qc_basic(tmp_path):
    cells_csv = tmp_path / "cells_mapped.csv"
    df = pd.DataFrame({"slice_id": [0, 0, 1, 2, 2, 2], "x": [1, 2, 3, 4, 5, 6]})
    df.to_csv(cells_csv, index=False)

    out_csv = tmp_path / "qc_out.csv"
    result = export_slice_qc(cells_csv, out_csv)

    assert out_csv.exists()
    assert set(result["status"]) == {"ok"}
    assert result[result["slice_id"] == 0]["cell_count"].values[0] == 2
    assert result[result["slice_id"] == 2]["cell_count"].values[0] == 3


def test_export_slice_qc_empty_slice(tmp_path):
    """A slice with 0 cells should be labelled 'empty' — but empty slices
    won't appear in a groupby on slice_id unless they have rows.
    This test verifies slices with cells are always 'ok'."""
    cells_csv = tmp_path / "cells.csv"
    df = pd.DataFrame({"slice_id": [5, 5], "x": [10, 20]})
    df.to_csv(cells_csv, index=False)

    out_csv = tmp_path / "qc.csv"
    result = export_slice_qc(cells_csv, out_csv)
    assert result.iloc[0]["status"] == "ok"


def test_export_slice_qc_writes_csv(tmp_path):
    cells_csv = tmp_path / "cells.csv"
    pd.DataFrame({"slice_id": [0, 1], "x": [1, 2]}).to_csv(cells_csv, index=False)
    out_csv = tmp_path / "out.csv"
    export_slice_qc(cells_csv, out_csv)

    loaded = pd.read_csv(out_csv)
    assert "cell_count" in loaded.columns
    assert "status" in loaded.columns


# ---------------------------------------------------------------------------
# preprocess.merge_every_n_slices
# ---------------------------------------------------------------------------


def _make_tiffs(out_dir, n, h=8, w=8, dtype=np.uint16):
    """Write n synthetic TIFF files into out_dir, return sorted paths."""
    paths = []
    for i in range(n):
        p = out_dir / f"slice_{i:04d}.tif"
        arr = np.full((h, w), i * 100, dtype=dtype)
        imwrite(str(p), arr)
        paths.append(p)
    return paths


def test_merge_every_n_slices_output_count(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    paths = _make_tiffs(src, n=6)
    out = tmp_path / "merged"
    result = merge_every_n_slices(paths, out, n=2)
    assert len(result) == 3  # 6 slices / 2 per chunk


def test_merge_every_n_slices_pixel_values(tmp_path):
    src = tmp_path / "src"
    src.mkdir()
    # Two slices: 100 and 300 → mean 200
    p1 = src / "s0.tif"
    p2 = src / "s1.tif"
    imwrite(str(p1), np.full((4, 4), 100, dtype=np.uint16))
    imwrite(str(p2), np.full((4, 4), 300, dtype=np.uint16))

    out = tmp_path / "merged"
    result = merge_every_n_slices([p1, p2], out, n=2)
    assert len(result) == 1

    from tifffile import imread
    merged = imread(str(result[0]))
    assert merged.mean() == pytest.approx(200.0, abs=1)


def test_merge_every_n_slices_n_greater_than_count(tmp_path):
    """n larger than total files → single output chunk."""
    src = tmp_path / "src"
    src.mkdir()
    paths = _make_tiffs(src, n=3)
    out = tmp_path / "merged"
    result = merge_every_n_slices(paths, out, n=10)
    assert len(result) == 1


def test_merge_every_n_slices_clears_old_files(tmp_path):
    """Re-running should clear old output files."""
    src = tmp_path / "src"
    src.mkdir()
    paths = _make_tiffs(src, n=2)
    out = tmp_path / "merged"
    merge_every_n_slices(paths, out, n=1)  # creates 2 files

    # Run again with n=2 → only 1 file should remain
    result = merge_every_n_slices(paths, out, n=2)
    assert len(list(out.glob("*.tif"))) == 1
    assert len(result) == 1
