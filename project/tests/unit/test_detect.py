from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from tifffile import imwrite

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.detect import CellposeDetectionError, detect_cells, detect_cells_cellpose


def _cellpose_cfg(**detection_overrides) -> dict:
    detection = {
        "primary_model": "cellpose_cyto2",
        "secondary_model": "cellpose_nuclei",
        "merge_primary_secondary": False,
        "within_slice_dedup_px": 4.0,
    }
    detection.update(detection_overrides)
    return {"detection": detection, "compute": {"device": "cpu"}, "input": {"pixel_size_um_xy": 1.0}}


class DetectCellsTests(unittest.TestCase):
    def test_requested_cellpose_does_not_silently_fallback_on_runtime_error(self) -> None:
        cfg = _cellpose_cfg()
        with (
            patch("scripts.detect._run_cellpose_by_name", side_effect=CellposeDetectionError("boom")),
            patch("scripts.detect.detect_cells_log_fallback") as log_fallback,
            patch("scripts.detect.detect_cells_fallback") as peak_fallback,
        ):
            with self.assertRaises(CellposeDetectionError):
                detect_cells(Path("slice_0001.tif"), cfg)
        log_fallback.assert_not_called()
        peak_fallback.assert_not_called()

    def test_requested_cellpose_can_return_empty_without_fallback(self) -> None:
        cfg = _cellpose_cfg()
        empty = pd.DataFrame(columns=["cell_id", "x", "y", "score", "detector", "area_px"])
        with (
            patch("scripts.detect._run_cellpose_by_name", return_value=empty),
            patch("scripts.detect.detect_cells_log_fallback") as log_fallback,
            patch("scripts.detect.detect_cells_fallback") as peak_fallback,
        ):
            out = detect_cells(Path("slice_0002.tif"), cfg)
        self.assertTrue(out.empty)
        log_fallback.assert_not_called()
        peak_fallback.assert_not_called()

    def test_explicit_allow_fallback_keeps_legacy_behavior(self) -> None:
        cfg = _cellpose_cfg(allow_fallback=True, fallback_model="log")
        fallback_df = pd.DataFrame(
            [{"cell_id": 1, "x": 10.0, "y": 12.0, "score": 99.0, "detector": "fallback_log", "area_px": 5.0}]
        )
        with (
            patch("scripts.detect._run_cellpose_by_name", side_effect=CellposeDetectionError("boom")),
            patch("scripts.detect.detect_cells_log_fallback", return_value=fallback_df),
        ):
            out = detect_cells(Path("slice_0003.tif"), cfg)
        self.assertEqual(len(out), 1)
        self.assertEqual(str(out.iloc[0]["detector"]), "fallback_log")

    def test_cellpose_external_tiling_offsets_tile_centroids(self) -> None:
        class FakeModel:
            def eval(self, img, **kwargs):
                masks = np.zeros(img.shape[:2], dtype=np.int32)
                masks[-1, -1] = 1
                return masks, None, None, None

        with tempfile.TemporaryDirectory() as td:
            slice_path = Path(td) / "tile_test.tif"
            imwrite(str(slice_path), np.ones((4, 4), dtype=np.uint16))
            with patch("scripts.detect._load_cellpose_model", return_value=FakeModel()):
                out = detect_cells_cellpose(
                    slice_path,
                    model_type="cyto2",
                    use_gpu=False,
                    external_tile_size_px=2,
                    external_tile_overlap_px=0,
                )

        coords = sorted((float(row.x), float(row.y)) for row in out.itertuples(index=False))
        self.assertEqual(len(coords), 4)
        self.assertIn((1.0, 1.0), coords)
        self.assertIn((3.0, 1.0), coords)
        self.assertIn((1.0, 3.0), coords)
        self.assertIn((3.0, 3.0), coords)


if __name__ == "__main__":
    unittest.main()
