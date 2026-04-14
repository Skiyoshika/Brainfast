from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.main import _normalize_sampling_mode, _resolve_processing_files


class MainSamplingTests(unittest.TestCase):
    def test_normalize_sampling_mode_defaults_to_single(self) -> None:
        mode, interval = _normalize_sampling_mode({"input": {}})
        self.assertEqual(mode, "single")
        self.assertEqual(interval, 1)

    def test_normalize_sampling_mode_honors_merge_interval(self) -> None:
        mode, interval = _normalize_sampling_mode(
            {"input": {"sampling_mode": "merge5", "slice_interval_n": 5}}
        )
        self.assertEqual(mode, "merged")
        self.assertEqual(interval, 5)

    def test_resolve_processing_files_keeps_native_slices_for_single_mode(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            files = [Path(td) / "a.tif", Path(td) / "b.tif"]
            out, sampling = _resolve_processing_files(
                {"input": {"sampling_mode": "single", "slice_interval_n": 5}},
                files,
                Path(td),
            )
        self.assertEqual(out, files)
        self.assertEqual(sampling["sampling_mode"], "single")
        self.assertEqual(sampling["slice_interval_n"], 1)

    def test_resolve_processing_files_merges_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            files = [Path(td) / "a.tif", Path(td) / "b.tif"]
            merged = [Path(td) / "merged_0000.tif"]
            with patch("scripts.main.merge_every_n_slices", return_value=merged) as merge_mock:
                out, sampling = _resolve_processing_files(
                    {"input": {"sampling_mode": "merge", "slice_interval_n": 3}},
                    files,
                    Path(td),
                )
        self.assertEqual(out, merged)
        self.assertEqual(sampling["sampling_mode"], "merged")
        self.assertEqual(sampling["slice_interval_n"], 3)
        merge_mock.assert_called_once()


if __name__ == "__main__":
    unittest.main()
