"""Unit tests for scripts.z_smoothness."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest


def _rows(zs: list[int], ok: bool = True) -> list[dict]:
    return [{"slice_id": i, "best_z": z, "registration_ok": ok} for i, z in enumerate(zs)]


class TestSmoothApSeries:
    def test_empty_rows(self):
        from scripts.z_smoothness import smooth_ap_series

        report = smooth_ap_series([])
        assert report["outlier_count"] == 0
        assert report["slice_ids"] == []

    def test_single_row(self):
        from scripts.z_smoothness import smooth_ap_series

        report = smooth_ap_series(_rows([250]))
        assert report["outlier_count"] == 0
        assert report["original_z"] == [250]
        assert report["smoothed_z"] == [250]

    def test_smooth_monotone_series(self):
        from scripts.z_smoothness import smooth_ap_series

        # Perfectly monotone series with step=5: rolling median edge effect ≤ step_size/2
        report = smooth_ap_series(_rows([200, 205, 210, 215, 220]))
        assert report["outlier_count"] == 0
        # Edge slices may deviate up to half the step size (rolling median boundary)
        assert all(d <= 5 for d in report["deviation"])

    def test_detects_outlier(self):
        from scripts.z_smoothness import smooth_ap_series

        # One slice jumps 30 voxels from trend
        zs = [200, 205, 210, 240, 220, 225, 230]
        report = smooth_ap_series(_rows(zs), max_dev=8)
        assert report["outlier_count"] >= 1
        assert any(report["is_outlier"])

    def test_no_outlier_within_threshold(self):
        from scripts.z_smoothness import smooth_ap_series

        zs = [200, 205, 210, 215, 222, 225]  # max jump ~7
        report = smooth_ap_series(_rows(zs), max_dev=8)
        assert report["outlier_count"] == 0

    def test_failed_slices_excluded_from_fit(self):
        from scripts.z_smoothness import smooth_ap_series

        rows = [
            {"slice_id": 0, "best_z": 200, "registration_ok": True},
            {"slice_id": 1, "best_z": 999, "registration_ok": False},  # failed
            {"slice_id": 2, "best_z": 210, "registration_ok": True},
        ]
        report = smooth_ap_series(rows, max_dev=8)
        # Failed slice should not cause outlier flagging of valid slices
        assert 0 in report["slice_ids"]
        assert 2 in report["slice_ids"]
        assert report["outlier_count"] == 0

    def test_report_keys_present(self):
        from scripts.z_smoothness import smooth_ap_series

        report = smooth_ap_series(_rows([200, 210, 220]))
        for key in (
            "slice_ids", "original_z", "smoothed_z", "deviation",
            "is_outlier", "outlier_count", "max_deviation", "mean_deviation",
            "monotone", "max_dev_threshold",
        ):
            assert key in report, f"missing key: {key}"

    def test_smoothed_z_clipped(self):
        from scripts.z_smoothness import smooth_ap_series

        # Edge-case: values near atlas boundaries
        report = smooth_ap_series(_rows([0, 0, 0]))
        assert all(0 <= z <= 527 for z in report["smoothed_z"])


class TestWriteSmoothnessReport:
    def test_writes_valid_json(self, tmp_path):
        from scripts.z_smoothness import smooth_ap_series, write_smoothness_report

        report = smooth_ap_series(_rows([200, 210, 220]))
        out = tmp_path / "report.json"
        write_smoothness_report(report, out)
        assert out.exists()
        loaded = json.loads(out.read_text(encoding="utf-8"))
        assert loaded["outlier_count"] == report["outlier_count"]

    def test_creates_parent_dirs(self, tmp_path):
        from scripts.z_smoothness import smooth_ap_series, write_smoothness_report

        report = smooth_ap_series(_rows([200]))
        deep = tmp_path / "a" / "b" / "c" / "report.json"
        write_smoothness_report(report, deep)
        assert deep.exists()
