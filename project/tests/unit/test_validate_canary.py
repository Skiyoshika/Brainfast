"""Tests for scripts/validate_canary.py — automated canary gate validation."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from project.scripts.validate_canary import (
    GateResult,
    _parse_metrics_csv,
    gate1_no_crash,
    gate2_qc_outputs,
    gate3_registration_quality,
    gate4_truth_coverage,
    gate5_cell_detection,
    run_gates,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_progress(output_dir: Path, percent: int = 100) -> None:
    """Write a pipeline_progress.json with the given percent."""
    (output_dir / "pipeline_progress.json").write_text(
        json.dumps({"percent": percent, "stageName": "Quantification"}),
        encoding="utf-8",
    )


def _write_metrics_csv(output_dir: Path, metrics: dict[str, float]) -> None:
    """Write a tall-format registration_metrics.csv."""
    ants_dir = output_dir / "ants_registration"
    ants_dir.mkdir(parents=True, exist_ok=True)
    csv_path = ants_dir / "registration_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric", "value"])
        writer.writeheader()
        for name, val in metrics.items():
            writer.writerow({"metric": name, "value": val})


def _write_cells_csv(output_dir: Path, n_rows: int = 5) -> None:
    """Write a cells_mapped.csv with n_rows data rows."""
    csv_path = output_dir / "cells_mapped.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["x", "y", "z", "region_id"])
        for i in range(n_rows):
            writer.writerow([i * 10, i * 20, 0, 100 + i])


def _setup_full_canary(output_dir: Path) -> None:
    """Set up a minimal output dir that passes all gates."""
    _write_progress(output_dir, 100)
    _write_metrics_csv(
        output_dir,
        {
            "NCC": 0.5,
            "NMI": 1.2,
            "SSIM": 0.3,
            "Dice": 0.85,
            "MSE": 0.05,
            "PSNR": 13.0,
        },
    )
    truth_dir = output_dir / "truth_export"
    truth_dir.mkdir(parents=True, exist_ok=True)
    (truth_dir / "slice_0000_label.tif").write_bytes(b"fake")
    (output_dir / "volume").mkdir(parents=True, exist_ok=True)
    _write_cells_csv(output_dir, 10)


# ---------------------------------------------------------------------------
# Gate 1: No Hard Crash
# ---------------------------------------------------------------------------


class TestGate1:
    def test_pass_on_percent_100(self, tmp_path: Path) -> None:
        _write_progress(tmp_path, 100)
        result = gate1_no_crash(tmp_path)
        assert result.passed

    def test_fail_on_missing_file(self, tmp_path: Path) -> None:
        result = gate1_no_crash(tmp_path)
        assert not result.passed
        assert "Missing" in result.details

    def test_fail_on_incomplete(self, tmp_path: Path) -> None:
        _write_progress(tmp_path, 50)
        result = gate1_no_crash(tmp_path)
        assert not result.passed
        assert "50" in result.details

    def test_fail_on_corrupt_json(self, tmp_path: Path) -> None:
        (tmp_path / "pipeline_progress.json").write_text("{bad", encoding="utf-8")
        result = gate1_no_crash(tmp_path)
        assert not result.passed


# ---------------------------------------------------------------------------
# Gate 2: Non-Empty QC Outputs
# ---------------------------------------------------------------------------


class TestGate2:
    def test_pass_with_all_artifacts(self, tmp_path: Path) -> None:
        _write_metrics_csv(
            tmp_path,
            {
                "NCC": 0.5,
                "NMI": 1.2,
                "SSIM": 0.3,
                "Dice": 0.85,
                "MSE": 0.05,
                "PSNR": 13.0,
            },
        )
        truth = tmp_path / "truth_export"
        truth.mkdir()
        (truth / "slice.tif").write_bytes(b"x")
        (tmp_path / "volume").mkdir()
        result = gate2_qc_outputs(tmp_path)
        assert result.passed

    def test_fail_missing_metrics(self, tmp_path: Path) -> None:
        (tmp_path / "truth_export").mkdir()
        (tmp_path / "truth_export" / "s.tif").write_bytes(b"x")
        (tmp_path / "volume").mkdir()
        result = gate2_qc_outputs(tmp_path)
        assert not result.passed
        assert "registration_metrics.csv" in result.details

    def test_fail_empty_truth(self, tmp_path: Path) -> None:
        _write_metrics_csv(
            tmp_path,
            {
                "NCC": 0.5,
                "NMI": 1.2,
                "SSIM": 0.3,
                "Dice": 0.85,
                "MSE": 0.05,
                "PSNR": 13.0,
            },
        )
        (tmp_path / "truth_export").mkdir()
        (tmp_path / "volume").mkdir()
        result = gate2_qc_outputs(tmp_path)
        assert not result.passed
        assert "empty" in result.details


# ---------------------------------------------------------------------------
# Gate 3: Registration Quality
# ---------------------------------------------------------------------------


class TestGate3:
    def test_pass_canary_a(self, tmp_path: Path) -> None:
        _write_metrics_csv(
            tmp_path,
            {
                "NCC": 0.5,
                "NMI": 1.2,
                "SSIM": 0.3,
                "Dice": 0.85,
                "MSE": 0.05,
                "PSNR": 13.0,
            },
        )
        result = gate3_registration_quality(tmp_path, "A")
        assert result.passed

    def test_fail_low_dice_canary_a(self, tmp_path: Path) -> None:
        _write_metrics_csv(
            tmp_path,
            {
                "NCC": 0.5,
                "NMI": 1.2,
                "SSIM": 0.3,
                "Dice": 0.20,
                "MSE": 0.05,
                "PSNR": 13.0,
            },
        )
        result = gate3_registration_quality(tmp_path, "A")
        assert not result.passed  # A threshold is 0.30
        assert "Dice" in result.details

    def test_pass_relaxed_dice_canary_b(self, tmp_path: Path) -> None:
        _write_metrics_csv(
            tmp_path,
            {
                "NCC": 0.5,
                "NMI": 1.2,
                "SSIM": 0.3,
                "Dice": 0.55,
                "MSE": 0.05,
                "PSNR": 13.0,
            },
        )
        result = gate3_registration_quality(tmp_path, "B")
        assert result.passed  # B threshold is 0.01 (sparse 5-slice volume)

    def test_fail_nan_metric(self, tmp_path: Path) -> None:
        _write_metrics_csv(
            tmp_path,
            {
                "NCC": 0.5,
                "NMI": float("nan"),
                "SSIM": 0.3,
                "Dice": 0.85,
                "MSE": 0.05,
                "PSNR": 13.0,
            },
        )
        result = gate3_registration_quality(tmp_path, "A")
        assert not result.passed
        assert "NaN" in result.details


# ---------------------------------------------------------------------------
# _parse_metrics_csv
# ---------------------------------------------------------------------------


class TestParseMetricsCsv:
    def test_tall_format(self, tmp_path: Path) -> None:
        _write_metrics_csv(tmp_path, {"Dice": 0.9, "NMI": 1.1})
        csv_path = tmp_path / "ants_registration" / "registration_metrics.csv"
        metrics = _parse_metrics_csv(csv_path)
        assert abs(metrics["Dice"] - 0.9) < 1e-6
        assert abs(metrics["NMI"] - 1.1) < 1e-6

    def test_missing_file(self, tmp_path: Path) -> None:
        metrics = _parse_metrics_csv(tmp_path / "nonexistent.csv")
        assert metrics == {}


# ---------------------------------------------------------------------------
# Gate 4: Truth Label Coverage
# ---------------------------------------------------------------------------


class TestGate4:
    def test_pass_with_files(self, tmp_path: Path) -> None:
        truth = tmp_path / "truth_export"
        truth.mkdir()
        (truth / "s.tif").write_bytes(b"x")
        result = gate4_truth_coverage(tmp_path)
        assert result.passed

    def test_fail_missing_dir(self, tmp_path: Path) -> None:
        result = gate4_truth_coverage(tmp_path)
        assert not result.passed


# ---------------------------------------------------------------------------
# Gate 5: Cell Detection Sanity
# ---------------------------------------------------------------------------


class TestGate5:
    def test_pass_with_cells(self, tmp_path: Path) -> None:
        _write_cells_csv(tmp_path, 5)
        result = gate5_cell_detection(tmp_path)
        assert result.passed
        assert "5 mapped" in result.details

    def test_fail_missing(self, tmp_path: Path) -> None:
        result = gate5_cell_detection(tmp_path)
        assert not result.passed

    def test_fail_header_only(self, tmp_path: Path) -> None:
        _write_cells_csv(tmp_path, 0)
        result = gate5_cell_detection(tmp_path)
        assert not result.passed
        assert "No data rows" in result.details

    def test_reads_artifact_from_progress(self, tmp_path: Path) -> None:
        """Gate 5 should find cells_mapped.csv via pipeline_progress.json artifacts."""
        alt_dir = tmp_path / "alt"
        alt_dir.mkdir()
        _write_cells_csv(alt_dir, 3)
        cells_path = alt_dir / "cells_mapped.csv"

        # Write progress pointing to the alt location
        (tmp_path / "pipeline_progress.json").write_text(
            json.dumps({"percent": 100, "artifacts": {"cells_mapped_csv": str(cells_path)}}),
            encoding="utf-8",
        )
        result = gate5_cell_detection(tmp_path)
        assert result.passed
        assert "3 mapped" in result.details


# ---------------------------------------------------------------------------
# Integration: run_gates
# ---------------------------------------------------------------------------


class TestRunGates:
    def test_all_pass(self, tmp_path: Path) -> None:
        _setup_full_canary(tmp_path)
        results = run_gates(tmp_path, "A")
        assert all(r.passed for r in results), [r for r in results if not r.passed]

    def test_mixed_results(self, tmp_path: Path) -> None:
        _setup_full_canary(tmp_path)
        # Remove cells_mapped.csv to trigger Gate 5 failure
        (tmp_path / "cells_mapped.csv").unlink()
        # Remove the progress file's artifact pointer too
        (tmp_path / "pipeline_progress.json").write_text(
            json.dumps({"percent": 100}),
            encoding="utf-8",
        )
        results = run_gates(tmp_path, "A")
        gate5 = [r for r in results if r.gate == 5][0]
        assert not gate5.passed
        # Other gates should still pass
        other_gates = [r for r in results if r.gate != 5]
        assert all(r.passed for r in other_gates)
