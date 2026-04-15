"""Automated pass/fail gate validation for canary pipeline runs.

Usage:
    python scripts/validate_canary.py --output-dir outputs/ChATe27_35_v7
    python scripts/validate_canary.py --output-dir outputs/canary_b --canary B
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import NamedTuple


class GateResult(NamedTuple):
    """Result of a single gate check."""

    gate: int
    name: str
    passed: bool
    details: str


# ---------------------------------------------------------------------------
# Threshold configuration per canary variant
# ---------------------------------------------------------------------------

# Dice thresholds are set for cross-modality half-brain registration
# (fluorescence input vs Nissl-stained Allen template).  Tissue mask
# overlap (Dice) is inherently low (~0.35) because the half-hemisphere
# template covers only a fraction of the tissue section.  NMI and SSIM
# are the primary quality metrics for cross-modality alignment.
THRESHOLDS: dict[str, dict[str, float]] = {
    "A": {"dice_min": 0.30, "ssim_min": 0.05, "nmi_min": 1.01},
    "B": {"dice_min": 0.01, "ssim_min": 0.05, "nmi_min": 1.01},  # sparse 5-slice vol
    "C": {"dice_min": 0.15, "ssim_min": 0.05, "nmi_min": 1.01},  # shorter AP range
}


# ---------------------------------------------------------------------------
# Gate implementations
# ---------------------------------------------------------------------------


def gate1_no_crash(output_dir: Path) -> GateResult:
    """Gate 1: Pipeline completed without hard crash (percent == 100)."""
    progress_file = output_dir / "pipeline_progress.json"
    if not progress_file.exists():
        return GateResult(1, "No Hard Crash", False, f"Missing {progress_file}")

    try:
        data = json.loads(progress_file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return GateResult(1, "No Hard Crash", False, f"Cannot parse progress file: {exc}")

    percent = data.get("percent")
    if percent != 100:
        return GateResult(1, "No Hard Crash", False, f"percent={percent}, expected 100")

    return GateResult(1, "No Hard Crash", True, "pipeline_progress.json percent=100")


def gate2_qc_outputs(output_dir: Path) -> GateResult:
    """Gate 2: Non-empty QC outputs exist."""
    issues: list[str] = []

    # registration_metrics.csv with >= 6 data rows
    metrics_csv = output_dir / "ants_registration" / "registration_metrics.csv"
    if not metrics_csv.exists():
        issues.append(f"Missing {metrics_csv.relative_to(output_dir)}")
    else:
        try:
            with metrics_csv.open(encoding="utf-8") as fh:
                reader = csv.reader(fh)
                rows = list(reader)
            # rows[0] is header, rest are data
            data_rows = len(rows) - 1 if len(rows) > 0 else 0
            if data_rows < 6:
                issues.append(f"registration_metrics.csv has {data_rows} data rows, need >= 6")
        except OSError as exc:
            issues.append(f"Cannot read registration_metrics.csv: {exc}")

    # truth_export/ has at least 1 file
    truth_dir = output_dir / "truth_export"
    if not truth_dir.exists():
        issues.append("Missing truth_export/ directory")
    else:
        truth_files = [f for f in truth_dir.iterdir() if f.is_file()]
        if len(truth_files) == 0:
            issues.append("truth_export/ is empty")

    # volume/ exists
    volume_dir = output_dir / "volume"
    if not volume_dir.exists():
        issues.append("Missing volume/ directory")

    if issues:
        return GateResult(2, "Non-Empty QC Outputs", False, "; ".join(issues))
    return GateResult(2, "Non-Empty QC Outputs", True, "All QC artifacts present")


def _parse_metrics_csv(metrics_csv: Path) -> dict[str, float]:
    """Parse registration_metrics.csv in either tall (metric,value) or wide format.

    Returns a dict mapping metric names (e.g. "Dice", "NMI") to float values.
    """
    metrics: dict[str, float] = {}
    try:
        with metrics_csv.open(encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
    except (OSError, csv.Error):
        return metrics

    if not rows:
        return metrics

    # Detect format: tall (columns: metric, value) vs wide (columns: NCC, NMI, ...)
    headers = set(rows[0].keys())
    if {"metric", "value"} <= headers:
        # Tall format: each row is one metric
        for row in rows:
            name = row.get("metric", "").strip()
            raw = row.get("value", "").strip()
            if name and raw:
                try:
                    metrics[name] = float(raw)
                except ValueError:
                    pass
    else:
        # Wide format: single row with metric columns
        for row in rows:
            for key, raw in row.items():
                if raw and key.strip():
                    try:
                        metrics[key.strip()] = float(raw)
                    except ValueError:
                        pass
    return metrics


def gate3_registration_quality(output_dir: Path, canary: str) -> GateResult:
    """Gate 3: Registration metrics meet quality thresholds."""
    metrics_csv = output_dir / "ants_registration" / "registration_metrics.csv"
    if not metrics_csv.exists():
        return GateResult(3, "Registration Quality", False, "registration_metrics.csv missing")

    thresh = THRESHOLDS[canary]
    metrics = _parse_metrics_csv(metrics_csv)

    if not metrics:
        return GateResult(3, "Registration Quality", False, "No metrics parsed from CSV")

    issues: list[str] = []
    details_parts: list[str] = []

    checks = [
        ("Dice", thresh["dice_min"]),
        ("SSIM", thresh["ssim_min"]),
        ("NMI", thresh["nmi_min"]),
    ]

    for metric_name, threshold in checks:
        val = metrics.get(metric_name)
        if val is None:
            issues.append(f"{metric_name} missing")
            continue

        if math.isnan(val) or math.isinf(val):
            issues.append(f"{metric_name}={val} (NaN/Inf)")
            continue

        if val < threshold:
            issues.append(f"{metric_name}={val:.4f} < {threshold}")
        else:
            details_parts.append(f"{metric_name}={val:.4f}")

    if issues:
        return GateResult(3, "Registration Quality", False, "; ".join(issues))
    return GateResult(3, "Registration Quality", True, f"All pass: {', '.join(details_parts)}")


def gate4_truth_coverage(output_dir: Path) -> GateResult:
    """Gate 4: At least 1 truth label slice exists."""
    truth_dir = output_dir / "truth_export"
    if not truth_dir.exists():
        return GateResult(4, "Truth Label Coverage", False, "truth_export/ missing")

    truth_files = [f for f in truth_dir.iterdir() if f.is_file()]
    count = len(truth_files)
    if count == 0:
        return GateResult(4, "Truth Label Coverage", False, "No truth slices found")

    return GateResult(4, "Truth Label Coverage", True, f"{count} truth slice(s) found")


def gate5_cell_detection(output_dir: Path) -> GateResult:
    """Gate 5: cells_mapped.csv exists and has > 0 data rows."""
    cells_csv = output_dir / "cells_mapped.csv"

    # Also check pipeline_progress.json for the artifact path
    if not cells_csv.exists():
        progress_file = output_dir / "pipeline_progress.json"
        if progress_file.exists():
            try:
                data = json.loads(progress_file.read_text(encoding="utf-8"))
                artifact_path = data.get("artifacts", {}).get("cells_mapped_csv", "")
                if artifact_path and Path(artifact_path).exists():
                    cells_csv = Path(artifact_path)
            except (json.JSONDecodeError, OSError):
                pass

    if not cells_csv.exists():
        return GateResult(5, "Cell Detection Sanity", False, "cells_mapped.csv missing")

    try:
        with cells_csv.open(encoding="utf-8") as fh:
            reader = csv.reader(fh)
            rows = list(reader)
    except OSError as exc:
        return GateResult(5, "Cell Detection Sanity", False, f"Cannot read file: {exc}")

    data_rows = len(rows) - 1 if len(rows) > 0 else 0
    if data_rows <= 0:
        return GateResult(5, "Cell Detection Sanity", False, "No data rows (header only or empty)")

    return GateResult(5, "Cell Detection Sanity", True, f"{data_rows} mapped cell(s)")


def gate6_orientation(output_dir: Path) -> GateResult:
    """Gate 6: Orientation check (informational only -- always passes)."""
    return GateResult(
        6,
        "Orientation (visual)",
        True,
        "REMINDER: Visually inspect overlay images to confirm correct L/R orientation",
    )


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_gates(output_dir: Path, canary: str) -> list[GateResult]:
    """Execute all automated gates and return results."""
    return [
        gate1_no_crash(output_dir),
        gate2_qc_outputs(output_dir),
        gate3_registration_quality(output_dir, canary),
        gate4_truth_coverage(output_dir),
        gate5_cell_detection(output_dir),
        gate6_orientation(output_dir),
    ]


def print_report(results: list[GateResult], canary: str, output_dir: Path) -> bool:
    """Print a human-readable summary. Returns True if all gates passed."""
    width = 60
    print("=" * width)
    print(f"  Canary {canary} Validation  --  {output_dir}")
    print("=" * width)

    all_passed = True
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        marker = "+" if r.passed else "X"
        print(f"  [{marker}] Gate {r.gate}: {r.name} ... {status}")
        print(f"      {r.details}")
        if not r.passed:
            all_passed = False

    print("-" * width)
    overall = "PASS" if all_passed else "FAIL"
    print(f"  Overall: {overall}")
    print("=" * width)
    return all_passed


def main(argv: list[str] | None = None) -> int:
    """Entry point. Returns 0 on success, 1 on any gate failure."""
    parser = argparse.ArgumentParser(
        description="Validate canary pipeline outputs against pass/fail gates.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Path to the pipeline output directory",
    )
    parser.add_argument(
        "--canary",
        choices=["A", "B", "C"],
        default="A",
        help="Canary variant (default: A). B uses relaxed Dice threshold.",
    )
    args = parser.parse_args(argv)

    output_dir: Path = args.output_dir.resolve()
    if not output_dir.is_dir():
        print(f"ERROR: output directory does not exist: {output_dir}", file=sys.stderr)
        return 1

    results = run_gates(output_dir, args.canary)
    all_passed = print_report(results, args.canary, output_dir)
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
