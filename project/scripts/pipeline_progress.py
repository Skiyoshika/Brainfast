"""Per-stage pipeline progress + ETA estimation.

The progress JSON is written atomically to the job's output dir after every
stage transition or sub-step update. We piggyback on it to capture stage start
times so a downstream ETA estimator can extrapolate within the current stage
*and* sum baselines for unstarted stages — without each pipeline call site
needing to know about ETA at all.

ETA design (see ``compute_eta``):

* Hardcoded per-stage cost model in ``DEFAULT_STAGE_BASELINES`` — derived from
  observed runs at 111 and 646 slices. Each entry is ``(kind, factor)``:
  ``"constant"`` → seconds, ``"per_slice"`` → seconds × slice_count.
* Within-stage ETR uses linear extrapolation from ``stageStartedTs`` once
  ≥5% progress is reported; falls back to baseline for the first sliver.
* Across remaining stages: sum of baselines for stages that haven't run yet.
* Self-correction: if the actual elapsed-so-far diverges from baseline-so-far,
  remaining baselines are scaled by ``actual / baseline`` (clamped 0.5..2.0×).
  This makes the estimate converge as the run progresses without ever needing
  per-job historical samples.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Per-stage cost baselines (seconds per stage)
# ---------------------------------------------------------------------------
# Calibrated against two observed runs:
#   - wiz-e2e-v2 (111 slices, 1636×1359 px): total 78 min
#   - real-35-full (646 slices, 1636×1359 px): total 280 min
# The constants here are the average between the two runs where the stage is
# slice-count-independent (ANTs, Laplacian) and per-slice rates derived from
# the larger run where the per-slice signal is cleaner.
DEFAULT_STAGE_BASELINES: dict[str, tuple[str, float]] = {
    # stage_name        kind          factor (seconds)
    "Volume Build":     ("per_slice", 0.6),     # I/O bound, 600ms/slice
    "Template Prep":    ("constant",  60.0),    # ~1 min, atlas crop
    "Intensity Adapt":  ("per_slice", 0.3),     # included in template prep stage
    "Axis Alignment":   ("constant",  120.0),   # vendored RegTools, ~2 min
    "ANTS Registration": ("constant", 2000.0),  # average of 27.8 + 40 min
    "Laplacian Refinement": ("constant", 30.0), # <1 min
    "Truth Export":     ("per_slice", 7.0),     # 7 sec/slice (consistent across runs)
    "Quantification":   ("per_slice", 16.5),    # LoG detect dominates, 16.5s/slice
}

# Stage index → name mapping (Brainfast 6-stage pipeline)
DEFAULT_PIPELINE_STAGES: list[str] = [
    "Volume Build",         # 1
    "Template Prep",        # 2
    "ANTS Registration",    # 3
    "Laplacian Refinement", # 4
    "Truth Export",         # 5
    "Quantification",       # 6
]


def _progress_path(outputs_dir: Path) -> Path:
    return Path(outputs_dir) / "pipeline_progress.json"


def write_stage_progress(
    outputs_dir: Path,
    stage_name: str,
    stage_index: int,
    stage_count: int,
    percent: int,
    message: str,
    artifacts: dict | None = None,
) -> Path:
    """Write a stage-progress snapshot, preserving ``stageStartedTs`` across
    writes within the same stage so ETA can extrapolate from the original
    stage entry.
    """
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)

    now = time.time()

    # If this write transitions to a new stage (different stageIndex from the
    # last write), reset stage_started_ts. Otherwise carry it forward so
    # within-stage ETR is anchored to the actual stage start.
    existing = read_stage_progress(outputs_dir)
    prev_index = existing.get("stageIndex")
    if prev_index == int(stage_index):
        stage_started_ts = float(existing.get("stageStartedTs", now))
    else:
        stage_started_ts = now
    # Preserve runStartedTs across all stages so total elapsed is computable
    # without re-walking artifact mtimes.
    run_started_ts = float(existing.get("runStartedTs", now))

    payload = {
        "stageName": str(stage_name),
        "stageIndex": int(stage_index),
        "stageCount": int(stage_count),
        "percent": int(percent),
        "message": str(message),
        "artifacts": dict(artifacts or {}),
        "ts": now,
        "stageStartedTs": stage_started_ts,
        "runStartedTs": run_started_ts,
    }
    progress_path = _progress_path(outputs_dir)
    fd, tmp_name = tempfile.mkstemp(
        prefix="pipeline_progress.",
        suffix=".tmp",
        dir=str(outputs_dir),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2))
        last_error: OSError | None = None
        for attempt in range(8):
            try:
                tmp_path.replace(progress_path)
                last_error = None
                break
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.01 * (attempt + 1))
        if last_error is not None:
            raise last_error
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return progress_path


def read_stage_progress(outputs_dir: Path) -> dict:
    progress_path = _progress_path(Path(outputs_dir))
    if not progress_path.exists():
        return {}
    return json.loads(progress_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# ETA estimation
# ---------------------------------------------------------------------------


def _stage_baseline_seconds(
    stage_name: str,
    slice_count: int,
    baselines: dict[str, tuple[str, float]] | None = None,
) -> float:
    """Return the baseline seconds for a stage given slice_count input size."""
    table = baselines or DEFAULT_STAGE_BASELINES
    entry = table.get(stage_name)
    if entry is None:
        # Unknown stage — fall back to a small constant so it doesn't dominate
        return 60.0
    kind, factor = entry
    if kind == "constant":
        return float(factor)
    if kind == "per_slice":
        return float(factor) * max(1, int(slice_count))
    return float(factor)  # safe default


def compute_eta(
    progress: dict,
    *,
    slice_count: int,
    pipeline_stages: list[str] | None = None,
    baselines: dict[str, tuple[str, float]] | None = None,
    now: float | None = None,
) -> dict:
    """Estimate remaining seconds + total ETA from current progress snapshot.

    Returns a dict with keys:

    * ``etr_in_stage_s`` — seconds remaining in the *current* stage (linear
      extrapolation from stageStartedTs once percent ≥ 5).
    * ``etr_remaining_stages_s`` — sum of baselines for stages after current.
    * ``etr_total_s`` — sum of the two; this is what the UI should display.
    * ``baseline_total_s`` — pristine baseline total assuming nominal speed.
    * ``correction_factor`` — how much we scaled remaining baselines based on
      observed-vs-expected so far. 1.0 means on baseline, 1.5 means 50% slower.
    * ``method`` — either ``"baseline_only"`` (very early), ``"linear_in_stage"``
      (mid-stage linear), or ``"linear_corrected"`` (mid-stage with correction
      applied to remaining baselines).
    """
    if not progress:
        return {
            "etr_in_stage_s": None,
            "etr_remaining_stages_s": None,
            "etr_total_s": None,
            "baseline_total_s": None,
            "correction_factor": 1.0,
            "method": "no_progress",
        }

    stages = pipeline_stages or DEFAULT_PIPELINE_STAGES
    table = baselines or DEFAULT_STAGE_BASELINES
    now_ts = float(now if now is not None else time.time())

    stage_index = int(progress.get("stageIndex", 1))
    stage_count = int(progress.get("stageCount", len(stages)))
    percent = float(progress.get("percent", 0.0))
    stage_started_ts = float(progress.get("stageStartedTs", now_ts))
    run_started_ts = float(progress.get("runStartedTs", stage_started_ts))

    # Current stage name from progress snapshot, falling back to the index map
    current_stage_name = str(progress.get("stageName", "")) or (
        stages[stage_index - 1] if 0 < stage_index <= len(stages) else "Unknown"
    )

    current_baseline = _stage_baseline_seconds(
        current_stage_name, slice_count, table
    )
    elapsed_in_stage = max(0.0, now_ts - stage_started_ts)

    # Within-stage ETR
    if percent >= 5.0:
        etr_in_stage = max(0.0, elapsed_in_stage * (100.0 - percent) / max(percent, 1.0))
        in_stage_method = "linear"
    else:
        # Too early — use baseline minus elapsed
        etr_in_stage = max(0.0, current_baseline - elapsed_in_stage)
        in_stage_method = "baseline"

    # Remaining stages baseline (everything strictly after stage_index)
    remaining_stages = stages[stage_index:stage_count]
    etr_remaining = sum(
        _stage_baseline_seconds(name, slice_count, table) for name in remaining_stages
    )

    # Self-correction: compare actual elapsed across completed stages + current
    # stage so far against baseline-so-far. Scale remaining baselines by ratio.
    completed_stages = stages[: stage_index - 1] if stage_index > 0 else []
    baseline_through_current = (
        sum(_stage_baseline_seconds(n, slice_count, table) for n in completed_stages)
        + current_baseline * (percent / 100.0)
    )
    elapsed_total = max(0.0, now_ts - run_started_ts)

    correction = 1.0
    if baseline_through_current > 60.0 and percent >= 5.0:
        # Only correct if we have enough signal (>= 1min of baseline elapsed)
        correction = elapsed_total / baseline_through_current
        correction = max(0.5, min(2.0, correction))
        etr_remaining *= correction
        method = "linear_corrected"
    elif percent >= 5.0:
        method = "linear_in_stage"
    else:
        method = "baseline_only"

    etr_total = etr_in_stage + etr_remaining
    baseline_total = sum(
        _stage_baseline_seconds(name, slice_count, table)
        for name in stages[:stage_count]
    )

    return {
        "etr_in_stage_s": int(etr_in_stage),
        "etr_remaining_stages_s": int(etr_remaining),
        "etr_total_s": int(etr_total),
        "baseline_total_s": int(baseline_total),
        "correction_factor": round(correction, 2),
        "method": method,
        "in_stage_method": in_stage_method,
        "elapsed_total_s": int(elapsed_total),
        "elapsed_in_stage_s": int(elapsed_in_stage),
    }
