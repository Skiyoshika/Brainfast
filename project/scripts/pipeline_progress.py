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
    "Volume Build": ("per_slice", 0.6),  # I/O bound, 600ms/slice
    "Template Prep": ("constant", 60.0),  # ~1 min, atlas crop
    "Intensity Adapt": ("per_slice", 0.3),  # included in template prep stage
    "Axis Alignment": ("constant", 120.0),  # vendored RegTools, ~2 min
    "ANTS Registration": ("constant", 2000.0),  # average of 27.8 + 40 min
    "Laplacian Refinement": ("constant", 30.0),  # <1 min
    "Truth Export": ("per_slice", 7.0),  # 7 sec/slice (consistent across runs)
    "Quantification": ("per_slice", 16.5),  # LoG detect dominates, 16.5s/slice
}

# Stage index → name mapping (Brainfast 6-stage pipeline)
DEFAULT_PIPELINE_STAGES: list[str] = [
    "Volume Build",  # 1
    "Template Prep",  # 2
    "ANTS Registration",  # 3
    "Laplacian Refinement",  # 4
    "Truth Export",  # 5
    "Quantification",  # 6
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

    # Carry forward + accumulate per-stage completion times. Two trigger paths:
    # (a) on stage transition: record the *previous* stage's elapsed; (b) on
    # final-stage 100%: record the current stage's elapsed before declaring
    # the run complete (so callers can append the whole record to history).
    stage_completions: dict[str, float] = dict(existing.get("stageCompletions", {}) or {})
    if prev_index is not None and prev_index != int(stage_index):
        prev_name = existing.get("stageName")
        prev_started = float(existing.get("stageStartedTs", now))
        if prev_name and prev_name not in stage_completions:
            stage_completions[prev_name] = max(0.0, now - prev_started)
    if int(stage_index) == int(stage_count) and int(percent) >= 100:
        # Final write — make sure the closing stage is captured too
        if str(stage_name) not in stage_completions:
            stage_completions[str(stage_name)] = max(0.0, now - stage_started_ts)

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
        "stageCompletions": stage_completions,
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

    current_baseline = _stage_baseline_seconds(current_stage_name, slice_count, table)
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
    baseline_through_current = sum(
        _stage_baseline_seconds(n, slice_count, table) for n in completed_stages
    ) + current_baseline * (percent / 100.0)
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
        _stage_baseline_seconds(name, slice_count, table) for name in stages[:stage_count]
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


# ---------------------------------------------------------------------------
# History persistence + per-stage baseline learning
# ---------------------------------------------------------------------------
#
# Each completed run appends one JSON line to ``eta_history.jsonl`` with its
# slice_count and the per-stage elapsed seconds it observed. Future runs can
# load this history and, for stages with ≥ ``MIN_HISTORY_SAMPLES`` recordings,
# replace the hardcoded ``DEFAULT_STAGE_BASELINES`` entries with means derived
# from real local hardware. This makes ETA self-calibrating without ever
# needing to ship baselines to the user.

MIN_HISTORY_SAMPLES: int = 2


def append_run_to_history(history_path: Path | str, run_record: dict) -> None:
    """Append a completed run's per-stage timings to the history JSONL file.

    ``run_record`` should contain at least:
    * ``run_id`` — string identifier (the job id is fine)
    * ``completed_at`` — UNIX seconds when the run finished
    * ``slice_count`` — input slice count, for per_slice baseline derivation
    * ``stage_completions`` — ``{stage_name: elapsed_seconds}`` mapping

    The file is opened in append-binary mode + line-buffered, so concurrent
    writers from sibling jobs don't trample each other (each writes one
    self-contained line).
    """
    history_path = Path(history_path)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(run_record, ensure_ascii=False) + "\n"
    # Use 'a' mode for atomic-ish line append on POSIX + Windows
    with history_path.open("a", encoding="utf-8") as fh:
        fh.write(line)


def _load_history_records(history_path: Path) -> list[dict]:
    if not history_path.exists():
        return []
    records: list[dict] = []
    with history_path.open("r", encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                # Skip a corrupt line rather than nuking the entire learner
                continue
    return records


def compute_baselines_from_history(
    history_path: Path | str,
    *,
    defaults: dict[str, tuple[str, float]] | None = None,
    min_samples: int = MIN_HISTORY_SAMPLES,
) -> dict[str, tuple[str, float]]:
    """Read the history file and return a baselines dict keyed by stage name.

    For each stage we know the *kind* (``constant`` or ``per_slice``) from the
    defaults table. Aggregation strategy depends on kind:

    * ``constant`` — average ``elapsed_seconds`` directly across runs.
    * ``per_slice`` — average ``elapsed_seconds / slice_count`` across runs.

    A stage with fewer than ``min_samples`` observations keeps its hardcoded
    default; this prevents a single-run outlier from polluting the model.
    """
    history_path = Path(history_path)
    table = dict(defaults or DEFAULT_STAGE_BASELINES)

    records = _load_history_records(history_path)
    if not records:
        return table

    # Collect per-stage observations
    observations: dict[str, list[tuple[float, int]]] = {}
    for rec in records:
        slice_count = int(rec.get("slice_count") or 0)
        completions = rec.get("stage_completions") or {}
        if not isinstance(completions, dict):
            continue
        for stage_name, elapsed in completions.items():
            try:
                elapsed_f = float(elapsed)
            except (TypeError, ValueError):
                continue
            if elapsed_f <= 0:
                continue
            observations.setdefault(str(stage_name), []).append((elapsed_f, slice_count))

    for stage_name, obs in observations.items():
        if len(obs) < int(min_samples):
            continue  # not enough signal yet
        kind, _factor = table.get(stage_name, ("constant", 60.0))
        if kind == "per_slice":
            # Convert each observation to s/slice, then mean
            per_slice_rates = [elapsed / max(1, slice_count) for elapsed, slice_count in obs]
            mean_rate = sum(per_slice_rates) / len(per_slice_rates)
            table[stage_name] = ("per_slice", mean_rate)
        else:  # constant
            mean_seconds = sum(elapsed for elapsed, _ in obs) / len(obs)
            table[stage_name] = ("constant", mean_seconds)

    return table


def maybe_record_run_completion(
    outputs_dir: Path | str,
    *,
    history_path: Path | str,
    run_id: str,
    slice_count: int,
) -> bool:
    """If the latest progress in ``outputs_dir`` represents a completed run
    (final stage at 100%) and we have stageCompletions captured, append a
    history record. Returns ``True`` when a record was written.

    Idempotent: if the same run_id is already at the end of the history file,
    we don't duplicate the record. (The simplest invariant the runner can
    rely on without extra plumbing.)
    """
    progress = read_stage_progress(outputs_dir)
    if not progress:
        return False
    stage_index = int(progress.get("stageIndex", 0) or 0)
    stage_count = int(progress.get("stageCount", 0) or 0)
    percent = int(progress.get("percent", 0) or 0)
    if stage_index < stage_count or percent < 100 or stage_count == 0:
        return False
    completions = progress.get("stageCompletions") or {}
    if not completions:
        return False

    history_path = Path(history_path)
    # Idempotency check: peek at last line
    if history_path.exists():
        with history_path.open("rb") as fh:
            try:
                fh.seek(-2048, os.SEEK_END)
            except OSError:
                fh.seek(0)
            tail = fh.read().decode("utf-8", errors="replace")
        if tail:
            last_line = tail.strip().splitlines()[-1] if tail.strip() else ""
            try:
                last_rec = json.loads(last_line)
                if last_rec.get("run_id") == run_id:
                    return False
            except (json.JSONDecodeError, AttributeError):
                pass

    append_run_to_history(
        history_path,
        run_record={
            "run_id": str(run_id),
            "completed_at": time.time(),
            "slice_count": int(slice_count),
            "stage_completions": dict(completions),
        },
    )
    return True
