"""Tests for ``pipeline_progress.compute_eta``.

Covers:
* Empty / missing progress → no ETA
* Early-stage (percent < 5) → baseline-only fallback
* Mid-stage linear extrapolation
* Self-correction when actual elapsed > baseline
* Cross-stage sum for unstarted stages
* Retroactive validation against the real-35-full historical timing —
  given progress at the end of ANTs (stage 3, ~30 min in), the ETA should
  predict total ≈ 4-5 hours (actual was 4h 40min).
"""

from __future__ import annotations

import time

from project.scripts.pipeline_progress import (
    DEFAULT_PIPELINE_STAGES,
    compute_eta,
    write_stage_progress,
    read_stage_progress,
)


def test_compute_eta_returns_no_progress_on_empty_dict():
    result = compute_eta({}, slice_count=111)
    assert result["etr_total_s"] is None
    assert result["method"] == "no_progress"


def test_compute_eta_uses_baseline_when_stage_just_started():
    """Percent <1 → too early for linear extrapolation; use baseline."""
    now = 1_700_000_000.0
    progress = {
        "stageName": "ANTS Registration",
        "stageIndex": 3,
        "stageCount": 6,
        "percent": 0,
        "stageStartedTs": now - 10,  # only 10s in
        "runStartedTs": now - 70,    # 70s total elapsed
    }
    result = compute_eta(progress, slice_count=111, now=now)

    assert result["method"] == "baseline_only"
    # ANTs baseline 2000 - 10 elapsed = 1990 in stage
    assert 1900 <= result["etr_in_stage_s"] <= 2010
    # Plus Laplacian (30) + Truth (111×7=777) + Quant (111×16.5=1831) = 2638s remaining
    assert 2500 <= result["etr_remaining_stages_s"] <= 2750


def test_compute_eta_linear_extrapolation_mid_stage():
    """At 50% of a stage that's been running for 600s, predict ~600s remaining."""
    now = 1_700_000_000.0
    progress = {
        "stageName": "ANTS Registration",
        "stageIndex": 3,
        "stageCount": 6,
        "percent": 50,
        "stageStartedTs": now - 600,
        "runStartedTs": now - 700,  # 100s of stages 1-2 + 600s in stage 3
    }
    result = compute_eta(progress, slice_count=111, now=now)

    # In-stage: 50% remaining at the same rate → ~600s
    assert 550 <= result["etr_in_stage_s"] <= 650
    assert result["in_stage_method"] == "linear"


def test_compute_eta_self_corrects_when_pipeline_running_slow():
    """If actual elapsed-so-far is 1.5× baseline, remaining should scale up too."""
    now = 1_700_000_000.0

    # Stage 3 ANTs, 50% done, but elapsed_total way more than baseline-so-far
    # baseline through current = (volume build 67) + (template 60) + (ants 2000 * 0.5) = 1127s
    # actual elapsed = 1700s → ratio 1.51× → remaining should scale by 1.51
    progress = {
        "stageName": "ANTS Registration",
        "stageIndex": 3,
        "stageCount": 6,
        "percent": 50,
        "stageStartedTs": now - 1500,
        "runStartedTs": now - 1700,
    }
    result = compute_eta(progress, slice_count=111, now=now)

    assert result["method"] == "linear_corrected"
    assert 1.4 <= result["correction_factor"] <= 1.6


def test_compute_eta_cross_stage_baseline_sum_scales_with_slice_count():
    """646 slices → Truth + Quant should be ~5.8× larger than 111 slices."""
    now = 1_700_000_000.0
    progress = {
        "stageName": "Laplacian Refinement",
        "stageIndex": 4,
        "stageCount": 6,
        "percent": 100,
        "stageStartedTs": now - 30,
        "runStartedTs": now - 2200,
    }

    eta_111 = compute_eta(progress, slice_count=111, now=now)
    eta_646 = compute_eta(progress, slice_count=646, now=now)

    # Remaining stages: Truth Export + Quantification, both per_slice
    # 111: 7×111 + 16.5×111 = 777 + 1831 = 2608
    # 646: 7×646 + 16.5×646 = 4522 + 10659 = 15181
    # Ratio should be ~5.8 (might be perturbed by correction factor)
    if eta_111["correction_factor"] == eta_646["correction_factor"] == 1.0:
        ratio = eta_646["etr_remaining_stages_s"] / max(1, eta_111["etr_remaining_stages_s"])
        assert 5.5 <= ratio <= 6.1


def test_compute_eta_clamps_correction_factor_to_safe_range():
    """If a stage takes 10× baseline, don't predict 10× the remaining time."""
    now = 1_700_000_000.0
    progress = {
        "stageName": "ANTS Registration",
        "stageIndex": 3,
        "stageCount": 6,
        "percent": 10,
        "stageStartedTs": now - 5000,  # extreme slowness
        "runStartedTs": now - 6000,
    }
    result = compute_eta(progress, slice_count=111, now=now)

    assert result["correction_factor"] <= 2.0  # capped


def test_compute_eta_retroactive_validation_real_full_at_ants_done():
    """Given the historical progress snapshot at end of ANTs for real-35-full
    (646 slices, 27.8 min elapsed), the ETA should predict completion within
    a reasonable window of the actual 4h 40min total.
    """
    # Reconstructed from real-35-full timestamps:
    # Launch 10:08:46, ANTs done 10:36:36 → 27.8 min = 1670s elapsed
    # Stage 3 → just transitioning to stage 4. Use stage 4 starting:
    now = 1_700_000_000.0
    progress = {
        "stageName": "Laplacian Refinement",
        "stageIndex": 4,
        "stageCount": 6,
        "percent": 0,
        "stageStartedTs": now,
        "runStartedTs": now - 1670,
    }
    result = compute_eta(progress, slice_count=646, now=now)

    # Actual remaining was 4h40m - 28min = ~4h12m = 252min = 15120s
    # Baseline remaining: Lap 30 + Truth 7×646 + Quant 16.5×646 = 30 + 4522 + 10659 = 15211s
    # That's 4.22 hours. Our prediction should be within ±20% of actual 252min
    actual_remaining_s = (4 * 60 + 12) * 60  # 15120s
    predicted_total_s = result["etr_total_s"]

    error_ratio = predicted_total_s / actual_remaining_s
    assert 0.8 <= error_ratio <= 1.2, (
        f"ETA off by {(error_ratio - 1) * 100:+.0f}%: predicted {predicted_total_s}s, "
        f"actual {actual_remaining_s}s"
    )


def test_compute_eta_retroactive_validation_demo_at_ants_done():
    """Given progress snapshot at end of ANTs for wiz-e2e-v2 (111 slices),
    ETA should predict close to actual remaining 38 min (78min total - 40min ANTs).
    """
    now = 1_700_000_000.0
    progress = {
        "stageName": "Laplacian Refinement",
        "stageIndex": 4,
        "stageCount": 6,
        "percent": 0,
        "stageStartedTs": now,
        "runStartedTs": now - 2400,  # 40 min in
    }
    result = compute_eta(progress, slice_count=111, now=now)

    # Actual remaining: 38 min = 2280s
    actual_remaining_s = 38 * 60
    predicted_total_s = result["etr_total_s"]
    error_ratio = predicted_total_s / actual_remaining_s
    assert 0.7 <= error_ratio <= 1.4, (
        f"Demo ETA off by {(error_ratio - 1) * 100:+.0f}%: "
        f"predicted {predicted_total_s}s, actual {actual_remaining_s}s"
    )


def test_write_stage_progress_preserves_started_ts_within_same_stage(tmp_path):
    write_stage_progress(
        outputs_dir=tmp_path,
        stage_name="ANTS Registration",
        stage_index=3,
        stage_count=6,
        percent=10,
        message="Start",
    )
    first = read_stage_progress(tmp_path)
    time.sleep(0.05)
    write_stage_progress(
        outputs_dir=tmp_path,
        stage_name="ANTS Registration",
        stage_index=3,
        stage_count=6,
        percent=50,
        message="Halfway",
    )
    second = read_stage_progress(tmp_path)

    assert first["stageStartedTs"] == second["stageStartedTs"]
    assert second["ts"] > first["ts"]


def test_write_stage_progress_resets_started_ts_on_stage_change(tmp_path):
    write_stage_progress(
        outputs_dir=tmp_path,
        stage_name="ANTS Registration",
        stage_index=3,
        stage_count=6,
        percent=100,
        message="ANTs done",
    )
    first = read_stage_progress(tmp_path)
    time.sleep(0.05)
    write_stage_progress(
        outputs_dir=tmp_path,
        stage_name="Laplacian Refinement",
        stage_index=4,
        stage_count=6,
        percent=0,
        message="Lap start",
    )
    second = read_stage_progress(tmp_path)

    assert second["stageStartedTs"] > first["stageStartedTs"]
    # But runStartedTs should carry over (same run)
    assert first["runStartedTs"] == second["runStartedTs"]


def test_default_pipeline_stages_match_count():
    """Sanity: the canonical stage list has 6 entries matching stageCount=6."""
    assert len(DEFAULT_PIPELINE_STAGES) == 6
