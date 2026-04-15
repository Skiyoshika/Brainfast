# Registration Quality Gate

**Date:** 2026-04-12
**Status:** Active policy
**Purpose:** Registration quality is a HARD gate before region mapping. If registration is not acceptable, cell-to-region assignments are meaningless and must not be exported or presented as results.

---

## 1. The Operator Rule

> If the overlay does not look right, the region counts are not trustworthy.

This is the single most important principle. No metric can fully substitute for visual inspection of atlas-on-tissue overlays. Automated thresholds catch gross failures; the operator catches the rest.

---

## 2. Numeric Acceptance Thresholds

### 2.1 Cross-modality registration (fluorescence vs Nissl atlas)

This is the default mode for cleared-tissue samples (e.g., Sample 35, ChATe27).

| Metric | Minimum | Notes |
|--------|---------|-------|
| Dice (tissue mask overlap) | 0.30 | Half-hemisphere template inherently low (~0.35 typical) |
| NMI (normalized mutual information) | 1.01 | Primary cross-modality metric |
| SSIM (structural similarity) | 0.05 | Low bar because modalities differ fundamentally |

These are the thresholds used by `validate_canary.py` for Canary A/C buckets. Canary B (sparse 5-slice volume) uses a relaxed Dice threshold of 0.01 because sparse volumes have minimal tissue overlap by design.

### 2.2 Same-modality registration (Nissl vs Nissl, or autofluorescence vs template)

Not yet implemented. When added, thresholds should be significantly higher:

| Metric | Minimum (proposed) | Notes |
|--------|-------------------|-------|
| Dice | 0.65 | Same contrast = better overlap expected |
| NMI | 1.10 | Higher baseline from matched modalities |
| SSIM | 0.30 | Meaningful structural similarity expected |

These values will be calibrated when same-modality samples are available.

### 2.3 Additional hard checks (no threshold -- binary pass/fail)

- **No hemisphere inversion:** Atlas medial/lateral edges must match tissue medial/lateral edges. See taxonomy Category 2.
- **No AP mismatch:** Major anatomical landmarks (hippocampus, striatum, cortex boundaries) must correspond between atlas and tissue. See taxonomy Category 1.
- **Annotation coverage > 20%:** Fraction of tissue pixels with non-zero atlas labels must exceed 20%. Below this, interpolation or padding is wrong (taxonomy Category 4).

---

## 3. Per-Slice vs Volume-Level Gates

### 3.1 `fail_score_threshold` (per-slice, 2D)

Configured in `run_config.template.json` as `registration.fail_score_threshold` (default: 0.65). This is the edge-SSIM-based score used during the registration loop to decide whether a single slice's registration is acceptable or should trigger landmark fallback.

- Operates **during** registration, not after.
- Controls retry/fallback behavior for individual slices.
- A slice below this threshold gets re-registered with landmark guidance (`landmark_fallback: true`).

**Critical: modality-dependent calibration required.**

Empirical evidence from Sample 35 (cross-modality, 111 slices):
```
Score distribution (edge_ssim):
  min=0.1535  max=0.5749  mean=0.3138  median=0.2952

Failure rates by threshold:
  0.10 →   0/111 fail ( 0.0%)  ← Sample 35 config uses this
  0.20 →  20/111 fail (18.0%)
  0.30 →  56/111 fail (50.5%)
  0.65 → 111/111 fail (100%)   ← template default would reject ALL
```

This proves the template default (0.65) is ONLY appropriate for same-modality.
Cross-modality configs MUST override to ~0.10 to avoid false rejections.
The per-sample config pattern is correct by design.

### 3.2 `validate_canary.py` Gate 3 (volume-level, 3D)

Operates **after** the full pipeline completes. Reads `registration_metrics.csv` which contains aggregate metrics (Dice, NMI, SSIM) computed over the entire registered volume.

- This is the hard gate that blocks release progression (see `stability-exit-criteria.md` Stages A-D).
- A volume can have individual slices that passed `fail_score_threshold` but still fail the volume-level gate if aggregate quality is poor.

### 3.3 Relationship

```
Per-slice (fail_score_threshold=0.65)
  --> controls: retry/fallback during registration
  --> scope: single 2D slice NCC score

Volume-level (validate_canary.py Gate 3)
  --> controls: pipeline pass/fail after completion
  --> scope: aggregate Dice/NMI/SSIM across all registered slices
```

Both must be satisfied. A volume where every slice barely passes per-slice threshold may still fail volume-level metrics due to cross-slice inconsistency.

---

## 4. Behavior When Registration Fails the Gate

### 4.1 Automated behavior

When `validate_canary.py` Gate 3 fails:

1. **Block advancement:** The sample does not progress to the next stability stage (B, C, D). See `stability-exit-criteria.md` section 5 (Rollback Policy).
2. **No region mapping trust:** `cells_mapped.csv` and `cell_counts_hierarchy.csv` exist but are flagged as unreliable.
3. **Exit code 1:** The validation script returns non-zero, which blocks CI and any downstream automation.

### 4.2 User-facing behavior (web UI)

When registration quality is below threshold:

1. The QC tab surfaces a warning banner: registration metrics below acceptable range.
2. The Results tab displays counts with a disclaimer that region assignments may be inaccurate.
3. Export functions should include a header/metadata field indicating registration quality status.

### 4.3 Operator action required

Classify the failure using the debugging checklist in `registration-failure-taxonomy.md` (Categories 1-7, in order). Fix the root cause before re-running. Do not tune nonlinear warp parameters until AP, hemisphere, and scale issues are resolved.

---

## 5. How `validate_canary.py` Implements These Gates

The script (`project/scripts/validate_canary.py`) runs 6 sequential gates:

| Gate | What it checks | Blocks on failure |
|------|---------------|-------------------|
| 1 - No Hard Crash | `pipeline_progress.json` has `percent=100` | Yes |
| 2 - QC Outputs | `registration_metrics.csv` has >= 6 rows; `truth_export/` and `volume/` exist | Yes |
| 3 - Registration Quality | Dice >= threshold, SSIM >= 0.05, NMI >= 1.01 (thresholds vary by canary bucket) | Yes -- this is the quality gate |
| 4 - Truth Coverage | At least 1 truth label slice in `truth_export/` | Yes |
| 5 - Cell Detection | `cells_mapped.csv` has > 0 data rows | Yes |
| 6 - Orientation | Always passes -- prints a reminder to visually inspect L/R | No (informational only) |

Key implementation details:

- **Threshold lookup:** `THRESHOLDS` dict maps canary variant (A/B/C) to per-metric minimums. Canary B has relaxed Dice (0.01) for sparse volumes.
- **CSV parsing:** `_parse_metrics_csv()` handles both tall (metric, value) and wide (column-per-metric) CSV formats.
- **NaN/Inf rejection:** Any metric that is NaN or Inf is treated as a failure, not silently ignored.
- **All-or-nothing:** If any single metric fails, the entire Gate 3 fails. There is no partial pass.

---

## 6. The Orientation Gate: From Reminder to Signoff

### Current state (Gate 6)

Gate 6 always returns `passed=True` with a text reminder. This means a hemisphere-inverted registration can pass all automated gates.

### Required upgrade

Gate 6 must become a documented manual signoff step in the operator workflow:

1. **Before exporting results**, the operator must visually inspect overlay images for at least 3 slices (anterior, middle, posterior).
2. **Check specifically for:**
   - Cortex on the correct side (lateral vs medial).
   - Hippocampus orientation matches tissue.
   - Midline structures align with the tissue's medial edge.
3. **Record the signoff:** A file `orientation_signoff.json` in the output directory with:
   ```json
   {
     "operator": "name",
     "date": "2026-04-12",
     "slices_inspected": ["slice_020", "slice_055", "slice_090"],
     "orientation_correct": true,
     "notes": ""
   }
   ```
4. **Gate 6 upgrade path:** Once the signoff file format is stable, `validate_canary.py` Gate 6 should check for the existence of `orientation_signoff.json` with `orientation_correct: true`. Until then, the operator checklist (see `canary-operator-checklist.md`) must include this step.

### Why this matters

Hemisphere inversion is a Category 2 failure (see taxonomy) -- severity Fatal. Every region label maps to the wrong hemisphere. Automated metrics (Dice, NMI, SSIM) cannot reliably detect L/R inversion because the metrics are symmetric. Only visual inspection catches this.

---

## 7. Summary: What Must Be True Before Trusting Region Counts

All of the following, with no exceptions:

- [ ] `validate_canary.py` exits 0 (all 6 gates pass)
- [ ] Volume-level Dice, NMI, SSIM meet modality-appropriate thresholds
- [ ] No per-slice `fail_score_threshold` failures that were not recovered by landmark fallback
- [ ] Annotation coverage > 20% of tissue area
- [ ] Operator has visually inspected >= 3 overlay slices across the Z range
- [ ] Operator confirms correct L/R orientation (manual signoff)
- [ ] No AP mismatch visible in inspected slices

If any item fails, region counts from that run are not trustworthy and must not be used for biological conclusions.
