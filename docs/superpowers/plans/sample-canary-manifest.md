# Brainfast Canary Sample Manifest

> **Purpose:** Define three sample buckets for staged pipeline validation before
> full-batch rollout.  Each canary must pass twice in a clean shell before
> expanding sample scope (per Task 5 of the stability rollout plan).

---

## Canary Buckets

### Canary A — Known-Good Baseline (Sample 35 Demo, 111 slices)

| Field | Value |
|---|---|
| **Data path** | `project/data/35_C0_demo` |
| **Config** | `project/configs/run_config_35.json` |
| **Slice count** | 111 (z0050 → z0600, step 5) |
| **Pixel size** | 5 µm/px XY |
| **Z-step** | 25 µm |
| **Hemisphere** | `right_flipped` (lateral=LEFT, medial=RIGHT) |
| **ml_flip** | `false` |
| **Detector** | `cpsam` with LoG fallback |
| **Registration** | `miki_3d` + ANTs SyNRA + Mattes MI |
| **Why chosen** | Already runs end-to-end (v7 completed all 6 stages). Establishes reproducibility baseline. |
| **Known issues** | NCC = −0.08 (cross-modality), annotation coverage ~41% after extrapolation. These are baseline numbers, not failures. |

### Canary B — Reduced Slice Subset (Sample 35 Test, 5 slices)

| Field | Value |
|---|---|
| **Data path** | `project/data/35_C0_test` |
| **Config** | `project/configs/run_config_35_canary_b.json` (create from `run_config_35.json`) |
| **Slice count** | 5 (z0200, z0225, z0250, z0300, z0350) |
| **Pixel size** | 5 µm/px XY |
| **Z-step** | 25 µm (but sparse — irregular gaps) |
| **Hemisphere** | `right_flipped` |
| **ml_flip** | `false` |
| **Detector** | `cpsam` with LoG fallback |
| **Registration** | `miki_3d` + ANTs SyNRA + Mattes MI |
| **Why chosen** | Tests pipeline robustness with very few slices and irregular AP spacing. Exercises edge cases in volume construction, AP interpolation, and deduplication with sparse Z-stack. Fast to run (~5 min). |
| **Expected challenges** | Sparse Z-stack may produce poor 3D volume reconstruction. AP formula coverage is narrow (z200→z350 = AP 290→260). Template-to-input volume ratio is extreme. |

### Canary C — Full Z-Stack Stress Test (Sample 35 Full, 100 slices)

| Field | Value |
|---|---|
| **Data path** | `project/data/35_C0_full` |
| **Config** | `project/configs/run_config_35_canary_c.json` (create from `run_config_35.json`) |
| **Slice count** | 100 (z0050 → z0545, step 5) |
| **Pixel size** | 5 µm/px XY |
| **Z-step** | 25 µm |
| **Hemisphere** | `right_flipped` |
| **ml_flip** | `false` |
| **Detector** | `cpsam` with LoG fallback |
| **Registration** | `miki_3d` + ANTs SyNRA + Mattes MI |
| **Why chosen** | Near-identical to Canary A but with 100 vs 111 slices (z0545 cutoff vs z0600). Tests whether the pipeline is robust to slight Z-range differences and confirms that results are comparable across marginally different input sets. |
| **Expected challenges** | Slightly shorter AP range. Output counts should be proportionally similar to Canary A. If results differ dramatically, it signals Z-range sensitivity. |

---

## Pass/Fail Gates

Every canary run **must** satisfy ALL of these gates to pass:

### Gate 1: No Hard Crash
- Pipeline completes without Python exception or non-zero exit code
- `pipeline_progress.json` shows `stageIndex == stageCount` and `percent == 100`

### Gate 2: Non-Empty QC Outputs
- `ants_registration/registration_metrics.csv` exists and has 6 rows (NCC, NMI, SSIM, Dice, MSE, PSNR)
- `truth_export/` directory exists and contains at least 1 annotated truth slice
- `volume/` directory exists with the merged 3D volume

### Gate 3: Registration Quality Bounds
- **Dice ≥ 0.70** (tissue overlap with template)
- **SSIM ≥ 0.05** (cross-modality baseline; fluorescence vs Nissl will be low)
- **NMI ≥ 1.01** (above independence = some alignment signal)
- If any metric is NaN or Inf → FAIL

### Gate 4: Truth Label Coverage
- At least 1 truth export slice has non-zero annotation labels
- Annotation coverage (tissue pixels with label > 0 / total tissue pixels) ≥ 15%
- At least 3 distinct Allen CCF structure IDs present in the annotation

### Gate 5: Cell Detection Sanity
- `cells_mapped.csv` exists and has > 0 rows
- If `cpsam` was requested: verify it was used (not silently fell back to LoG) — check log for `"Using Cellpose"` or `"cpsam"` model string
- Cell count is non-zero when tissue is clearly present

### Gate 6: Orientation Spot Check
- Overlay of truth slice at ~50% AP shows:
  - Cortical areas (SSp, MOp, SSs) on the **lateral** side (LEFT in `right_flipped` orientation)
  - Subcortical structures (CP, thalamus) in the **center**
  - Medial structures (ACA, RSP) on the **medial** side (RIGHT)
- No obvious left-right or anterior-posterior inversion

### Gate 7: Reproducibility
- Running the same canary twice with the same config and `random_seed: 42` produces:
  - Identical `registration_metrics.csv` values (within floating-point tolerance ε < 1e-6)
  - Same number of detected cells (±5%)
  - Same pipeline stage completion pattern

---

## Promotion Rules

| Milestone | Requirement |
|---|---|
| Canary A passes once | Proceed to Canary B |
| Canary B passes once | Proceed to Canary C |
| All 3 canaries pass once | Run each canary a **second** time |
| All 3 canaries pass **twice** | Promote to Task 6 (20% rollout) |
| Any canary fails | Stop. Diagnose. Fix. Re-run from Canary A. |

---

## Notes

- **Cross-modality NCC will be low or negative.** This is expected when comparing
  fluorescence (input) vs Nissl-stained (Allen template). NCC assumes linear
  intensity relationship. Use NMI ≥ 1.01 as the primary alignment quality metric.
- **Annotation coverage < 50% is expected** for half-hemisphere registration with
  extrapolation. Coverage > 15% with correct laterality is the minimum bar.
- **Canary B (5 slices) will likely have poor 3D reconstruction.** The goal is to
  verify the pipeline does not crash, not to get high-quality registration from 5 slices.
  Apply relaxed quality gates: Dice ≥ 0.50, coverage ≥ 5%.
- When a new sample from a different animal/preparation becomes available, create
  a **Canary D** bucket and re-run the promotion sequence from scratch.
