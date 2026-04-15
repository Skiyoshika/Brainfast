# Review-Driven Rollout Handoff

**Date:** 2026-04-12
**Purpose:** Rebase the sample rollout plan on the stabilization evidence gathered
in Tasks 1-6, and define the orientation-aware canary set.

---

## 1. Evidence Summary from Tasks 1-6

### Task 1 (Environment Validation): COMPLETE
- `check_env.py` performs real import smoke tests (not just `find_spec`)
- Version bounds enforced: numpy>=1.26,<3; scipy>=1.12,<2; skimage>=0.22,<1
- Config-aware: ants/cellpose promoted from WARN to FAIL when required by config

### Task 2 (CI Default Path): COMPLETE
- `smoke-default-path` lane installs `.[full,dev]` and runs:
  - `check_env.py` against template config
  - Import smoke tests for numerical stack + ANTs + Cellpose
  - Unit tests with full deps
  - Default-path integration smoke test (`test_default_path_smoke.py`)
- Integration test exercises: volume build, template prep, ANTs registration (mocked), truth export, quantification

### Task 3 (Cellpose v4 Cleanup): COMPLETE
- `channels` kwarg isolated to legacy `models.Cellpose` only; never passed to `CellposeModel` (v4+)
- 6 dedicated tests: v4 no-channels, v3 with-channels, v4 3-value eval, v3 4-value eval, model resolution, load branching

### Task 4 (ml_flip Evidence): COMPLETE
- 3-sample matrix across Samples 35, 41, 44 (all hemisected left-hemisphere, `right_flipped`)
- NMI consistently favors `ml_flip=false`; Dice/NCC consistently favor `ml_flip=true`
- Differences are small and within noise for all samples
- **Decision: `ml_flip=false` remains template default**
- See `ml-flip-matrix-summary.md` for full evidence

### Task 5 (Registration Quality Gate): COMPLETE
- Per-slice gate in `main.py`: slices below `fail_score_threshold` are skipped, pipeline raises `RuntimeError` if any fail
- Volume-level gate via `validate_canary.py` (6 sequential gates including Dice/NMI/SSIM bounds)
- Cross-modality threshold calibrated: 0.65 rejects 100% of fluorescence-vs-Nissl slices; 0.10 is correct for cross-modality
- Operator rule documented in `registration-quality-gate.md`

### Task 6 (User Workflow): COMPLETE
- Detect preview: POST `/api/detect/preview` with overlay + CSV output
- UI: detect button, result banner, overlay toggle, CSV download
- 7 unit tests for the detect preview blueprint
- Full happy path: load sample -> preview atlas -> register -> correct -> detect -> export

---

## 2. Revised Canary Set (Orientation-Aware)

The original canary manifest used only Sample 35 in three size variants. The
revised set uses samples from different animals and preparations.

### Canary A: Easy Baseline (Sample 35, ChATe27)
- **Config:** `configs/run_config_35.json`
- **Data:** `data/35_C0_demo` (111 slices) or `data/35_C0_test` (5 slices for quick runs)
- **Hemisphere:** `right_flipped`, `ml_flip=false`
- **Marker:** ChAT (fluorescence)
- **Why:** Known-good. Completed all pipeline stages. Establishes reproducibility baseline.

### Canary B: Different Animal, Same Marker (Sample 41, ChATe27)
- **Config:** `configs/run_config_41.json`
- **Data:** `data/41_C0_test` (5 slices)
- **Hemisphere:** `right_flipped`, `ml_flip=false`
- **Marker:** ChAT (dual-laser 560nm+640nm)
- **Why:** Tests pipeline robustness across animals. Dual-laser acquisition may
  produce different intensity distributions. "both" in filename = both lasers,
  NOT both hemispheres.
- **Known issues:** Cellpose cpsam falls back to LoG on this sample.

### Canary C: Different Marker (Sample 44, PVe3)
- **Config:** `configs/run_config_44.json`
- **Data:** `data/44_C0_test` (5 slices)
- **Hemisphere:** `right_flipped`, `ml_flip=false`
- **Marker:** PV (different protein target)
- **Why:** Tests marker-agnosticism. PV-expressing cells have different
  morphology and density compared to ChAT. Different tissue preparation (PVe3
  vs ChATe27).
- **Known issues:** Cellpose cpsam falls back to LoG. Registration metrics
  comparable to Sample 41 but from different tissue prep.

### Why This Set
1. **Cross-animal:** Samples 35 vs 41 are different animals of the same line (ChATe27).
2. **Cross-marker:** Sample 44 uses a different protein marker (PV vs ChAT).
3. **Orientation-consistent:** All use `right_flipped` (all tissue is left hemisphere).
4. **All have A/B test data:** ml_flip evidence exists for all three samples.

---

## 3. Promotion Rules (Unchanged)

| Milestone | Requirement |
|---|---|
| Canary A passes once | Proceed to Canary B |
| Canary B passes once | Proceed to Canary C |
| All 3 canaries pass once | Run each canary a second time |
| All 3 canaries pass twice | Promote to 20% rollout |
| Any canary fails | Stop. Diagnose. Fix. Re-run from Canary A. |

---

## 4. Remaining Rollout Cohorts

After all canaries pass twice:

1. **20% rollout:** Mix samples from ChATe27, PVe3, SSTe4. Stop on first new failure class.
2. **60% rollout:** After 20% is clean. Include edge cases (very sparse, very dense, multi-channel).
3. **100% rollout:** After 60% is clean. Every sample completes or fails with an explicit typed reason.

Use `project/scripts/run_batch_manifest.py` with a manifest CSV for cohort execution.

---

## 5. What Changed vs. the Original Rollout Plan

| Original (2026-04-10) | Revised (2026-04-12) |
|---|---|
| Canary set = 3 variants of Sample 35 | Canary set = 3 different samples (35, 41, 44) |
| No cross-animal or cross-marker coverage | Cross-animal (ChATe27) + cross-marker (PV) |
| `atlas_hemisphere` assumed correct | All configs verified as `right_flipped` |
| `ml_flip` evidence from 1 sample | `ml_flip` evidence from 3-sample matrix |
| `fail_score_threshold` = 0.65 (template default) | Calibrated to 0.10 for cross-modality |
| No CI integration test for default path | `test_default_path_smoke.py` added to CI |
| No detect preview in UI | Detect preview API + UI complete |
