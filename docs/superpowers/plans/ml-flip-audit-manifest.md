# ml_flip Evidence Audit Manifest

**Date:** 2026-04-12
**Purpose:** Map every number in `ml-flip-matrix-summary.md` to a specific
retained artifact on disk, so that claims are traceable and reproducible.

---

## Artifact Layout

After running `ml_flip_ab_test.py` with per-sample output directories:

```text
project/outputs/ml_flip_ab/
  sample_35/
    ml_flip_false/ants_registration/registration_metrics.csv
    ml_flip_true/ants_registration/registration_metrics.csv
    summary.csv
    manifest.json
  sample_41/
    ml_flip_false/ants_registration/registration_metrics.csv
    ml_flip_true/ants_registration/registration_metrics.csv
    summary.csv
    manifest.json
  sample_44/
    ml_flip_false/ants_registration/registration_metrics.csv
    ml_flip_true/ants_registration/registration_metrics.csv
    summary.csv
    manifest.json
```

## Current Artifact Status

All three samples have been re-run with the per-sample output layout.
Each sample's heavy outputs coexist on disk and are fully traceable.

### Sample 35 (ChATe27)

| Artifact | Path | Status |
|----------|------|--------|
| Config | `configs/run_config_35.json` | Retained |
| Input dir | `data/35_C0_test` (5 slices) | Retained |
| Summary CSV | `outputs/ml_flip_ab/sample_35/summary.csv` | Retained |
| Manifest | `outputs/ml_flip_ab/sample_35/manifest.json` | Retained |
| False outputs | `outputs/ml_flip_ab/sample_35/ml_flip_false/` | Retained |
| True outputs | `outputs/ml_flip_ab/sample_35/ml_flip_true/` | Retained |

### Sample 41 (ChATe27)

| Artifact | Path | Status |
|----------|------|--------|
| Config | `configs/run_config_41.json` | Retained |
| Input dir | `data/41_C0_test` (5 slices) | Retained |
| Summary CSV | `outputs/ml_flip_ab/sample_41/summary.csv` | Retained |
| Manifest | `outputs/ml_flip_ab/sample_41/manifest.json` | Retained |
| False outputs | `outputs/ml_flip_ab/sample_41/ml_flip_false/` | Retained |
| True outputs | `outputs/ml_flip_ab/sample_41/ml_flip_true/` | Retained |

### Sample 44 (PVe3)

| Artifact | Path | Status |
|----------|------|--------|
| Config | `configs/run_config_44.json` | Retained |
| Input dir | `data/44_C0_test` (5 slices) | Retained |
| Summary CSV | `outputs/ml_flip_ab/sample_44/summary.csv` | Retained |
| Manifest | `outputs/ml_flip_ab/sample_44/manifest.json` | Retained |
| False outputs | `outputs/ml_flip_ab/sample_44/ml_flip_false/` | Retained |
| True outputs | `outputs/ml_flip_ab/sample_44/ml_flip_true/` | Retained |

## Rerun Instructions

To rebuild all evidence with per-sample isolation:

```bash
cd project
python scripts/ml_flip_ab_test.py --sample 35
python scripts/ml_flip_ab_test.py --sample 41
python scripts/ml_flip_ab_test.py --sample 44
```

Each run now writes to `outputs/ml_flip_ab/sample_XX/` and never overwrites
other samples' artifacts.

## Tested Sample Class

All 3 samples are:
- Hemisected LEFT hemisphere tissue
- `atlas_hemisphere: "right_flipped"`
- Cleared-tissue fluorescence vs Allen Nissl template (cross-modality)
- Mouse brain (2 lines: ChATe27, PVe3)

No non-`right_flipped`, whole-brain, or same-modality samples have been tested.
