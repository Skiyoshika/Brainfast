# Canary Run Operator Checklist

> Fill out one copy per canary run.  Two passes per canary required before
> promoting to Task 6.

---

## Run Metadata

| Field | Value |
|---|---|
| **Date** | |
| **Operator** | |
| **Canary bucket** | A / B / C |
| **Run number** | 1st / 2nd |
| **Machine** | |
| **Python version** | |
| **Install command** | `pip install -e ".[wholebrain,advanced]"` |
| **Config file** | |
| **Sample data path** | |
| **Output path** | |
| **Number of input slices** | |

---

## Pre-Run Checks

- [ ] `python project/scripts/check_env.py --config <config>` exits 0
- [ ] `python -m ruff check project/scripts/` exits 0
- [ ] `python -m pytest project/tests/unit/ -q` all green

---

## Run Command

```bash
cd D:\Brainfast\project
python scripts/main.py --config configs/<config_file> --run-real-input <data_path> --output-dir outputs/<canary_output_dir>
```

---

## Post-Run Gates

### Gate 1: No Hard Crash
- [ ] Pipeline exited with code 0
- [ ] `pipeline_progress.json` shows `percent: 100` and final stage complete

### Gate 2: Non-Empty QC Outputs
- [ ] `ants_registration/registration_metrics.csv` exists with 6 metric rows
- [ ] `truth_export/` contains at least 1 annotated slice file
- [ ] `volume/` contains the merged 3D volume `.nii.gz`

### Gate 3: Registration Quality
- [ ] Dice ≥ 0.70 (relaxed to ≥ 0.50 for Canary B)
- [ ] SSIM ≥ 0.05
- [ ] NMI ≥ 1.01
- [ ] No NaN or Inf values in any metric

Record actual values:

| Metric | Value | Pass? |
|---|---|---|
| NCC | | |
| NMI | | |
| SSIM | | |
| Dice | | |
| MSE | | |
| PSNR | | |

### Gate 4: Truth Label Coverage
- [ ] At least 1 truth slice has non-zero annotation labels
- [ ] Annotation coverage ≥ 15% (relaxed to ≥ 5% for Canary B)
- [ ] At least 3 distinct Allen CCF structure IDs present

### Gate 5: Cell Detection Sanity
- [ ] `cells_mapped.csv` exists with > 0 rows
- [ ] Detector used: `cpsam` / `LoG fallback` (note which)
- [ ] Cell count > 0 for tissue-containing slices

### Gate 6: Orientation Spot Check
- [ ] Cortical areas on lateral side (LEFT for `right_flipped`)
- [ ] Subcortical structures in center
- [ ] No left-right or AP inversion visible

### Gate 7: Reproducibility (2nd run only)
- [ ] Registration metrics match 1st run within ε < 1e-6
- [ ] Cell count within ±5% of 1st run
- [ ] Same pipeline stage completion pattern

---

## Result

| | |
|---|---|
| **Overall** | PASS / FAIL |
| **Failure reason** | |
| **Notes** | |

---

## Canary Promotion Tracker

| Canary | Run 1 | Run 2 | Promoted? |
|---|---|---|---|
| A (demo, 111 slices) | | | |
| B (test, 5 slices) | | | |
| C (full, 100 slices) | | | |

**All 6 passes recorded → proceed to Task 6 (20% rollout)**
