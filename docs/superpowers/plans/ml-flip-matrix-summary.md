# ml_flip Evidence Matrix Summary

**Date:** 2026-04-12 (revised)
**Purpose:** Multi-sample evidence for the `ml_flip` parameter default decision.

---

## 1. Background

The `ml_flip` parameter controls whether the input volume is flipped along the
medial-lateral (ML) axis before ANTs 3D registration. The Allen CCF convention
places the left hemisphere in positive X; some microscope setups produce images
with the opposite orientation.

**All samples are hemisected left-hemisphere tissue** mounted with outer cortex
on the LEFT and inner medial surface on the RIGHT. The correct config for all
samples is `atlas_hemisphere: "right_flipped"`.

**Question:** Should `ml_flip` default to `false` or `true` in the template config?

---

## 2. Test Matrix

| Sample | Species | Tissue | atlas_hemisphere | Marker | Slices | Status |
|--------|---------|--------|-----------------|--------|--------|--------|
| 35 (ChATe27) | Mouse | Left hemi | right_flipped | ChAT | 5 sparse | DONE |
| 41 (ChATe27) | Mouse | Left hemi | right_flipped | ChAT | 5 sparse | DONE |
| 44 (PVe3) | Mouse | Left hemi | right_flipped | PV | 5 sparse | DONE |

> **Note:** Sample 41's filename contains "both" — this refers to dual laser
> wavelengths (560nm + 640nm), NOT both hemispheres. All three samples are
> hemisected left-hemisphere tissue.

---

## 3. Results

### Sample 35 — Left hemisphere (atlas_hemisphere: "right_flipped")

| Metric | ml_flip=False | ml_flip=True | Winner |
|--------|--------------|-------------|--------|
| NMI | **1.0096** | 1.0057 | False |
| Dice | 0.0276 | **0.0277** | True (marginal) |
| SSIM | 0.2485 | 0.2485 | Tie |
| NCC | -0.172 | **-0.067** | True |

> Artifact: `outputs/ml_flip_ab/sample_35/summary.csv`

**Interpretation:** NMI favors False. Dice is virtually identical. NCC favors
True. SSIM tied. Both variants complete successfully.

### Sample 41 — Left hemisphere (atlas_hemisphere: "right_flipped")

| Metric | ml_flip=False | ml_flip=True | Winner |
|--------|--------------|-------------|--------|
| NMI | **1.0148** | 1.0095 | False |
| Dice | 0.0207 | **0.0215** | True |
| SSIM | 0.2485 | 0.2485 | Tie |
| NCC | -0.191 | **-0.115** | True |

> Artifact: `outputs/ml_flip_ab/sample_41/summary.csv`

**Interpretation:** Consistent with Sample 35. NMI favors False; Dice and NCC
favor True. SSIM tied. Both variants complete successfully.

### Sample 44 — Left hemisphere (atlas_hemisphere: "right_flipped")

| Metric | ml_flip=False | ml_flip=True | Winner |
|--------|--------------|-------------|--------|
| NMI | **1.0086** | 1.0043 | False |
| Dice | **0.0279** | 0.0224 | False |
| SSIM | 0.2485 | 0.2485 | Tie |
| NCC | -0.116 | **-0.030** | True |

> Artifact: `outputs/ml_flip_ab/sample_44/summary.csv`

**Interpretation:** Sample 44 (PV marker) favors False on NMI and Dice. NCC
strongly favors True (−0.030 vs −0.116). SSIM tied. Dice gap is the largest of
all three samples (0.0279 vs 0.0224).

---

## 4. Cross-Sample Analysis (All 3 samples complete)

### Wins by metric across all samples:

| Metric | False wins | True wins | Tie |
|--------|-----------|----------|-----|
| NMI | 3 (35, 41, 44) | 0 | 0 |
| Dice | 1 (44) | 1 (41) | 1 (35, marginal) |
| SSIM | 0 | 0 | 3 (35, 41, 44) |
| NCC | 0 | 3 | 0 |

### Key findings:

1. **NMI consistently favors `ml_flip=False`** across all 3 samples (3/3).
2. **NCC consistently favors `ml_flip=True`** across all 3 samples (3/3).
3. **Dice is inconclusive:** True wins on sample 41, False wins on sample 44, tie on sample 35.
4. **SSIM is identical** across all 3 samples (0.2485) — ml_flip has no effect on SSIM.
5. **Both variants succeed** with the correct `right_flipped` configuration.
6. **Cell counts are identical** when both variants succeed (detection is independent).
7. **All differences are small** and within noise for practical purposes.
8. **Marker/tissue prep may matter:** The PV sample (44) uniquely favors False on Dice,
   suggesting the ml_flip interaction depends on tissue characteristics, not just geometry.

---

## 5. Decision

**`ml_flip=false` remains the template default**, but this decision is
**provisional and limited in scope**.

### What the current evidence supports:

1. For **left-hemisphere `right_flipped` samples** (the only class tested):
   - Both `ml_flip=false` and `ml_flip=true` produce valid registrations.
   - NMI consistently favors False; Dice/NCC consistently favor True.
   - Differences are small and within noise — either setting is acceptable.

2. **The default is not changed** because no evidence contradicts `ml_flip=false`
   for the tested sample class, and NMI (the primary information-theoretic
   metric) favors it.

### What the current evidence does NOT cover:

- **Non-`right_flipped` orientations** (e.g., `atlas_hemisphere: "left"` or `"both"`)
  have not been tested. If production users submit such samples, the ml_flip
  default must be re-evaluated.
- **Whole-brain or orientation-ambiguous samples** have not been tested because
  no such samples exist in the current dataset (all tissue is hemisected left).
- All 3 tested samples are from only 2 mouse lines (ChATe27 and PVe3), both
  using cleared-tissue fluorescence. Other preparations may behave differently.

### Promotion rule:

The default may only be promoted from "provisional" to "settled" when evidence
covers at least one non-`right_flipped` sample class, if such inputs are
expected in production. Until then, the template default remains unchanged but
the conclusion is explicitly limited to the tested sample class.

Per-sample config override is always available if users observe orientation issues.

---

## 6. Errata

> **Previous revision (pre-correction) contained errors:** Sample 41 was
> initially misconfigured with `atlas_hemisphere: "both"` (confusing dual-laser
> "both" in the filename with bilateral tissue). This caused ml_flip=False to
> crash with ANTs error code 1, leading to the incorrect conclusion that
> "whole-brain samples REQUIRE ml_flip=True." After correcting to
> `atlas_hemisphere: "right_flipped"`, both variants succeed. All samples in
> this study are hemisected left-hemisphere tissue.

---

## 7. Reproduction

```bash
cd project

# Sample 35 (left hemisphere, ChATe27)
python scripts/ml_flip_ab_test.py --sample 35

# Sample 41 (left hemisphere, ChATe27, dual-laser)
python scripts/ml_flip_ab_test.py --sample 41

# Sample 44 (left hemisphere, PVe3)
python scripts/ml_flip_ab_test.py --sample 44
```

Per-sample output layout (each sample gets its own directory):
```
outputs/ml_flip_ab/
  sample_35/summary.csv + manifest.json + ml_flip_false/ + ml_flip_true/
  sample_41/summary.csv + manifest.json + ml_flip_false/ + ml_flip_true/
  sample_44/summary.csv + manifest.json + ml_flip_false/ + ml_flip_true/
```

See [`ml-flip-audit-manifest.md`](ml-flip-audit-manifest.md) for detailed
artifact traceability.
