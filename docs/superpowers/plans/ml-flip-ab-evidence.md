# ml_flip A/B Test Evidence

> **Date:** 2026-04-12
> **Sample:** ChATe27 Sample 35 (fluorescence, right hemisphere)
> **Data:** 5 sparse slices (z200, z225, z250, z300, z350) from `data/35_C0_test`

## Context

The `ml_flip` parameter controls whether the input volume is flipped along the
medial-lateral (ML) axis before ANTs 3D registration.  The Allen CCF convention
places the left hemisphere in the positive X direction; some microscope setups
produce images with the opposite orientation.

Sample 35 uses `atlas_hemisphere: "right_flipped"`, which means the registration
pipeline already handles hemisphere placement independently of `ml_flip`.

## Results

| Metric | ml_flip=False | ml_flip=True | Interpretation |
|--------|--------------|-------------|----------------|
| NMI | **1.0085** | 1.0031 | False slightly better |
| Dice | 0.0283 | **0.0311** | True slightly better |
| SSIM | 0.2485 | **0.3047** | True better |
| NCC | -0.149 | **-0.042** | True better (closer to 0) |
| MSE | 0.375 | **0.370** | True marginally better |
| Cells | 3667 | 3667 | Identical |

## Conclusion

**Neither setting is clearly superior for Sample 35.**  The differences are small
and within run-to-run variability.  This is expected because:

1. `atlas_hemisphere: "right_flipped"` already handles the L/R placement.
2. `ml_flip` has its largest effect on whole-brain (non-hemisected) samples
   where the tissue orientation is ambiguous.

**Default recommendation:** Keep `ml_flip=false` in the template config.
Per-sample configs can override to `true` if needed.

See [`ml-flip-matrix-summary.md`](ml-flip-matrix-summary.md) for the full
3-sample evidence matrix confirming this recommendation.

## Reproduction

```bash
cd project
python scripts/ml_flip_ab_test.py --sample 35 --slices 5
```

Results CSV: `outputs/ml_flip_ab_results.csv`
