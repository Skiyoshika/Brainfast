# IdleBrain MVP

## Quick start
```bash
python scripts/main.py --config configs/run_config.template.json --make-sample-tiff outputs/sample_input
python scripts/main.py --config configs/run_config.template.json --init-registration
python scripts/main.py --config configs/run_config.template.json --run-real-input outputs/sample_input
```

## Current outputs
- `outputs/cells_detected.csv`
- `outputs/cells_dedup.csv`
- `outputs/cells_mapped.csv`
- `outputs/cell_counts_leaf.csv`
- `outputs/cell_counts_hierarchy.csv`
- `outputs/dedup_stats.csv`
- `outputs/slice_qc.csv`
- `outputs/qc_overlays/*.png`

## Notes
- Real Allen label query is wired via `scripts/atlas_mapper.py`.
- If label volume is absent, mapper uses fallback region buckets for pipeline continuity.
- Detection uses Cellpose primary route with fallback detector.

