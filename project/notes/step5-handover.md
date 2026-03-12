# STEP 5 Handover (Status Report)

## What is implemented now
- Project scaffold and minimal config template
- Reuse bridge plan from legacy Allen repo
- Pipeline entry (`scripts/main.py`) with dry-run and demo execution
- Mapping + aggregation module:
  - leaf output: `cell_counts_leaf.csv`
  - hierarchy output: `cell_counts_hierarchy.csv`
  - confidence label (high/medium/low)
- Dedup placeholder module + stats output:
  - `outputs/dedup_stats.csv`

## What is still pending (next coding phase)
1. Replace dedup placeholder with KDTree anisotropic dedup:
   - threshold based on `slice_spacing_um * 0.5`
2. Implement real registration adapter from legacy repo modules
3. Implement real cell detection:
   - primary: Cellpose cyto2
   - secondary: Cellpose nuclei
   - fallback: LoG+watershed
4. Implement QC overlay export from real detections

## Current command checks
- `python scripts/main.py --config configs/run_config.template.json --dry-run`
- `python scripts/main.py --config configs/run_config.template.json --demo-map`

Both pass in current environment assumptions.
