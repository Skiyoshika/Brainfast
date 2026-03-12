# Reuse Plan from uci-allen-brainrepositorycodegui

## Directly reusable
- `run_registration_cellcounting.py`: registration + counting flow scaffold
- `cell_regions.py`, `count_labels.py`: region hierarchy aggregation logic
- `tif_to_nii.py`: image stack to NII conversion

## Adaptation required
- Replace legacy threshold detection with `cellpose` (+ `LoG` fallback)
- Add KDTree 3D dedup (threshold based on slice spacing)
- Standardize outputs:
  - `cell_counts_leaf.csv`
  - `cell_counts_hierarchy.csv`
  - `slice_qc.csv`, `dedup_stats.csv`

## Not reused as-is
- Huge `requirements.txt` (too heavy). Build minimal env for RTX 3070 Ti.
