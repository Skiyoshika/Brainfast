# Brainfast Reproduction Guide

This document gives a compact end-to-end reproduction path for the current
repository state. It covers both a software-only verification path and a local
sample rerun path.

## 1. Create a clean Python environment

From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# Minimal 2D usage (no Cellpose, no ANTs):
pip install -e ".[dev]"

# Cellpose-enabled detection:
pip install -e ".[advanced,dev]"

# Whole-brain 3D registration (requires ANTs):
pip install -e ".[wholebrain,dev]"

# Everything:
pip install -e ".[full,dev]"
```

Expected result:

- Python 3.10+ is active
- Flask, NumPy, SciPy, scikit-image, tifffile, nibabel, pytest, and ruff are installed
- Cellpose is installed only if `advanced` or `full` extra was selected
- ANTsPy (`import ants`) is installed only if `wholebrain` or `full` extra was selected

## 2. Validate the runtime and required assets

```powershell
python project\scripts\check_env.py --config project\configs\run_config.template.json
```

This step verifies:

- Python version
- required Python modules
- required repository assets such as `annotation_25.nii.gz`
- config syntax and semantic checks

If this command fails, fix the reported environment or asset issue before
continuing.

If the atlas annotation file is missing, you can bootstrap the required atlas
assets first:

```powershell
python project\download_atlas.py --ensure
```

## 3. Run the software-level regression suite

```powershell
python -m pytest project\tests\test_regression_suite.py -v
python -m pytest project\tests\unit -v
```

This reproduces the software behavior without requiring a full microscopy
dataset. The regression suite exercises atlas auto-pick, overlay rendering,
mapping, deduplication, aggregation, and the current frontend service layer.

## 4. Re-run a representative 2D pipeline on a local slice folder

Create a local config by copying the template and editing the input paths:

```powershell
Copy-Item project\configs\run_config.template.json local.run_config.json
```

Edit `local.run_config.json` and set at least:

- `project.name`
- `input.slice_dir`
- `input.pixel_size_um_xy`
- `input.slice_spacing_um`
- `input.channel_map`
- `input.active_channel`

Then run the pipeline:

```powershell
python project\scripts\main.py --config local.run_config.json --run-real-input "D:\path\to\your\slice_folder"
```

Expected primary outputs:

- `project\outputs\cells_detected.csv`
- `project\outputs\cells_dedup.csv`
- `project\outputs\cells_mapped.csv`
- `project\outputs\cell_counts_leaf.csv`
- `project\outputs\cell_counts_hierarchy.csv`
- `project\outputs\slice_registration_qc.csv`
- `project\outputs\qc_overlays\`

## 5. Inspect the result through the UI and the output folder

Start the local UI:

```powershell
python project\frontend\server.py
```

Then open:

```text
http://127.0.0.1:8787
```

## Whole-brain 3D path

If the configuration uses `registration.scope = whole` and `registration.whole_brain_backend = miki_3d`, the automatic run becomes volume-first. The main outputs to expect are:

- `outputs/volume/input_volume.nii.gz`
- `outputs/template_prep/template_half.nii.gz`
- `outputs/template_prep/annotation_half.nii.gz`
- `outputs/ants_registration/ants_result.nii.gz`
- `outputs/ants_registration/annotation_registered.nii.gz`
- `outputs/laplacian_refinement/final_registered.nii.gz`
- `outputs/truth_export/slice_*_registered_label.tif`
- `outputs/truth_export/slice_*_overlay.png`
- `outputs/slice_registration_qc.csv`
- `outputs/volume_registration_qc.csv`

Review at least:

- registration preview and QC slices
- `Cell Counts by Brain Region`
- exported methods text from `/api/export/methods-text`
- output CSV files in `project\outputs\`

## Optional local sample note

If your local workspace includes the untracked `Sample\Miki` reference folder,
use it as a visual reference dataset for 3D registration outputs. That folder is
helpful for manual comparison, but it is not required for the software-level
reproduction path above.
