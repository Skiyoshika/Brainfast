# Brainfast

Chinese documentation index: [docs/README.zh-CN.md](docs/README.zh-CN.md)

## Index
- [What Brainfast Does](#what-brainfast-does)
- [Current Scope](#current-scope)
- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Setup](#setup)
- [Quick Start](#quick-start)
- [Main Workflows](#main-workflows)
- [Key Outputs](#key-outputs)
- [Cell Counting Notes](#cell-counting-notes)
- [Testing](#testing)
- [Other Documents](#other-documents)
- [License](#license)

## What Brainfast Does

Brainfast is a local workflow for atlas registration and cell counting from microscopy TIFF data.

It covers four practical jobs:
- 2D slice registration against the Allen atlas
- manual review and calibration
- 3D volume registration with report generation
- Cellpose-based cell detection, deduplication, region mapping, and count aggregation

The project is built for a single workstation. It is not a multi-user service.

## Current Scope

The repository currently supports two main processing paths:

1. A slice-based pipeline for detection, mapping, and region counts.
2. A 3D registration pipeline that generates per-run reports, overview images, and QC metrics.

The desktop UI is the main entry point for day-to-day use. The Python scripts remain the most direct way to validate a run, reproduce results, and debug pipeline behavior.

## Repository Layout

| Path | Purpose |
| --- | --- |
| `configs/` | Runtime configs, atlas metadata, and sample configs |
| `data/` | Local sample slices used for tests and demos |
| `docs/` | Additional workflow notes |
| `frontend/` | Flask app, static UI, desktop launcher scripts |
| `outputs/` | Registration reports, counts, QC files, and temporary runtime artifacts |
| `scripts/` | Registration, detection, mapping, report generation, and utility scripts |
| `tests/` | Unit and integration tests |
| `train_data_set/` | Saved calibration samples used for parameter learning |

## Requirements

- Python 3.10 or newer
- Windows is the primary target environment
- An NVIDIA GPU is strongly recommended for Cellpose-based counting
- Atlas assets in this repository, including `annotation_25.nii.gz` and the Allen structure CSV/JSON files

## Setup

All commands below assume the repository root, for example `D:\Brainfast`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[advanced,dev]"
```

Notes:
- `advanced` installs the optional packages used for Cellpose, ANTs, and SimpleITK-based processing.
- `dev` installs the test and lint tools.
- If you only need the UI and basic scripts, `pip install -e .` is enough.

## Quick Start

### 1. Validate the environment

```powershell
python project\scripts\check_env.py --config project\configs\run_config.template.json
```

### 2. Start the UI

Windows launcher:

```powershell
.\Start_Brainfast.bat
```

Direct Python entry:

```powershell
python project\frontend\server.py
```

Then open `http://127.0.0.1:8787`.

### 3. Run a sample 2D pipeline from the command line

```powershell
python project\scripts\main.py --config project\configs\run_config_35_quick.json
```

### 4. Run a sample 3D registration

```powershell
python project\scripts\run_3d_registration.py --config project\configs\run_config_3d_ants_sample.json
```

## Main Workflows

### UI Workflow

1. Start Brainfast.
2. Choose the input slices or volume.
3. Run registration or counting from the UI.
4. Review the Results page, registration report cards, and QC plots.
5. Use manual correction tools only when the automatic result is not good enough.

### 2D Slice Pipeline

The slice pipeline does the following:

1. Extract the active channel from source TIFF files.
2. Register each slice against the atlas.
3. Detect cells on the selected slices.
4. Deduplicate detections across neighboring slices.
5. Map detections into atlas regions.
6. Write leaf-level counts, hierarchy counts, and QC files.

### 3D Registration Pipeline

The 3D path does the following:

1. Convert a stack TIFF or a `z*.tif` folder into a NIfTI volume.
2. Crop the Allen template and annotation to the target AP range and hemisphere.
3. Register the volume with ANTs or Elastix.
4. Optionally refine the result with the Laplacian step.
5. Generate a report page, an overview image, metrics, and staining statistics.

### Manual Calibration Loop

Manual review is part of the workflow, not an afterthought.

- Use liquify and landmark tools when automatic alignment is close but not acceptable.
- Save useful corrections into `train_data_set/`.
- Re-run parameter learning when the sample set becomes large enough to justify it.

## Key Outputs

| File or folder | Purpose |
| --- | --- |
| `outputs/cells_detected.csv` | Raw detections before cross-slice deduplication |
| `outputs/cells_dedup.csv` | Deduplicated detections |
| `outputs/cells_mapped.csv` | Detections mapped to atlas regions |
| `outputs/cell_counts_leaf.csv` | Counts at the leaf region level |
| `outputs/cell_counts_hierarchy.csv` | Counts aggregated through the Allen structure tree |
| `outputs/detection_summary.json` | Detector choice, sampling mode, and detection totals |
| `outputs/detection_samples/` | Three real slice overlays used as confidence checks for counting |
| `outputs/slice_registration_qc.csv` | Per-slice registration scores and timing |
| `outputs/index.html` | Generated index of 3D registration reports |
| `outputs/<run_name>/report.html` | Detailed report page for one 3D registration run |

The same report content is also surfaced in the Results page inside the frontend.

## Cell Counting Notes

- The current default path uses Cellpose on single slices, not on merged slice averages.
- If a config explicitly requests Cellpose and Cellpose is unavailable, the run should fail instead of silently switching to a blob detector.
- GPU execution is strongly preferred. CPU execution is possible, but full-size raw slices are slow.
- The Results page includes three sample detection overlays so you can inspect whether the count looks believable before trusting the totals.

## Testing

Run the unit tests:

```powershell
pytest project\tests\unit -v
```

Run the full test suite:

```powershell
pytest project\tests -v
```

A practical validation pass before a real run:

```powershell
python project\scripts\check_env.py --config project\configs\run_config.template.json --require-input-dir
```

## Other Documents

- [Chinese documentation index](docs/README.zh-CN.md)
- [Frontend notes](frontend/README.md)
- [Internal 3D sample workflow](docs/internal_3d_sample_workflow.md)
- [Root repository landing page](../README.md)

## License

Brainfast is distributed under the GNU AGPL-3.0 license. See [../LICENSE](../LICENSE) for the full text.
