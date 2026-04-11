# Brainfast Demo And Batch Runbook

## What Is Actually Validated

- The histology/image-analysis sub-pipeline is reproducible enough for demo and data production.
- It is not a full reproduction of the entire AAV toolbox paper.
- The strongest validated artifacts in this workspace are:
  - `project/outputs/codex_demo_35_full`
  - `project/outputs/codex_smoke_35_test_autodemo`

## Demo Plan For Advisor Meeting

### Demo A: Open the precomputed full run

1. Start one server only:

```powershell
python project/frontend/server.py
```

2. Open:

```text
http://127.0.0.1:8787
```

3. Show these tabs in order:

- `Results`
- `Batch QC Review`

4. Use this output folder as the main artifact:

```text
D:\Brainfast\project\outputs\codex_demo_35_full
```

5. Files worth highlighting:

- `paper_report/paper_report_summary.txt`
- `paper_aav_region_summary.csv`
- `cell_counts_hierarchy.csv`
- `demo_panel.jpg`
- `demo_best_slice.jpg`
- `demo_annotated_slice.jpg`

### Demo B: Live smoke run if needed

This is the fastest real-input demonstration currently verified in this repo.

```powershell
python project/scripts/main.py `
  --config project/configs/run_config_35.json `
  --run-real-input project/data/35_C0_test `
  --output-name live_demo_35_test `
  --debug
```

Expected runtime on this machine:

- about 5 to 6 minutes for `35_C0_test`

Expected output folder:

```text
D:\Brainfast\project\outputs\live_demo_35_test
```

## Guardrails

- Do not start multiple `server.py` processes on port `8787`.
- If the port is already occupied, the server now fails immediately with a clear error.
- Fresh runs now auto-generate demo assets into the active run folder instead of the stale default outputs root.

## Batch Processing For 9 Brains

### Step 1: Copy the manifest template

Template:

```text
D:\Brainfast\project\configs\batch_manifest.template.csv
```

Create your own manifest and fill one row per brain TIFF folder:

- `sample_id`: short stable name for the brain
- `input_dir`: folder containing that brain's TIFFs
- `config`: optional per-brain config, or leave blank and use `--default-config`
- `output_name`: optional display/run name
- `output_dir`: optional explicit output directory

### Step 2: Dry-run the batch

```powershell
python project/scripts/run_batch_manifest.py `
  --manifest D:/your_manifest/brains.csv `
  --default-config D:/Brainfast/project/configs/run_config_35.json `
  --dry-run
```

### Step 3: Execute the batch

```powershell
python project/scripts/run_batch_manifest.py `
  --manifest D:/your_manifest/brains.csv `
  --default-config D:/Brainfast/project/configs/run_config_35.json
```

If you want later rows to continue even when one job fails:

```powershell
python project/scripts/run_batch_manifest.py `
  --manifest D:/your_manifest/brains.csv `
  --default-config D:/Brainfast/project/configs/run_config_35.json `
  --keep-going
```

## Minimal Preflight Before Real Thesis Data

Run this once before launching the 9-brain batch:

```powershell
python project/scripts/check_env.py --config project/configs/run_config_35.json
python -m pytest project/tests/unit -v
python -m pytest project/tests/test_regression_suite.py -v
```

## Known Scope Limits From The Handoff

- Paper preset/SOP is still not fully encoded as a locked production preset.
- Colocalization and representative-slice logic are not yet a full paper-faithful reproduction.
- Real paper-style validation on all final thesis datasets still needs to be done on your own 9-brain cohort.
