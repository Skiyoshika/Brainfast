# Brainfast Manual Acceptance

## Browser acceptance

1. Start the server:
   ```powershell
   python project/frontend/server.py
   ```
2. Open `http://127.0.0.1:8787`.
3. Load a known-good sample config.
4. Run detector preview once and confirm:
   - overlay renders
   - CSV download works
   - detector name is shown
5. Run one whole-brain pipeline sample and confirm:
   - QC overlay renders
   - registration CSVs exist
   - cell count CSVs download
   - methods/export output is non-empty

## Closed-loop acceptance (v1 release gate)

The three learning loops must each round-trip cleanly before a release is
signed off. The synthetic hosted smoke suite
(`project/tests/integration/test_default_path_smoke.py`) guards the
calibration loop automatically; the remaining items are manual.

### 1. Calibration learn → next-run uptake

1. Launch Brainfast on the sample you just ran.
2. Open the 2D overlay editor on at least one slice that shows a visible
   atlas-vs-tissue offset.
3. Use the `Save Calibration + Learn` button.
4. Wait for the learn status to report `ok=True`. Confirm the shared
   JSON exists:
   ```
   outputs/state/calibration/trainset_tuned_params.json
   ```
5. Launch a *fresh* run against the same sample (new `jobId`). Confirm:
   - `outputs/jobs/<new_job>/trainset_tuned_params.json` is a snapshot
     of the shared JSON (captured at run start for reproducibility).
   - `outputs/jobs/<new_job>/truth_export/slice_*_overlay.png` differs
     from the pre-learn baseline. The difference must match the
     learned `fit_mode` + `edge_smooth_iter` values.
6. Hosted CI regression:
   `pytest project/tests/integration/test_default_path_smoke.py::
   TestDefaultRuntimePath::test_learned_calibration_reaches_default_whole_brain_path`

### 2. Class-prior auto-seed

1. Accumulate `MIN_SAMPLES_FOR_APPLY` (default 3) runs of the same class
   (e.g. `ChATe27`) via `Save job → class prior` in 3D Liquify.
2. Start a fresh 4th same-class job and open 3D Liquify. Confirm:
   - auto-detected class is populated in the `Class` input,
   - the banner reads "✓ <class>: auto-applied N prior landmark(s) to
     this empty job.",
   - the landmark table is non-empty without any button click.
3. Repeat with a job that already has manual pairs. Confirm:
   - the banner reads "... manual overwrite required ..."
   - the job's manual pairs remain intact.
4. Hosted CI regressions:
   `pytest project/tests/unit/test_class_prior.py
   project/tests/unit/test_frontend_regressions.py -k auto_warm_start`

### 3. Liquify finalize

1. On any completed whole-brain run, add ≥ 5 landmark pairs distributed
   across z in 3D Liquify.
2. Click `Apply 3D warp (Laplacian)`; wait for completion.
3. Click `Finalize & re-export cell counts`; confirm:
   - `cell_counts_hierarchy_liquify3d.csv` exists and is different from
     the pre-liquify hierarchy file,
   - `cells_mapped_liquify3d.csv` exists,
   - `annotation_refined_liquify3d.nii.gz` exists.

### 4. Cellpose training-tab smoke

1. Open the `Cellpose Model Training` tab.
2. Confirm the page lists `outputs/state/cellpose_training/` as the data
   source (not `<project_root>/cellpose_training/`).
3. If ≥ 1 training sample already exists there, confirm the training
   button is enabled and `Apply Model` selector lists any custom
   checkpoints.

### 5. State hygiene

1. Confirm no mutable state has been written under
   `<project_root>/train_data_set/` or `<project_root>/cellpose_training/`
   during this session. All new artifacts must be under
   `outputs/state/`.
2. If `BRAINFAST_STATE_DIR` was set during the session, confirm all the
   above artifacts live under that root instead.

## Sign-off checklist

- [ ] Calibration learn round-trip verified (Step 1 above).
- [ ] Calibration next-run uptake verified (Step 1 above).
- [ ] Class-prior auto-seed verified (Step 2 above).
- [ ] Liquify finalize verified (Step 3 above).
- [ ] Cellpose training-tab smoke check verified (Step 4 above).
- [ ] Shared-state hygiene verified (Step 5 above).
- [ ] `pytest project/tests/unit` is green on the release branch.
- [ ] `ruff check` + `ruff format --check` are clean on
      `project/scripts`, `project/frontend/blueprints`,
      `project/frontend/server_context.py`,
      `project/frontend/app_metadata.py`,
      `project/frontend/update_checker.py`.
