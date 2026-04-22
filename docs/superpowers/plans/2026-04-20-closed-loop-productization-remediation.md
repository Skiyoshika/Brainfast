# Closed-Loop Calibration and Productization Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the April 20 review gaps so calibration learning, class-prior reuse, detector-training documentation, and release gates all match the shipped `whole + miki_3d + cpsam` product behavior.

**Architecture:** First establish one shared runtime-state and learned-artifact path contract, then thread that contract through the default whole-brain path so a UI-learned calibration actually affects later runs. After the backend contract is correct, automate safe class-prior warm-starting in the UI, move mutable data out of the source tree, align docs to the real feature set, and expand CI/manual acceptance so the closed loop is proven instead of implied.

**Tech Stack:** Python 3.10/3.11, Flask blueprints, browser JavaScript in `project/frontend/app.js`, `pytest`, `ruff`, GitHub Actions.

---

## Product Contract This Plan Enforces

- `Save Calibration + Learn` is treated as a learned overlay/truth-export tuning loop, not detector training and not ANTs model training.
- Learned calibration must affect the default shipped whole-brain path by changing the exported registered-label rasters used for QC and region mapping.
- Class priors must auto-seed a new job when confidence is high enough and the job has no manual pairs yet.
- Detector retraining remains a separate workflow (`Mask Editor` -> `Save to Training Set` -> `Cellpose Model Training` -> `Apply Model`) and must be documented as such.
- Mutable calibration, prior, and training artifacts must live under a writable runtime-state directory, not under tracked source directories.

## File Map

**Core path and state contract**
- Modify: `project/scripts/paths.py`
- Modify: `project/frontend/server_context.py`
- Modify: `project/frontend/blueprints/api_overlay.py`
- Modify: `project/frontend/blueprints/api_liquify_3d.py`
- Modify: `project/frontend/blueprints/api_cellpose.py`

**Whole-brain closed loop**
- Modify: `project/scripts/main.py`
- Modify: `project/scripts/whole_brain_3d.py`
- Modify: `project/scripts/truth_export_3d.py`
- Modify: `project/scripts/liquify_3d_finalize.py`

**Class-prior UX**
- Modify: `project/scripts/class_prior.py`
- Modify: `project/frontend/app.js`

**Tests**
- Modify: `project/tests/unit/test_api_pipeline.py`
- Modify: `project/tests/unit/test_main.py`
- Modify: `project/tests/unit/test_truth_export_3d.py`
- Modify: `project/tests/unit/test_whole_brain_3d.py`
- Modify: `project/tests/unit/test_api_liquify_3d.py`
- Modify: `project/tests/unit/test_class_prior.py`
- Modify: `project/tests/unit/test_mask_endpoints.py`
- Modify: `project/tests/unit/test_frontend_regressions.py`
- Modify: `project/tests/integration/test_default_path_smoke.py`

**Docs and release gates**
- Modify: `README.md`
- Modify: `docs/user_guide.md`
- Modify: `docs/release/known-limitations.md`
- Modify: `docs/release/manual-acceptance.md`
- Modify: `.github/workflows/test.yml`

**Hygiene**
- Modify: `project/frontend/blueprints/api_pipeline.py`

---

### Task 1: Establish One Shared Runtime-State Contract

**Files:**
- Modify: `project/scripts/paths.py`
- Modify: `project/frontend/server_context.py`
- Modify: `project/frontend/blueprints/api_liquify_3d.py`
- Modify: `project/frontend/blueprints/api_cellpose.py`
- Test: `project/tests/unit/test_api_pipeline.py`
- Test: `project/tests/unit/test_api_liquify_3d.py`
- Test: `project/tests/unit/test_mask_endpoints.py`

- [ ] **Step 1: Write failing path-contract tests**

Add coverage that `RunPaths.from_project_root(..., outputs_dir=job_out)` exposes a shared runtime-state root and does not point mutable artifacts back into `PROJECT_ROOT / "train_data_set"` or `PROJECT_ROOT / "cellpose_training"`.

Run: `python -m pytest project/tests/unit/test_api_pipeline.py -q`
Expected: new assertions fail until shared-state paths exist.

- [ ] **Step 2: Define the shared state layout in `RunPaths`**

Extend `RunPaths` with explicit shared-state paths rooted at `project_root / "outputs" / "state"` by default, with `BRAINFAST_STATE_DIR` as an override:

```text
outputs/state/
  calibration/
    samples/
    trainset_tuned_params.json
  class_priors/
  cellpose_training/
```

The goal is one canonical place for learned artifacts that must survive across jobs.

- [ ] **Step 3: Migrate backend callers to the shared state layout**

Update:
- `server_context._save_calibration_pair()` to write calibration samples under `outputs/state/calibration/samples`
- `server_context._learn_from_trainset_async()` to read/write the shared calibration directory and shared tuned JSON
- `api_liquify_3d._class_priors_root()` to use `outputs/state/class_priors`
- `api_cellpose._training_dir()` to use `outputs/state/cellpose_training`

Do not keep compatibility writes into the old source-tree locations.

- [ ] **Step 4: Preserve backward readability where needed**

If migration is needed for existing local data, read from legacy locations only as a fallback and log a clear migration message, but always write to the new shared state location.

- [ ] **Step 5: Verify the state contract**

Run:
- `python -m pytest project/tests/unit/test_api_pipeline.py -q`
- `python -m pytest project/tests/unit/test_api_liquify_3d.py -q`
- `python -m pytest project/tests/unit/test_mask_endpoints.py -q`

Expected: the new tests pass and endpoint tests create runtime artifacts only under `outputs/state/...`.

**Definition of done**
- Calibration samples, class priors, and Cellpose training samples no longer dirty tracked source directories.

---

### Task 2: Make Learned Calibration Reach the Default Whole-Brain Path

**Files:**
- Modify: `project/scripts/main.py`
- Modify: `project/scripts/whole_brain_3d.py`
- Modify: `project/scripts/truth_export_3d.py`
- Modify: `project/scripts/liquify_3d_finalize.py`
- Modify: `project/frontend/server_context.py`
- Test: `project/tests/unit/test_main.py`
- Test: `project/tests/unit/test_truth_export_3d.py`
- Test: `project/tests/unit/test_whole_brain_3d.py`
- Test: `project/tests/integration/test_default_path_smoke.py`

- [ ] **Step 1: Write failing tests for learned-parameter resolution**

Add unit coverage for:
- `_load_tuned_overlay_params(...)` falling back from `outputs/jobs/<job_id>/trainset_tuned_params.json` to the shared calibrated JSON
- the default `scope=whole` + `whole_brain_backend=miki_3d` path carrying learned `warp_params`, `fit_mode`, and `edge_smooth_iter` into truth export

Run:
- `python -m pytest project/tests/unit/test_main.py -q`
- `python -m pytest project/tests/unit/test_whole_brain_3d.py -q`

Expected: new assertions fail before the code path is wired through.

- [ ] **Step 2: Resolve tuned params before the whole-brain early return**

Refactor `project/scripts/main.py` so tuned parameters are loaded before the `run_whole_brain_3d(...)` return path. The whole-brain branch must receive:
- `warp_params`
- `fit_mode`
- `edge_smooth_iter`

This removes the current bypass where the default path never sees learned calibration.

- [ ] **Step 3: Remove hard-coded truth-export defaults**

Update `export_registered_truth_slices(...)` and the callers in:
- `project/scripts/whole_brain_3d.py`
- `project/scripts/liquify_3d_finalize.py`

so they accept caller-provided `fit_mode` and `edge_smooth_iter` instead of hard-coding `"cover"` and `0`.

- [ ] **Step 4: Snapshot the resolved tuned params into each job output**

When a job starts, copy or materialize the resolved tuned JSON into that job's output directory for reproducibility. Reading should use:
1. job-local tuned JSON if present
2. shared tuned JSON otherwise

Writing a job-local snapshot avoids "mystery behavior" when a later shared learning run changes the global artifact.

- [ ] **Step 5: Prove the closed loop in the hosted smoke path**

Extend `project/tests/integration/test_default_path_smoke.py` so the synthetic default-path run verifies that learned truth-export parameters are consumed on the shipped whole-brain path.

Run:
- `python -m pytest project/tests/unit/test_main.py -q`
- `python -m pytest project/tests/unit/test_truth_export_3d.py -q`
- `python -m pytest project/tests/unit/test_whole_brain_3d.py -q`
- `python -m pytest project/tests/integration/test_default_path_smoke.py -q`

Expected: a learned parameter artifact changes the exported registered-label contract on the default whole-brain path.

**Definition of done**
- A UI-triggered calibration learn result is consumed by the next default whole-brain run and affects the truth-export rasters used for mapping/QC.

---

### Task 3: Auto-Apply Class Priors Without Overwriting Manual Work

**Files:**
- Modify: `project/frontend/app.js`
- Modify: `project/scripts/class_prior.py`
- Test: `project/tests/unit/test_api_liquify_3d.py`
- Test: `project/tests/unit/test_class_prior.py`
- Test: `project/tests/unit/test_frontend_regressions.py`

- [ ] **Step 1: Write failing tests for automatic warm-start behavior**

Add coverage for:
- auto-detected class + ready prior + empty job state -> warm-start is applied automatically
- existing manual landmark pairs -> no forced overwrite
- prior banner text distinguishes `ready`, `auto-applied`, and `manual overwrite required`

Run:
- `python -m pytest project/tests/unit/test_api_liquify_3d.py -q`
- `python -m pytest project/tests/unit/test_frontend_regressions.py -q`

Expected: new regression assertions fail before the UI flow is added.

- [ ] **Step 2: Make `ClassPriorStore.update()` iterator-safe**

Convert `pairs` to a list once inside `ClassPriorStore.update()` before iterating and logging. Add a unit test that passes an iterator instead of a list and asserts `pair_count` remains correct.

Run: `python -m pytest project/tests/unit/test_class_prior.py -q`
Expected: the new iterator test fails before the fix and passes after it.

- [ ] **Step 3: Add a single auto-warm-start client flow**

In `project/frontend/app.js`, introduce one client-side function that runs when:
- a class is auto-detected or manually selected
- liquify state has been refreshed
- the job has zero existing landmark pairs

That function should:
- check prior status
- auto-apply warm-start when `ready_for_warm_start` is true
- update the banner to show what happened
- never auto-force overwrite an existing job

- [ ] **Step 4: Keep the manual override path**

Retain the explicit `Warm-start from prior` button. When a job already has manual pairs, the button can still offer the current confirmation flow with `force=true`.

- [ ] **Step 5: Verify the UX contract**

Run:
- `python -m pytest project/tests/unit/test_api_liquify_3d.py -q`
- `python -m pytest project/tests/unit/test_class_prior.py -q`
- `python -m pytest project/tests/unit/test_frontend_regressions.py -q`

Expected: auto-seeding happens once for empty jobs, manual work is not clobbered, and iterator logging stays correct.

**Definition of done**
- A mature class prior actually reduces operator work on the next same-class job without requiring an extra button click.

---

### Task 4: Align the Public Product Contract With the Real Feature Set

**Files:**
- Modify: `README.md`
- Modify: `docs/user_guide.md`
- Modify: `docs/release/known-limitations.md`

- [ ] **Step 1: Rewrite the workflow-boundary section in `README.md`**

Make the distinctions explicit:
- `Save Calibration + Learn` tunes overlay/truth-export behavior
- 3D liquify is a landmark-based correction/finalization loop
- detector retraining exists in the UI
- default runtime remains `miki_3d + cpsam`

- [ ] **Step 2: Fix stale install and startup instructions in `docs/user_guide.md`**

Replace the current stale statements:
- `https://github.com/<org>/Brainfast.git`
- `pip install -e .`
- `pip install -e ".[advanced]"`
- `project/frontend/StartIdleBrainTrial.bat`
- `primary_model: log or cellpose`
- `python frontend/server.py --port 8788`

with the actual current product contract and the `BRAINFAST_PORT` environment-variable override.

- [ ] **Step 3: Rewrite `known-limitations.md` to match the post-fix behavior**

After Tasks 1-3 land, the limitations file must say what still is not closed-loop, rather than repeating the old "2D preview only" statement if that is no longer true.

- [ ] **Step 4: Add one feature matrix**

Document these three separate learning loops so operators stop conflating them:
- calibration learn
- class-prior warm-start
- Cellpose retraining

- [ ] **Step 5: Verify the docs no longer contain stale claims**

Run:
- `Select-String -Path README.md,docs/user_guide.md,docs/release/known-limitations.md -Pattern '<org>/Brainfast|\\[advanced\\]|StartIdleBrainTrial|--port 8788|does not yet ship a detector-specific manual relabel / retrain workflow'`

Expected: no stale matches remain.

**Definition of done**
- The docs describe exactly what the product does today, with no false negatives and no false promises.

---

### Task 5: Rebuild Release Gates Around the Supported Closed Loop

**Files:**
- Modify: `.github/workflows/test.yml`
- Modify: `docs/release/manual-acceptance.md`
- Modify: `project/tests/integration/test_default_path_smoke.py`

- [ ] **Step 1: Add hosted coverage for the closed-loop regression**

Use the existing synthetic smoke framework to verify:
- a learned calibration artifact is consumed by the default whole-brain path
- truth export still succeeds
- class-prior auto-apply remains unit-tested

This keeps the critical behavior under GitHub-hosted CI instead of depending on the disabled self-hosted integration job.

- [ ] **Step 2: Keep self-hosted integration optional, but stop relying on it for core assurance**

Do not block the plan on removing `if: false` from the self-hosted integration job unless the runner is actually available. The supported path must already be defended by hosted tests.

- [ ] **Step 3: Expand manual acceptance**

Update `docs/release/manual-acceptance.md` to require:
- one `Save Calibration + Learn` round-trip
- one next-run verification that the learned result is loaded
- one class-prior auto-seed verification
- one liquify finalize verification
- one Cellpose training-tab smoke check

- [ ] **Step 4: Verify the release gates match the product claim**

Run:
- `python -m pytest project/tests/integration/test_default_path_smoke.py -q`
- inspect `.github/workflows/test.yml` to confirm the hosted smoke job executes the new check

Expected: the shipped path and the claimed learning loop are both under repeatable verification.

**Definition of done**
- The release checklist and CI cover the user-visible workflow that the docs advertise.

---

### Task 6: Clean Remaining Release Blockers and Engineering Hygiene

**Files:**
- Modify: `project/frontend/blueprints/api_pipeline.py`
- Re-run: all test and lint targets already used by CI

- [ ] **Step 1: Fix the current `ruff` failure**

Repair the import ordering issue in `project/frontend/blueprints/api_pipeline.py` so the repo is green under its own lint gate.

- [ ] **Step 2: Run the focused verification stack**

Run:
- `python -m pytest project/tests/unit/test_api_pipeline.py -q`
- `python -m pytest project/tests/unit/test_main.py -q`
- `python -m pytest project/tests/unit/test_truth_export_3d.py -q`
- `python -m pytest project/tests/unit/test_whole_brain_3d.py -q`
- `python -m pytest project/tests/unit/test_api_liquify_3d.py -q`
- `python -m pytest project/tests/unit/test_class_prior.py -q`
- `python -m pytest project/tests/unit/test_mask_endpoints.py -q`
- `python -m pytest project/tests/unit/test_frontend_regressions.py -q`
- `python -m pytest project/tests/integration/test_default_path_smoke.py -q`

Expected: all changed-path tests pass.

- [ ] **Step 3: Run the repo-level gates**

Run:
- `python -m pytest project/tests/unit -q`
- `python -m ruff check project/scripts project/frontend/blueprints project/frontend/server_context.py project/frontend/app_metadata.py project/frontend/update_checker.py`
- `python -m ruff format --check project/scripts project/frontend/blueprints project/frontend/server_context.py project/frontend/app_metadata.py project/frontend/update_checker.py`

Expected: unit suite passes and both `ruff` commands are clean.

**Definition of done**
- The remediation branch is green under the same unit/lint gates the repository expects in CI.

---

## Execution Order

1. Task 1
2. Task 2
3. Task 3
4. Task 4
5. Task 5
6. Task 6

## Coverage Check

- Finding 1 (`server_context.py` learned params written where jobs do not read them): covered by Tasks 1 and 2.
- Finding 2 (`main.py` default whole-brain path bypasses calibration learning): covered by Task 2.
- Finding 3 (`app.js` class-prior learning is manual-only): covered by Task 3.
- Productization drift (docs vs reality): covered by Task 4.
- Release-gate gap (CI/manual acceptance do not prove the feature): covered by Task 5.
- Mutable runtime state under source tree: covered by Task 1.
- Current lint failure and iterator-safety bug: covered by Task 6 and Task 3.

## Out of Scope

- Replacing ANTs with a learned registration model
- Designing a full 3D viewer beyond the current slice-based liquify workflow
- Broad refactors unrelated to the closed-loop/productization findings

Plan complete and saved to `docs/superpowers/plans/2026-04-20-closed-loop-productization-remediation.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
