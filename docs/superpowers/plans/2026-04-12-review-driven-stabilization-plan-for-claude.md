# Brainfast Review-Driven Stabilization Plan For Claude

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Convert the current review findings into an execution-first stabilization plan that restores repo-level trust, proves the default runtime path, and unblocks a real end-to-end user workflow.

**Architecture:** Work from repo-level correctness downward: first fix environment truthfulness and CI coverage, then tighten the default `miki_3d + cpsam` runtime path, then gather registration evidence on real samples before changing defaults, and only after that unblock the interactive single-sample workflow and wider sample rollout.

**Tech Stack:** Python 3.10/3.11, NumPy/SciPy/scikit-image, ANTsPyX, Cellpose v4 / Cellpose-SAM (`cpsam`), pytest, ruff, GitHub Actions, Brainfast frontend.

---

## Ground Rules

- Do not reopen already closed findings just to create activity.
- `registration_3d_ants.py` transform persistence is already fixed; do not spend time re-fixing it.
- Do not argue from a single sample when the question is a template default.
- Do not call the repo "stable" until environment checks, CI, and the user-visible workflow all agree.

## Repo-Proven Execution Hooks

Use the scripts and fixtures that already exist before inventing new helpers:

- `project/scripts/check_env.py`
- `project/scripts/ml_flip_ab_test.py`
- `project/scripts/validate_canary.py`
- `project/scripts/run_batch_manifest.py`
- `project/tests/unit/test_detect.py`
- `project/tests/integration/test_regression_suite.py::SyntheticPipelineSmokeTests::test_synthetic_slice_end_to_end`
- `project/configs/run_config_35.json`
- `project/configs/run_config_35_canary_b.json`
- `project/configs/run_config_35_canary_c.json`

Current limitations that must be treated as work items, not ignored:

- `ml_flip_ab_test.py` is still hard-wired to sample 35 config/data; it is not yet a general matrix runner.
- `validate_canary.py` has real artifact gates for outputs and metrics, but its orientation gate is still informational only.
- The integration smoke test is a synthetic 2D slice baseline; it does not validate the default whole-brain runtime path.
- Frontend already has `/api/overlay/preview`, `/api/overlay/liquify-drag`, and `/api/overlay/calibration/finalize`; there is no detector-preview route yet.

---

### Task 1: Make environment validation truthful

**Files:**
- Modify: `project/scripts/check_env.py`
- Modify: `README.md`

- [ ] **Step 1: Replace `find_spec()`-only checks with real import smoke tests for critical runtime modules**

Critical modules:
- `numpy`
- `scipy`
- `skimage`
- `cellpose` when active config requires it
- `ants` when active config requires it

Expected behavior:
- missing module -> `FAIL`
- import-time ABI/version crash -> `FAIL`
- only truly optional modules remain `WARN`

- [ ] **Step 2: Add minimal compatibility assertions for the numerical stack**

Required checks:
- reject `numpy>=2` when installed `scipy` / `scikit-image` cannot import
- print exact package versions in the environment report

- [ ] **Step 3: Verify the fix against the current broken environment**

Run:

```powershell
python project/scripts/check_env.py --config project/configs/run_config.template.json
python -m pytest project/tests/unit -q
```

Expected:
- non-zero exit
- explicit failure on broken `scipy` / `skimage` import path
- environment report and pytest import behavior must no longer disagree

- [ ] **Step 4: Update docs so users understand what `check_env.py` actually guarantees**

Add one short section:
- "module found" is not enough
- Brainfast now verifies importability of the active runtime path

---

### Task 2: Make CI prove the default shipped path, not just the dev path

**Files:**
- Modify: `.github/workflows/test.yml`
- Modify: `README.md`

- [ ] **Step 1: Keep the current fast dev-path unit job**

Do not remove the lightweight `.[dev]` lane.

- [ ] **Step 2: Add a second runtime-validation lane for the default template config**

Required install:

```yaml
pip install -e ".[wholebrain,advanced,dev]"
```

Required commands:

```yaml
python project/scripts/check_env.py --config project/configs/run_config.template.json
python -m pytest project/tests/unit/test_detect.py -q
python -m pytest project/tests/integration/test_regression_suite.py -k synthetic_slice_end_to_end -q
```

Required interpretation:

- the synthetic integration test is only a floor, not proof of the shipped whole-brain path
- if there is still no smallest default whole-brain smoke test, create one in `project/tests/integration/` instead of hand-waving the gap away

- [ ] **Step 2a: Add or wire one default-path smoke test if CI still lacks it**

Target:
- one test that exercises `scope=whole`, `whole_brain_backend=miki_3d`, `primary_model=cpsam`
- keep it small enough for CI by using sparse slices or mocked heavy transforms where necessary
- name it so failures obviously read as "default runtime path is broken"

- [ ] **Step 3: Make CI failure messages map clearly to the default product path**

Examples:
- environment invalid for default config
- `cpsam` runtime unavailable
- ANTs runtime unavailable

- [ ] **Step 4: Verify the workflow still keeps lint and unit feedback fast**

Expected:
- one fast lane for ordinary code review
- one default-path lane for product/runtime truth

---

### Task 3: Finish the Cellpose-SAM v4 cleanup instead of stopping at "it basically works"

**Files:**
- Modify: `project/scripts/detect.py`
- Modify: `project/tests/unit/test_detect.py`
- Modify: `README.md`

- [ ] **Step 1: Narrow legacy `channels` handling to the actual legacy branch**

Current problem:
- mixed-version layouts can still send `channels` even when `CellposeModel` exists

Required target behavior:
- `channels` only added when the code is truly using legacy `models.Cellpose`

- [ ] **Step 2: Add explicit tests for the default `cpsam` runtime path**

Must cover:
- `_resolve_model_type("cpsam")`
- `CellposeModel(pretrained_model=...)`
- v4 3-value `eval()` handling
- no legacy `channels` on the v4 path

Add one test that fails specifically if a mixed-version layout still injects `channels` into the v4 kwargs.

- [ ] **Step 3: Keep the no-silent-fallback guarantee**

Retain:
- explicit Cellpose failure must raise when `auto_switch_on_distortion` is false
- fallback is only allowed when config explicitly allows it

- [ ] **Step 4: Clarify the product truth in docs**

Document:
- Brainfast integrates Cellpose v4 / Cellpose-SAM through the `cellpose` package
- there is no separate `segment_anything` / `SAM2` runtime in this repo
- `Save Calibration + Learn` does not train Cellpose-SAM

---

### Task 4: Replace `ml_flip` arguments with evidence

**Files:**
- Modify: `project/configs/run_config.template.json` only if evidence justifies it
- Modify: `docs/superpowers/plans/ml-flip-ab-evidence.md`
- Create: `docs/superpowers/plans/ml-flip-matrix-summary.md`
- Modify: `README.md`

- [ ] **Step 1: Treat the existing `ml-flip-ab-evidence.md` as partial evidence, not final proof**

Current status:
- only one sample
- sample uses `atlas_hemisphere: "right_flipped"`
- that sample is not enough to decide the template default

- [ ] **Step 2: Build a proper evidence matrix**

Minimum coverage:
- one `right_flipped` hemisphere sample
- one `left` hemisphere sample
- one `whole` or orientation-ambiguous sample

Use these as the starting pool instead of inventing abstract categories:

- `right_flipped`: `project/configs/run_config_35.json` with `project/data/35_C0_test` or `project/data/35_C0_full`
- `left`: stage one real left-labeled sample from `Sample/PVe3/*left*`, `Sample/PVe4/*left*`, or `Sample/SSTe4/*left*`
- `whole` / ambiguous: stage one real sample from `Sample/ChATe27/*Bothlaser*` or `Sample/ChATe27/*both*`

Required implementation detail:

- first refactor `project/scripts/ml_flip_ab_test.py` so it accepts `--config` and `--input-dir`, instead of hard-coding sample 35
- keep sample 35 as the existing right-flipped baseline, not the only evidence source

For each sample compare:
- `ml_flip=false`
- `ml_flip=true`

Capture:
- QC metrics
- overlay screenshots
- left/right region plausibility
- whether mapped results obviously invert anatomy

Minimum command shape after refactor:

```powershell
python project/scripts/ml_flip_ab_test.py --config project/configs/run_config_35.json --input-dir project/data/35_C0_test
python project/scripts/ml_flip_ab_test.py --config <left-sample-config> --input-dir <left-sample-dir>
python project/scripts/ml_flip_ab_test.py --config <whole-or-ambiguous-config> --input-dir <whole-sample-dir>
```

- [ ] **Step 3: Keep the default unchanged until the matrix is complete**

Rule:
- do not defend or change the template default from a single-sample result

- [ ] **Step 4: Write the final decision as an evidence-backed recommendation**

Possible outcomes:
- keep `ml_flip=false`
- revert to `ml_flip=true`
- require per-sample override by sample class

But the document must state why.

---

### Task 5: Turn registration quality into a hard gate instead of a subjective complaint

**Files:**
- Modify: `project/configs/run_config.template.json`
- Modify: `project/scripts/main.py`
- Create: `docs/superpowers/plans/registration-quality-gate.md`

- [ ] **Step 1: Define what counts as "registration acceptable"**

Required checks:
- numeric score threshold
- overlay visual spot-check
- no obvious hemisphere inversion
- no obvious AP mismatch

Reuse `project/scripts/validate_canary.py` as the starting gate implementation instead of creating a parallel scoring system.

- [ ] **Step 2: Audit the current `fail_score_threshold` against real bad-looking overlays**

Question to answer:
- does the current threshold allow overlays that a human would reject?

Concrete evidence sources:

- `project/configs/run_config_35_canary_b.json`
- `project/configs/run_config_35_canary_c.json`
- `python project/scripts/validate_canary.py --output-dir <run-output> --canary B`
- `python project/scripts/validate_canary.py --output-dir <run-output> --canary C`

- [ ] **Step 3: Tighten behavior if poor overlays still pass**

Options:
- raise threshold
- use score + visual proxy combination
- block mapping/export when registration QC is poor
- convert "orientation" from a reminder-only gate into either a documented manual signoff step or a stronger automated proxy

- [ ] **Step 4: Write the operator rule**

Rule:
- poor registration overlay means region-level counts are not trustworthy
- detector quality and mapping quality must be judged separately

---

### Task 6: Build the first real user-complete workflow

**Files:**
- Modify: `project/frontend/index.html`
- Modify: `project/frontend/app.js`
- Modify: `project/frontend/README.md`
- Modify: `README.md`
- Modify: `project/frontend/blueprints/api_overlay.py`
- Modify: `project/frontend/blueprints/api_pipeline.py`

- [ ] **Step 1: Define the minimal happy path the user must be able to complete**

Required path:
1. load one sample
2. preview atlas
3. run auto registration
4. apply manual landmark or liquify correction
5. save calibration
6. run detector preview
7. export one understandable artifact bundle

Treat the current frontend endpoints as the baseline:

- already present: `/api/overlay/preview`
- already present: `/api/overlay/liquify-drag`
- already present: `/api/overlay/calibration/finalize`
- missing today: detector preview route and detector preview UI state

- [ ] **Step 2: Make the UI tell the truth about 2D vs 3D**

The user must be able to tell:
- which tools affect only 2D preview
- which tools affect whole-brain final outputs
- whether a saved calibration will influence Cellpose or only overlay parameters

Explicitly state in the UI text and docs:

- `Save Calibration + Learn` updates atlas/overlay calibration data
- it does not train Cellpose-SAM
- a successful 2D calibration does not by itself prove 3D whole-brain truth quality

- [ ] **Step 2a: Define the missing detector-preview contract**

Add one route and one UI view that let a user inspect detector output before full batch rollout.

Minimum contract:
- input: current sample or current preview slice
- output: overlayable detections plus a machine-readable table
- failure mode: if Cellpose runtime is unavailable, surface that explicitly instead of silently showing zero cells

- [ ] **Step 2b: Define the artifact bundle the user must leave with**

Minimum bundle for one interactive sample:
- calibrated label TIFF
- calibrated overlay PNG
- calibration manifest / learn status
- detector preview artifact (`.png` and `.csv` or equivalent)
- final export path shown in UI

- [ ] **Step 3: Block broad natural testing until this path works once, end-to-end**

Rule:
- no more "just run more samples" until one single sample can complete the full interactive path without manual code intervention

- [ ] **Step 4: Write down exactly what still remains outside this first workflow**

Examples:
- 3D truth integration
- detector-specific QA loop
- sample-batch automation

---

### Task 7: Only after Tasks 1-6, resume sample rollout

**Files:**
- Modify: `docs/superpowers/plans/2026-04-10-sample-stability-rollout-plan.md`
- Create: `docs/superpowers/plans/review-driven-rollout-handoff.md`

- [ ] **Step 1: Rebase the old rollout plan on the new evidence**

Carry forward only after:
- environment truthfulness fixed
- default-path CI added
- Cellpose-SAM cleanup done
- `ml_flip` matrix completed or explicitly deferred
- first interactive workflow proven

- [ ] **Step 2: Rebuild the canary set using orientation-aware coverage**

Canaries must include:
- one easy sample
- one noisy or weak-signal sample
- one orientation-sensitive sample

Use `project/scripts/run_batch_manifest.py` for cohort rollout instead of ad-hoc shell history.

Required handoff artifact:
- one reviewed manifest derived from `project/configs/batch_manifest.template.csv`
- one note per canary explaining why it belongs in easy / noisy / orientation-sensitive

- [ ] **Step 3: Resume rollout in cohorts**

Only then:
- 20%
- 60%
- 100%

Stop on first new failure class.

---

## Acceptance Criteria

This plan is complete only when all of the following are true:

- `check_env.py` fails on a broken numerical runtime instead of reporting false OK
- CI has a lane that validates the default template config, not only `.[dev]`
- the `cpsam` v4 path is cleanly separated from legacy `Cellpose` kwargs
- `ml_flip` default discussion is backed by a multi-sample evidence matrix
- registration quality is an explicit gate before region mapping trust
- the user can complete one full interactive single-sample workflow

## Immediate Priority Order

1. Task 1
2. Task 2
3. Task 3
4. Task 5
5. Task 4
6. Task 6
7. Task 7
