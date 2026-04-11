# Brainfast Sample Stability Rollout Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the default Brainfast pipeline reproducible and stable enough to process new samples beyond the current working set with controlled failure modes, actionable QC, and a staged rollout path.

**Architecture:** Stabilize the repo in three layers: first lock the runtime and CI environment, then harden the default `miki_3d + cpsam` execution path, then roll out across increasingly heterogeneous samples using a canary set before full-batch execution. Do not add broad new features until the default path is green and reproducible.

**Tech Stack:** Python 3.10/3.11, NumPy/SciPy/scikit-image, ANTsPyX, Cellpose v4 / Cellpose-SAM (`cpsam`), pytest, ruff, GitHub Actions.

**Current user-visible blockers:**
- Whole-brain registration is currently the highest-risk failure point; overlays are often poor enough that downstream region mapping cannot be trusted.
- The current UI exposes 2D manual landmark / liquify / calibration tools, but the 3D whole-brain path is still largely automatic and separate. The user cannot yet experience one stable end-to-end manual workflow.
- `Save Calibration + Learn` currently tunes atlas overlay / warp behavior, not Cellpose-SAM detection quality. There is no detector-specific manual review / relabel / retrain loop in the UI.
- Repeated "natural" end-to-end failures are preventing workflow validation. The plan must prioritize a single-sample interactive happy path before broad sample rollout.

---

### Task 0: Make the current workflow gaps explicit before further rollout (Day 0-1)

**Files:**
- Create: `docs/superpowers/plans/interactive-workflow-gap-audit.md`
- Create: `docs/superpowers/plans/registration-failure-taxonomy.md`
- Modify: `README.md`
- Modify: `project/frontend/README.md`

- [ ] **Step 1: Write down the actual product-state gap between 2D and 3D workflows**

Required conclusions to capture:
- 2D UI manual correction exists
- 3D whole-brain final truth does not yet consume that manual correction path end-to-end
- Cellpose-SAM is integrated as a backend detector, but has no detector-specific manual QA loop in the product

- [ ] **Step 2: Build a registration failure taxonomy from one good and one bad sample**

Use categories:
- wrong AP slice choice
- hemisphere / left-right inversion
- scale / fit mismatch
- tissue crop / padding mismatch
- nonlinear warp distortion
- output looks plausible in one slice but inconsistent across stack

- [ ] **Step 3: Add a hard trust policy to the docs**

Required policy:
- Do not trust region-level counts when registration overlays are visibly poor
- Do not use Cellpose quality as a scapegoat for atlas-mapping failures before registration is verified
- Do not expand sample coverage until at least one sample has completed a user-visible manual workflow

- [ ] **Step 4: Stop here unless the team agrees on the current missing product pieces**

Missing pieces that must be named explicitly:
- stable single-sample interactive workflow
- registration-first QC gate
- Cellpose-SAM review gap

---

### Task 1: Lock the numerical runtime and installation path (Day 1)

**Files:**
- Modify: `pyproject.toml`
- Modify: `project/requirements-min.txt`
- Modify: `.github/workflows/test.yml`
- Modify: `README.md`

- [ ] **Step 1: Align `pyproject.toml` with the known-good numerical stack**

Target state:

```toml
dependencies = [
    "flask>=3.0",
    "numpy>=1.26,<2",
    "packaging>=24.0",
    "pandas>=2.2",
    "pillow>=10.0",
    "scipy>=1.12,<1.13",
    "scikit-image>=0.22,<0.23",
    "tifffile>=2024.2",
    "nibabel>=5.2",
    "tqdm>=4.66",
    "matplotlib>=3.8",
]
```

- [ ] **Step 2: Keep `project/requirements-min.txt` as the fully pinned smoke-test matrix**

Run:

```powershell
python -m pip install --disable-pip-version-check -r project/requirements-min.txt
python - <<'PY'
import importlib.metadata as md
for pkg in ["numpy", "scipy", "scikit-image"]:
    print(pkg, md.version(pkg))
PY
```

Expected: `numpy 1.26.4`, `scipy 1.12.0`, `scikit-image 0.22.0`

- [ ] **Step 3: Make CI install the package the same way developers do**

Target state:

```yaml
- name: Install package + test tools
  run: pip install -e ".[dev]"

- name: Install optional runtime extras for default path checks
  run: pip install -e ".[wholebrain,advanced]"
```

- [ ] **Step 4: Update `README.md` to document three supported install modes**

Required modes:
- Minimal 2D only
- Default recommended runtime for `miki_3d + cpsam`
- Full developer stack

- [ ] **Step 5: Stop here unless a clean environment can import the numerical stack**

Run:

```powershell
python -m pytest project/tests/unit/test_detect.py -q
```

Expected: test collection succeeds; no `numpy.dtype size changed` or `_ARRAY_API` import errors

---

### Task 2: Turn configuration mismatch into early failure (Day 1-2)

**Files:**
- Modify: `project/scripts/check_env.py`
- Modify: `project/configs/run_config.template.json`
- Modify: `project/scripts/main.py`

- [ ] **Step 1: Make `check_env.py` validate optional modules against the active config**

Target behavior:

```python
needs_ants = (
    cfg.get("registration", {}).get("scope") == "whole"
    and cfg.get("registration", {}).get("whole_brain_backend") == "miki_3d"
)
needs_cellpose = any(
    str(cfg.get("detection", {}).get(key, "")).lower() in {"cpsam", "sam"}
    or str(cfg.get("detection", {}).get(key, "")).lower().startswith("cellpose")
    for key in ("primary_model", "secondary_model")
)
```

If `needs_ants` or `needs_cellpose` is true, missing modules must be reported as `FAIL`, not `WARN`.

- [ ] **Step 2: Narrow the broad import fallback in `main.py`**

Target behavior:

```python
try:
    ...
except ImportError as exc:
    raise
```

Do not mask a real `scipy` / `skimage` import failure behind a second `ModuleNotFoundError` from the fallback branch.

- [ ] **Step 3: Make the template config explicit about required extras**

Add a short adjacent comment block or README note mapping:
- `whole_brain_backend = miki_3d` -> needs `.[wholebrain]`
- `primary_model = cpsam` -> needs `.[advanced]`

- [ ] **Step 4: Verify the config gate**

Run:

```powershell
python project/scripts/check_env.py --config project/configs/run_config.template.json
```

Expected: on a machine missing `ants` or `cellpose`, this exits non-zero with direct dependency failures tied to the active config

---

### Task 3: Restore static quality gates (Day 2)

**Files:**
- Modify: `project/scripts/laplacian_refine_3d.py`
- Modify: `project/scripts/registration_3d_volume.py`

- [ ] **Step 1: Remove current `ruff` blockers without changing behavior**

Required cleanups:
- Remove unused `gaussian_filter`
- Expand semicolon-separated statements to one statement per line
- Remove or use the unused `affine` / `ann_affine` locals

- [ ] **Step 2: Re-run lint until the tracked quality gate is clean**

Run:

```powershell
python -m ruff check project/scripts/ project/frontend/blueprints/ project/frontend/server_context.py project/frontend/app_metadata.py project/frontend/update_checker.py
```

Expected: `All checks passed!`

- [ ] **Step 3: Keep formatting in sync**

Run:

```powershell
python -m ruff format --check project/scripts/ project/frontend/blueprints/ project/frontend/server_context.py project/frontend/app_metadata.py project/frontend/update_checker.py
```

Expected: no format drift

---

### Task 4: Make the default `cpsam` path testable and trustworthy (Day 3)

**Files:**
- Modify: `project/scripts/detect.py`
- Modify: `project/tests/unit/test_detect.py`
- Modify: `README.md`

- [ ] **Step 1: Add direct tests for Cellpose-SAM model selection**

Add tests that cover:
- `_resolve_model_type("cpsam") -> "cpsam"`
- `_resolve_model_type("sam") -> "cpsam"`
- `CellposeModel(pretrained_model=...)` branch when `models.CellposeModel` exists
- v4 three-value `eval()` return handling

Test sketch:

```python
def test_cellpose_sam_model_path_uses_cellposemodel(monkeypatch, tiny_slice):
    fake_models = type("M", (), {"CellposeModel": object})
    ...
```

- [ ] **Step 2: Add a no-silent-fallback regression for explicit Cellpose usage**

Run:

```powershell
python -m pytest project/tests/unit/test_detect.py -v
```

Expected: direct coverage for `cpsam`, load failure, inference failure, and `auto_switch_on_distortion` behavior

- [ ] **Step 3: Document what “Cellpose-SAM integrated” actually means**

Clarify in `README.md`:
- Brainfast uses the `cellpose` package's v4 `CellposeModel`
- There is no separate `segment_anything` / `SAM2` runtime in this repo
- Supported default model name is `cpsam`
- `Save Calibration + Learn` does not currently retrain or fine-tune Cellpose-SAM

- [ ] **Step 4: Add a detector trust rule to the operator docs**

Required rule:
- evaluate raw detector plausibility and region-mapped plausibility separately
- if registration is poor, detector output may still be locally reasonable but brain-region counts are not trustworthy
- do not call the detector "validated" until it has been reviewed on a registration-passing sample

---

### Task 4A: Create one complete manual workflow the user can actually experience (Day 3-4)

**Files:**
- Modify: `project/frontend/index.html`
- Modify: `project/frontend/app.js`
- Modify: `project/frontend/README.md`
- Modify: `README.md`
- Modify: `project/frontend/blueprints/api_pipeline.py`
- Modify: `project/frontend/blueprints/api_overlay.py`

- [ ] **Step 1: Define the minimum viable interactive workflow**

Required happy path:
1. load one sample
2. preview atlas
3. run auto registration
4. perform manual landmark or liquify correction
5. save calibrated result
6. run detector preview on the corrected sample
7. export one user-visible artifact bundle

- [ ] **Step 2: Make the workflow explicit in the UI and docs**

The user must be able to tell:
- which path is 2D interactive validation
- which path is 3D automatic whole-brain processing
- whether the current action affects registration only, or also affects downstream quantification

- [ ] **Step 3: Add a "single-sample validation" gate before batch rollout**

Success criteria for one sample:
- registration overlay is visibly acceptable
- manual correction tools work
- calibration save works
- detector preview can be run on the corrected sample
- final exported artifacts are understandable without reading the code

- [ ] **Step 4: Block batch rollout until this workflow is demo-able**

Rule:
- if the user still cannot complete one full manual workflow from the UI, do not spend more time on broad natural testing

---

### Task 5: Build a canary sample lane before full-batch rollout (Day 4-5)

**Files:**
- Create: `docs/superpowers/plans/sample-canary-manifest.md`
- Modify: `README.md`
- Modify: `project/configs/run_config.template.json` if sample-specific knobs must be surfaced

- [ ] **Step 1: Define three sample buckets**

Bucket design:
- `Canary A`: one sample that already runs end-to-end today
- `Canary B`: one weaker/noisier sample
- `Canary C`: one shape/orientation outlier

- [ ] **Step 2: Define pass/fail gates for each canary run**

Each run must produce:
- No hard crash
- Non-empty QC outputs
- Registered truth labels for every slice
- Cell counts that are non-zero when expected
- No obvious left/right or AP inversion in overlay spot checks

- [ ] **Step 3: Record a one-page operator checklist**

Checklist items:
- install command used
- config used
- sample path
- output path
- number of slices
- whether `cpsam` stayed primary or fell back
- whether manual review flagged hemisphere/AP mistakes

- [ ] **Step 4: Do not expand sample scope until all three canaries pass twice**

Run each canary twice in a clean shell.
Expected: same stage completion pattern, same artifact set, and comparable counts/QC

---

### Task 6: Roll out by percentage, not all at once (Day 6-8)

**Files:**
- Modify: `README.md`
- Create: `docs/superpowers/plans/sample-rollout-log.md`

- [ ] **Step 1: Run 20% of the remaining samples**

Rules:
- Mix easy and hard samples
- Stop immediately on the first new failure class
- Tag each failure as one of: environment, registration, truth export, detection, mapping, or QC-only

- [ ] **Step 2: Fix one failure class at a time**

Required loop:
1. Reproduce on the smallest failing sample
2. Add or update a unit/integration test
3. Implement the minimal fix
4. Re-run canaries
5. Re-run the blocked subset

- [ ] **Step 3: Expand to 60% only after the 20% cohort is clean**

Promotion gate:
- zero environment/setup failures
- zero import-time crashes
- zero missing-artifact failures
- only understood sample-specific failures, if any

- [ ] **Step 4: Run the full sample set only after 60% passes**

Full-batch success criteria:
- every sample completes or fails with an explicit, typed reason
- every completion has QC outputs
- rerunning the same sample does not produce different stage outcomes

---

### Task 7: Declare “stable for other samples” only after hard gates are met (Day 8+)

**Files:**
- Modify: `README.md`
- Create: `docs/superpowers/plans/stability-exit-criteria.md`

- [ ] **Step 1: Require these repo-level gates**

Mandatory gates:
- `ruff check` green
- unit tests green in the pinned runtime
- `check_env.py` correctly fails on config/dependency mismatch
- default `miki_3d + cpsam` path documented and reproducible

- [ ] **Step 2: Require these sample-level gates**

Mandatory gates:
- 3/3 canaries pass twice
- 20% cohort passes
- 60% cohort passes
- full batch completes with only known, triaged exceptions

- [ ] **Step 3: Freeze a release candidate environment**

Run:

```powershell
python -m pip freeze > outputs/release_candidate_requirements.txt
```

Expected: one exact environment snapshot tied to the batch run you trust

---

## Recommended Schedule

- **Day 0-1:** Task 0
- **Day 1:** Task 1
- **Day 2:** Task 2 + Task 3
- **Day 3:** Task 4
- **Day 3-4:** Task 4A
- **Day 4-5:** Task 5
- **Day 6-8:** Task 6
- **Day 8+:** Task 7 and release-candidate freeze

## What “done” means

This work is done only when a clean machine can install the documented stack, `check_env.py` rejects invalid default-path setups, registration quality has explicit trust gates, the user can complete one full interactive manual workflow on a single sample, the default `miki_3d + cpsam` path is covered by tests, the canary set is repeatably green, and the remaining samples have been rolled out in cohorts without introducing new untriaged failure classes.
