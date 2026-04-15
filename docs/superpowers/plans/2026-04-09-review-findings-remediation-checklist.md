# Review Findings Remediation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复本轮 code review 中确认的 5 个高优先级问题，并把单元测试与基础 lint 恢复到可作为合并门禁的状态。

**Architecture:** 先修数据正确性与失败可见性，再修 3D 编排契约和依赖声明，最后统一跑回归验证。改动应优先保持现有模块边界不变，只修正错误契约、异常处理和安装入口，避免在同一轮引入额外重构。

**Tech Stack:** Python 3.10+/3.11、Flask、NumPy、pandas、nibabel、tifffile、Cellpose、ANTs/`ants` runtime、pytest、ruff

---

## File Map

**Core code**
- Modify: `project/scripts/truth_export_3d.py`
- Modify: `project/scripts/detect.py`
- Modify: `project/scripts/main.py`
- Modify: `project/scripts/whole_brain_3d.py`
- Modify: `project/scripts/registration_3d_ants.py`
- Modify: `project/scripts/check_env.py`
- Modify: `pyproject.toml`
- Modify: `project/requirements-min.txt`

**Tests**
- Modify: `project/tests/unit/test_truth_export_3d.py`
- Modify: `project/tests/unit/test_whole_brain_3d.py`
- Modify: `project/tests/unit/test_registration_3d_ants.py`
- Create or modify: `project/tests/unit/test_detect.py` or an equivalent existing unit test file covering Cellpose failure paths

**Docs**
- Modify: `README.md`
- Modify: `project/README.md`
- Modify: `REPRODUCE.md`

---

### Task 1: Make Truth Export Use One Canonical Registered Label

**Files:**
- Modify: `project/scripts/truth_export_3d.py`
- Modify: `project/tests/unit/test_truth_export_3d.py`

- [ ] Update `export_registered_truth_slices()` so the `render_overlay()` call passes `warped_label_out=label_path`.
- [ ] Keep `prewarped_label=True`, because the 3D truth-export path should still bypass the 2D tissue-guided warp while allowing final resize/tissue masking to be written back to disk.
- [ ] Ensure the TIFF used later by `map_cells_with_registered_label_slice()` is the same raster that `render_overlay()` finalized for the overlay.
- [ ] Extend the unit test to assert that the exported `registered_label.tif` matches the post-render output contract, not just the pre-render slice.
- [ ] Run: `python -m pytest project/tests/unit/test_truth_export_3d.py -q`
- [ ] Expected: all tests in `test_truth_export_3d.py` pass.

**Definition of done**
- `slice_*_registered_label.tif` is the canonical, final atlas label for both overlay display and region mapping.

---

### Task 2: Fail Fast When Cellpose Is Selected but Unavailable or Broken

**Files:**
- Modify: `project/scripts/detect.py`
- Modify: `project/scripts/main.py`
- Modify: `project/scripts/exceptions.py` only if a more specific detection exception is needed
- Create or modify: `project/tests/unit/test_detect.py`

- [ ] Replace broad silent fallthrough in `detect_cells_cellpose()` with explicit error reporting when Cellpose model load or inference fails.
- [ ] Preserve the existing non-Cellpose fallback path only for cases where config explicitly selects fallback detection, not when `primary_model` or `secondary_model` starts with `cellpose`.
- [ ] Make `detect_cells()` distinguish between:
  - configured Cellpose but runtime failure
  - configured fallback detector
  - true no-cell result from a successful detector run
- [ ] In `run_real_input()`, surface Cellpose runtime failures as pipeline errors instead of interpreting them as empty biological detections.
- [ ] Add unit coverage for:
  - missing `cellpose` import
  - model init failure
  - inference failure
  - successful fallback path when fallback is intentionally configured
- [ ] Run: `python -m pytest project/tests/unit/test_detect.py -q`
- [ ] Run: `python -m pytest project/tests/unit/test_main.py -q`
- [ ] Expected: detection failure paths are covered and no selected-Cellpose failure is silently converted into an empty DataFrame.

**Definition of done**
- A broken or missing Cellpose runtime cannot produce a false “0 cells” success result when Cellpose was the configured detector.

---

### Task 3: Align Default Runtime Path with Declared Install Dependencies

**Files:**
- Modify: `pyproject.toml`
- Modify: `project/requirements-min.txt`
- Modify: `project/scripts/check_env.py`
- Modify: `README.md`
- Modify: `project/README.md`
- Modify: `REPRODUCE.md`
- Optionally modify: `project/configs/run_config.template.json` if the chosen install story requires changing the default execution path

- [ ] Decide one install contract and apply it consistently:
  - option A: declare the package that provides `import ants` in project install metadata
  - option B: keep ANTs optional, but stop routing a fresh install into the whole-brain `miki_3d` default path
- [ ] If keeping whole-brain as the default, add a dedicated extra such as `wholebrain` that installs the `ants` runtime used by `project/scripts/whole_brain_3d.py`.
- [ ] Keep `check_env.py` consistent with the declared dependency model:
  - required if default path needs it
  - optional if default path no longer requires it
- [ ] Update the README and reproduction docs so the installation instructions match the actual runtime contract.
- [ ] Verify that a fresh engineer can infer which command to run for:
  - minimal 2D usage
  - whole-brain 3D usage
  - Cellpose-enabled usage
- [ ] Run: `python project/scripts/check_env.py --config project/configs/run_config.template.json`
- [ ] Expected: the environment checker’s required/optional dependency status matches the documented install path.

**Definition of done**
- The repository’s default config, dependency metadata, environment checker, and setup docs all describe the same runnable path.

---

### Task 4: Respect the Actual Volume Path Returned by the Volume Builder

**Files:**
- Modify: `project/scripts/whole_brain_3d.py`
- Modify: `project/tests/unit/test_whole_brain_3d.py`

- [ ] Replace direct reuse of the local `volume_path` variable after `build_volume_from_tiffs()` with `Path(volume_meta["volume_path"])`.
- [ ] Apply the ML-flip step to the returned volume path, not the originally requested path literal.
- [ ] Keep all later consumers (`run_ants_registration`, truth export, inverse warp) pointed at the possibly flipped `volume_meta["volume_path"]`.
- [ ] Preserve the existing behavior where the flipped volume becomes the active moving image for registration.
- [ ] Re-run the stage-sequence test to confirm mocked builders that return alternate paths no longer break the orchestrator.
- [ ] Run: `python -m pytest project/tests/unit/test_whole_brain_3d.py -q`
- [ ] Expected: `test_run_whole_brain_3d_emits_expected_stage_sequence` passes.

**Definition of done**
- `run_whole_brain_3d()` trusts helper return values instead of assuming helper internals.

---

### Task 5: Make ANTs Transform Persistence Best-Effort Instead of Fatal

**Files:**
- Modify: `project/scripts/registration_3d_ants.py`
- Modify: `project/tests/unit/test_registration_3d_ants.py`

- [ ] Wrap forward/inverse transform persistence in existence checks and non-fatal error handling.
- [ ] If a transform file exists, copy it into `out_dir` as today.
- [ ] If a transform file does not exist, keep the original transform string in the returned metadata rather than crashing after registration success.
- [ ] Ensure metrics CSV and summary TXT are still written whenever the registered volume was produced successfully.
- [ ] Add a unit test for the case where `ants.registration()` returns transform paths that cannot be copied.
- [ ] Run: `python -m pytest project/tests/unit/test_registration_3d_ants.py -q`
- [ ] Expected: the function succeeds and returns transform metadata even when transform file copy is not possible.

**Definition of done**
- A successful ANTs registration does not get reclassified as failed solely because transform persistence was partial.

---

### Task 6: Rebuild the Quality Gate

**Files:**
- Verify only; no required code file

- [ ] Run the focused unit tests from Tasks 1-5 individually until all pass.
- [ ] Run: `python -m pytest project/tests/unit -q`
- [ ] Expected: current unit suite passes with zero failures.
- [ ] Run: `python -m ruff check project/scripts/ project/frontend/blueprints/ project/frontend/server_context.py project/frontend/app_metadata.py project/frontend/update_checker.py`
- [ ] Expected: zero lint errors in the paths enforced by `.github/workflows/test.yml`.
- [ ] If lint still fails on pre-existing unrelated issues, split them into:
  - blocking issues in CI-covered paths that must be fixed now
  - non-blocking full-repo cleanup work for a separate follow-up PR

**Definition of done**
- The exact unit-test and lint commands used by CI are green locally.

---

## Notes

- `project/tests/unit/test_registration_3d_volume.py` currently expects `0.025` zooms while the bundled `annotation_25.nii.gz` header reports `25.0` with unknown units. Treat that as a separate contract clarification item unless this remediation branch explicitly standardizes volume-unit semantics.
- `Cellpose-SAM` is not currently integrated in the active codebase. Do not bundle SAM-related changes into the five fixes above unless a separate requirement is added.

## Suggested Execution Order

1. Task 1
2. Task 2
3. Task 4
4. Task 5
5. Task 3
6. Task 6

Plan complete and saved to `docs/superpowers/plans/2026-04-09-review-findings-remediation-checklist.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
