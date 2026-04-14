# Brainfast Next-Step Remediation Plan For Claude

> **Scope:** This plan is for the next implementation pass only. It is not a broad roadmap. It is a hardening plan that closes the four still-valid findings before any new "done" claim is acceptable.

**Goal:** Restore trust in validation, make new outputs auditable, and prevent disputed `ml_flip` conclusions from being presented as settled before the evidence supports them.

**Why this plan is stricter than the prior one:** It does not ask for more features first. It first fixes the places where the repo can falsely look healthy:

- `check_env.py` says OK while real imports still crash
- new tests exist but do not run in the current broken numerical environment
- `ml_flip` result artifacts are overwritten between samples
- the summary document makes a stronger claim than the retained artifacts support

---

## Non-Negotiable Rules

- Do not touch `project/configs/run_config.template.json` for `ml_flip` in this pass unless the new evidence gate is satisfied.
- Do not mark this pass complete if `check_env.py` still passes while `pytest` import-time failures remain reproducible.
- Do not keep using shared `ab_test_mlflip_false` / `ab_test_mlflip_true` directories for multi-sample evidence.
- Do not publish a matrix-backed `ml_flip` conclusion unless the referenced CSVs, QC files, overlays, and config files are all retained and mutually consistent.

---

## Workstream 1: Fix Runtime Truth, Not Just Top-Level Imports

**Primary finding addressed**
- `project/scripts/check_env.py` still misses the broken SciPy path the runtime actually uses.

**Files**
- Modify: `project/scripts/check_env.py`
- Modify: `README.md`
- Modify: `.github/workflows/test.yml`

### Task 1.1: Validate runtime hotspots, not package shells

Current problem:
- `__import__("scipy")` passes
- `scipy.ndimage` import still fails in the same environment used by `server_context.py` and `whole_brain_3d.py`

Required implementation:
- add targeted import checks for:
  - `scipy.ndimage`
  - `skimage.segmentation`
  - `project.frontend.server_context`
  - `project.scripts.whole_brain_3d`
- run these checks in isolated subprocesses so one bad import does not poison the current interpreter state

Required behavior:
- if any of the runtime hotspots fail to import, `check_env.py` returns non-zero
- failure output must name the exact import target that failed

### Task 1.2: Align docs with real guarantees

Update docs so they only claim what the code now proves:

- `check_env.py` verifies runtime importability of the configured path
- it is not a shallow package-presence check

### Task 1.3: Verify against the currently broken machine state

Run:

```powershell
python project/scripts/check_env.py --config project/configs/run_config.template.json
python -m pytest project/tests/unit/test_detect_preview_api.py -q
python -m pytest project/tests/integration/test_default_path_smoke.py -q
```

Exit condition:
- if pytest still dies at import time, `check_env.py` must also fail
- there must no longer be a state where `check_env.py` says OK while these imports are still broken

---

## Workstream 2: Make Detector Preview Use the Real Active Config

**Primary finding addressed**
- `project/frontend/blueprints/api_detect_preview.py` does not use the actual active run config.

**Files**
- Modify: `project/frontend/blueprints/api_detect_preview.py`
- Modify: `project/tests/unit/test_detect_preview_api.py`
- Modify: `project/frontend/README.md`

### Task 2.1: Replace the dead `run_state["config"]` lookup

Current problem:
- `run_state` stores `config_path`, not a parsed `config` object
- detector preview therefore falls back to a hard-coded minimal config

Required implementation:
- load the active config from `ctx.run_state["config_path"]`, or reuse a shared helper that resolves the active config file
- if no prior run exists, fall back to `run_config.template.json` explicitly, not an ad hoc mini-config

### Task 2.2: Preserve the real detector knobs

Detector preview must use the same values the pipeline would use for:

- `primary_model`
- `secondary_model`
- `cellpose_channels`
- `cellpose_diameter_um`
- `cellpose_flow_threshold`
- `cellpose_cellprob_threshold`
- `fallback_model`
- `auto_switch_on_distortion`
- `input.pixel_size_um_xy`

### Task 2.3: Add the missing correctness tests

Add tests that verify:

- the route reads from `config_path`, not `run_state["config"]`
- custom detector values from the config reach `_run_detection`
- no prior run still uses the template config rather than the hard-coded fallback block
- runtime failure still surfaces a clear detector/runtime error

### Exit condition

Detector preview is only done when:

- it uses the same detector configuration as the active pipeline config
- its tests no longer depend on a nonexistent `run_state["config"]`

---

## Workstream 3: Make `ml_flip` Evidence Durable and Auditable

**Primary findings addressed**
- `project/scripts/ml_flip_ab_test.py` reruns erase prior sample artifacts.
- `docs/superpowers/plans/ml-flip-matrix-summary.md` is stronger than the retained artifact set supports.

**Files**
- Modify: `project/scripts/ml_flip_ab_test.py`
- Modify: `docs/superpowers/plans/ml-flip-matrix-summary.md`
- Modify: `docs/superpowers/plans/ml-flip-ab-evidence.md`
- Create: `docs/superpowers/plans/ml-flip-audit-manifest.md`

### Task 3.1: Stop overwriting sample outputs

Required output layout:

```text
project/outputs/ml_flip_ab/
  sample_35/
    ml_flip_false/
    ml_flip_true/
    summary.csv
    manifest.json
  sample_41/
    ml_flip_false/
    ml_flip_true/
    summary.csv
    manifest.json
  sample_44/
    ml_flip_false/
    ml_flip_true/
    summary.csv
    manifest.json
```

Required implementation:
- include sample label in the heavy output directory path, not just the summary CSV name
- never delete another sample's prior output

### Task 3.2: Emit a per-sample manifest

Each manifest must record:

- config path
- input dir
- sample label
- timestamps
- exact output directories for `ml_flip=false` and `ml_flip=true`
- summary CSV path
- whether the run used Cellpose or fell back to LoG

### Task 3.3: Downgrade claims until the retained evidence matches them

Before changing the summary back to a strong recommendation, first make it reproducible.

Immediate required edits:
- remove any citation to stale or mismatched artifacts
- if sample 35's retained CSV does not match the narrative, say so and mark that sample as needing rerun
- do not describe the result as a completed evidence matrix if all tested samples are still the same sample class

### Exit condition

This workstream is only done when:

- all three sample runs can coexist on disk
- the summary document points to the actual retained artifact paths
- every number in the summary can be traced back to a specific CSV still present in the repo outputs

---

## Workstream 4: Put the `ml_flip` Default Behind an Explicit Evidence Gate

**Primary finding handled carefully**
- `ml_flip=false` may still be wrong for some sample classes, but the current retained evidence is not broad enough to justify either a default flip change or a strong defense of the current default.

**Files**
- Modify: `docs/superpowers/plans/ml-flip-matrix-summary.md`
- Modify: `project/configs/run_config.template.json` only if this workstream passes

### Task 4.1: Separate "tested sample class" from "global default"

Required statement:
- current evidence is about left-hemi `right_flipped` samples
- current evidence is not yet enough to decide the global template default for all future samples

### Task 4.2: Define the sample classes required before changing the default

Minimum classes:

- left-hemi `right_flipped`
- a genuine non-`right_flipped` class if such inputs are expected in production
- any whole or orientation-ambiguous class actually used by the user, if it exists

If the repo does not currently contain those classes:
- say that explicitly
- keep the default decision open
- do not promote the current result to a universal default rule

### Task 4.3: Decision rule

Only change `project/configs/run_config.template.json` if:

- evidence covers the sample classes you expect users to run
- the retained artifacts are reproducible
- the summary document no longer depends on stale outputs

Otherwise:
- leave the default unchanged
- state that the default remains provisional

### Exit condition

This workstream is done when the repo is honest about the strength of the `ml_flip` evidence, even if the final answer is "do not change the default yet."

---

## Workstream 5: Make CI Prove the Fixed State, Not Just Contain New Files

**Files**
- Modify: `.github/workflows/test.yml`

### Task 5.1: Make the default-path smoke lane depend on the new environment truth check

Required order:

1. install full stack
2. run hardened `check_env.py`
3. run the default-path smoke test
4. run the detector-preview unit test

### Task 5.2: Use the tests that were added, but only after Workstream 1 is complete

Required commands:

```powershell
python project/scripts/check_env.py --config project/configs/run_config.template.json
python -m pytest project/tests/unit/test_detect_preview_api.py -q
python -m pytest project/tests/integration/test_default_path_smoke.py -q
```

Acceptance:
- if these tests still fail because of import-time numerical stack breakage, CI must fail before anyone can claim the default path is healthy

---

## Handoff Checklist For Claude

Claude is not done until all of the following are true:

- `check_env.py` fails on this machine if `scipy.ndimage` or `whole_brain_3d` import still fails
- detector preview uses the active config file, not a dead `run_state["config"]` lookup
- `ml_flip` A/B artifacts are stored per sample and no longer overwrite each other
- `ml_flip` summary citations point to real retained files whose numbers match the document
- the repo does not pretend the `ml_flip` default is settled unless the evidence gate is actually met
- CI runs the new detector-preview and default-path tests under the hardened environment gate

---

## Immediate Priority Order

1. Workstream 1
2. Workstream 2
3. Workstream 3
4. Workstream 5
5. Workstream 4

If Claude wants to argue about the `ml_flip` default before Workstreams 1-3 are closed, that is a process failure, not progress.
