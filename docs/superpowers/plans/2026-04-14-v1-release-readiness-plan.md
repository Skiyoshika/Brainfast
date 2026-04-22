# Brainfast v1 Release Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a first-generation Brainfast release that installs cleanly, runs the default `miki_3d + cpsam` path, exposes current workflow limits honestly, and produces reproducible release evidence from one tagged commit.

**Architecture:** Treat release as an evidence problem, not a feature problem. Freeze one clean commit, unify version metadata generation, harden the release workflow to validate the same runtime the product ships, and require both automated and manual acceptance artifacts before cutting the first public tag.

**Tech Stack:** Python 3.10/3.11, Flask, NumPy/SciPy/scikit-image, ANTsPyX, Cellpose-SAM via Cellpose v4 (`cpsam`), pytest, ruff, GitHub Actions, PyInstaller, Windows signtool.

---

## File Map

- `.github/workflows/release.yml`
  Purpose: tag-triggered Windows build and GitHub Release pipeline; currently builds without the same validation depth as the main test workflow.
- `README.md`
  Purpose: public install/runtime/support contract; currently contains stale release-facing statements such as the old test count and incomplete install guidance.
- `project/scripts/build_version_json.py`
  Purpose: release workflow version writer; currently emits a different JSON contract than `project/version.json` and `write_version_metadata.py`.
- `project/scripts/write_version_metadata.py`
  Purpose: canonical version metadata writer; already matches the tested `project/version.json` schema.
- `project/tests/unit/test_write_version_metadata.py`
  Purpose: asserts the shipped `version.json` schema; should be extended so the release workflow path cannot drift.
- `project/tests/integration/test_default_path_smoke.py`
  Purpose: proves the shipped default path can start on a fully provisioned runtime.
- `project/scripts/check_env.py`
  Purpose: runtime validation entrypoint; currently green and should be treated as a release gate.
- `docs/superpowers/plans/stability-exit-criteria.md`
  Purpose: sample-stability gate definitions; should stay aligned with the release plan rather than competing with it.
- `docs/release/known-limitations.md`
  Purpose: new user-facing scope and limitation document for the first release.
- `docs/release/manual-acceptance.md`
  Purpose: new exact operator script for the UI and exported-output acceptance run.
- `docs/release/rc1-cut-checklist.md`
  Purpose: new release captain checklist for clean-room verification, artifact collection, signing, tag creation, and GitHub Release publication.

## Current Verified Baseline (2026-04-14)

These commands were run in `D:\Brainfast` before writing this plan:

```powershell
python project/scripts/check_env.py --config project/configs/run_config.template.json
python -m pytest project/tests -q
python -m ruff check project/scripts/ project/frontend/blueprints/ project/frontend/server_context.py project/frontend/app_metadata.py project/frontend/update_checker.py
git -c core.excludesfile= status --short --untracked-files=all
```

Observed result:

- `check_env.py`: exit `0`
- `pytest`: `294 passed, 32 warnings`
- `ruff`: `All checks passed!`
- `git status`: clean worktree

This is enough to start release-readiness work. It is not enough to cut a public release, because the repo still lacks a clean release contract, clean-room proof, and a user-facing support boundary.

---

### Task 1: Unify the version metadata contract used by releases

**Files:**
- Create: `project/tests/unit/test_build_version_json.py`
- Modify: `project/scripts/build_version_json.py`
- Modify: `project/tests/unit/test_write_version_metadata.py`

- [ ] **Step 1: Write the failing tests for the release-writer path**

Create `project/tests/unit/test_build_version_json.py` with:

```python
from __future__ import annotations

import json
import os
import runpy
from pathlib import Path


def test_build_version_json_matches_release_contract(tmp_path, monkeypatch):
    out_path = tmp_path / "version.json"
    monkeypatch.setenv("GITHUB_REF_NAME", "v1.0.0")
    monkeypatch.setenv("GITHUB_SHA", "abcdef1234567890")
    monkeypatch.setenv("BRAINFAST_VERSION_JSON", str(out_path))

    runpy.run_module("project.scripts.build_version_json", run_name="__main__")

    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert {
        "version",
        "build_date",
        "commit",
        "repository",
        "releases_api",
        "releases_page",
    } <= set(payload)
    assert payload["version"] == "1.0.0"
    assert payload["commit"] == "abcdef1"
```

- [ ] **Step 2: Run the new test and confirm the current writer drifts**

Run:

```powershell
python -m pytest project/tests/unit/test_build_version_json.py -q
```

Expected: FAIL because `build_version_json.py` currently omits `repository`, `releases_api`, and `releases_page`.

- [ ] **Step 3: Replace duplicate release metadata logic with the canonical writer**

Modify `project/scripts/build_version_json.py` to delegate to `write_version_metadata.py`:

```python
from __future__ import annotations

import os
import subprocess
from datetime import date
from pathlib import Path

from project.scripts.write_version_metadata import main as write_version_main

REPO_ROOT = Path(__file__).parent.parent.parent


def _git_short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, cwd=REPO_ROOT
        ).strip()
    except Exception:
        return os.environ.get("GITHUB_SHA", "unknown")[:7]


if __name__ == "__main__":
    version = os.environ.get("GITHUB_REF_NAME", "").lstrip("v") or "0.0.0-dev"
    commit = _git_short_sha()
    output = os.environ.get(
        "BRAINFAST_VERSION_JSON",
        str(REPO_ROOT / "project" / "version.json"),
    )
    raise SystemExit(
        write_version_main(
            [
                "--version",
                version,
                "--commit",
                commit,
                "--build-date",
                str(date.today()),
                "--output",
                output,
            ]
        )
    )
```

Modify `project/scripts/write_version_metadata.py` so `main()` accepts an optional argv list:

```python
def main(argv: list[str] | None = None) -> int:
    ...
    args = parser.parse_args(argv)
```

- [ ] **Step 4: Expand the existing contract test so both writers stay locked together**

Add to `project/tests/unit/test_write_version_metadata.py`:

```python
def test_build_version_writer_and_metadata_writer_share_same_schema(tmp_path, monkeypatch):
    from project.scripts.write_version_metadata import main as write_main

    out_path = tmp_path / "version.json"
    rc = write_main(
        [
            "--version",
            "v1.0.0",
            "--commit",
            "abcdef1",
            "--build-date",
            "2026-04-14",
            "--output",
            str(out_path),
        ]
    )
    assert rc == 0
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["repository"].endswith("/Brainfast")
```

- [ ] **Step 5: Verify the release-writer path end to end**

Run:

```powershell
python -m pytest project/tests/unit/test_build_version_json.py project/tests/unit/test_write_version_metadata.py -q
python project/scripts/build_version_json.py
Get-Content project/version.json
```

Expected:
- all targeted tests pass
- `project/version.json` contains `version`, `build_date`, `commit`, `repository`, `releases_api`, `releases_page`

- [ ] **Step 6: Commit**

```powershell
git add project/scripts/build_version_json.py project/scripts/write_version_metadata.py project/tests/unit/test_build_version_json.py project/tests/unit/test_write_version_metadata.py
git commit -m "release: unify version metadata writer contract"
```

---

### Task 2: Rewrite the public release contract so it matches the actual product

**Files:**
- Modify: `README.md`
- Create: `docs/release/known-limitations.md`
- Create: `docs/release/install-modes.md`

- [ ] **Step 1: Replace stale public claims in `README.md`**

Update these sections:

````markdown
## Quick start

### Install modes

#### Minimal 2D runtime
```powershell
pip install -e ".[dev]"
```

#### Default recommended runtime (`miki_3d + cpsam`)
```powershell
pip install -e ".[full,dev]"
```

#### Packaging / desktop build
```powershell
pip install -e ".[full,desktop,dev]"
```
````

Replace stale bullets such as:
- `97 unit tests`
- instructions that only install `.[advanced,dev]`

With current, stable wording:
- `validated by the full pytest suite in this repo`
- default release path = `whole-brain miki_3d + cpsam`

- [ ] **Step 2: Add an explicit workflow-boundary section**

Insert this section into `README.md`:

```markdown
## Current workflow boundaries

- 2D manual landmark / liquify correction is available in the browser UI.
- The 3D whole-brain path is the default final-truth path and is still primarily automatic.
- Cellpose-SAM (`cpsam`) is available as the default detector backend.
- The product does not yet ship a detector-specific manual relabel / retrain workflow in the UI.
- Region-level counts should only be trusted when registration QC is acceptable.
```

- [ ] **Step 3: Create `docs/release/known-limitations.md`**

Create the file with:

```markdown
# Brainfast v1 Known Limitations

## Supported path

- Windows 10/11 primary support
- Python 3.10 or 3.11
- Default runtime: `miki_3d + cpsam`
- Cleared-tissue mouse samples using the Allen CCFv3 assets shipped in this repo

## Known limitations

- 3D whole-brain registration still requires QC review before trusting region-level outputs.
- UI manual correction currently improves the 2D preview path; it is not yet a full 3D detector-training loop.
- `ml_flip=false` remains provisional outside the tested left-hemisphere `right_flipped` sample class documented in `docs/superpowers/plans/ml-flip-matrix-summary.md`.
- GPU improves Cellpose performance but is not required for the fallback-free default path to start.
```

- [ ] **Step 4: Create `docs/release/install-modes.md`**

Create the file with:

```markdown
# Brainfast Install Modes

| Mode | Command | Use case |
|------|---------|----------|
| Minimal | `pip install -e ".[dev]"` | 2D development, tests, docs work |
| Default runtime | `pip install -e ".[full,dev]"` | shipped `miki_3d + cpsam` path |
| Desktop build | `pip install -e ".[full,desktop,dev]"` | Windows EXE build and signing |
```

- [ ] **Step 5: Verify docs consistency against the live repo state**

Run:

```powershell
python -m pytest project/tests -q
python project/scripts/check_env.py --config project/configs/run_config.template.json
```

Expected: still green after doc-only edits; README statements match those command results.

- [ ] **Step 6: Commit**

```powershell
git add README.md docs/release/known-limitations.md docs/release/install-modes.md
git commit -m "docs: align release contract with shipped runtime"
```

---

### Task 3: Make the release workflow prove the same path the product ships

**Files:**
- Modify: `.github/workflows/release.yml`
- Modify: `.github/workflows/test.yml`

- [ ] **Step 1: Write the failing release-workflow contract test in prose**

Before editing YAML, record the contract in the PR description or task notes:

- tag builds must install `.[full,dev]`
- tag builds must run `check_env.py`
- tag builds must run `test_default_path_smoke.py`
- tag builds must write `project/version.json` with the canonical schema
- signing must be optional but deterministic

- [ ] **Step 2: Harden `.github/workflows/release.yml`**

Replace the current install/build block with:

```yaml
- name: Install dependencies
  run: pip install -e ".[full,desktop,dev]"

- name: Validate runtime
  run: python project/scripts/check_env.py --config project/configs/run_config.template.json

- name: Run release smoke tests
  run: |
    python -m pytest project/tests/integration/test_default_path_smoke.py -q
    python -m pytest project/tests/unit/test_detect_preview_api.py -q

- name: Write version.json
  env:
    GITHUB_REF_NAME: ${{ github.ref_name }}
    GITHUB_SHA: ${{ github.sha }}
  run: python project/scripts/build_version_json.py
```

- [ ] **Step 3: Add optional signing without breaking unsigned RC builds**

Append to `.github/workflows/release.yml`:

```yaml
- name: Sign release asset
  if: ${{ secrets.BRAINFAST_CODESIGN_PFX != '' && secrets.BRAINFAST_CODESIGN_PASSWORD != '' }}
  shell: pwsh
  env:
    BRAINFAST_CODESIGN_PFX: ${{ secrets.BRAINFAST_CODESIGN_PFX }}
    BRAINFAST_CODESIGN_PASSWORD: ${{ secrets.BRAINFAST_CODESIGN_PASSWORD }}
  run: |
    python project/scripts/sign_release_assets.py `
      --file dist/brainfast-${{ github.ref_name }}-windows.exe `
      --pfx $env:BRAINFAST_CODESIGN_PFX `
      --password $env:BRAINFAST_CODESIGN_PASSWORD
```

- [ ] **Step 4: Keep `test.yml` and `release.yml` aligned**

Ensure both workflows install the same shipped default path for runtime-sensitive checks:

```yaml
pip install -e ".[full,dev]"
```

Do not let `release.yml` silently use a shallower dependency set than the smoke workflow.

- [ ] **Step 5: Verify the workflow inputs locally**

Run:

```powershell
python project/scripts/check_env.py --config project/configs/run_config.template.json
python -m pytest project/tests/integration/test_default_path_smoke.py -q
python -m pytest project/tests/unit/test_detect_preview_api.py -q
python project/scripts/build_version_json.py
Get-Content project/version.json
```

Expected:
- `check_env.py` passes
- both targeted test files pass
- `version.json` is valid and complete

- [ ] **Step 6: Commit**

```powershell
git add .github/workflows/release.yml .github/workflows/test.yml project/version.json
git commit -m "release: gate tagged builds with shipped runtime smoke checks"
```

---

### Task 4: Add a clean-room release candidate procedure that another machine can reproduce

**Files:**
- Create: `docs/release/rc1-cut-checklist.md`
- Create: `docs/release/manual-acceptance.md`
- Modify: `docs/superpowers/plans/stability-exit-criteria.md`

- [ ] **Step 1: Create `docs/release/rc1-cut-checklist.md`**

Use this content:

````markdown
# Brainfast RC1 Cut Checklist

## Clean-room machine

1. Clone the repo at the candidate commit.
2. Create a fresh venv.
3. Run:
   ```powershell
   pip install -e ".[full,desktop,dev]"
   python project/scripts/check_env.py --config project/configs/run_config.template.json
   python -m pytest project/tests -q
   ```
4. Save:
   - `python -m pip freeze > release_candidate_requirements.txt`
   - `python -VV > python_runtime.txt`
   - `git rev-parse HEAD > git_commit.txt`
```
````

- [ ] **Step 2: Create `docs/release/manual-acceptance.md`**

Use this operator script:

````markdown
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
```
````

- [ ] **Step 3: Tie the clean-room checklist back into the existing stability gate doc**

Append to `docs/superpowers/plans/stability-exit-criteria.md`:

```markdown
## 6. Release readiness gate

Before tag creation:
- clean-room install of `.[full,desktop,dev]`
- `check_env.py` green
- `pytest project/tests -q` green
- one manual browser acceptance pass recorded
- `release_candidate_requirements.txt` frozen from the candidate commit
```

- [ ] **Step 4: Verify the checklists are executable, not aspirational**

Run:

```powershell
python -m pytest project/tests -q
python project/scripts/check_env.py --config project/configs/run_config.template.json
```

Expected: both commands succeed and match the steps written into the checklist.

- [ ] **Step 5: Commit**

```powershell
git add docs/release/rc1-cut-checklist.md docs/release/manual-acceptance.md docs/superpowers/plans/stability-exit-criteria.md
git commit -m "docs: add release candidate and manual acceptance checklists"
```

---

### Task 5: Capture the release evidence bundle from a single candidate commit

**Files:**
- Create: `docs/release/release-evidence.md`
- Create: `docs/release/release-artifact-layout.md`

- [ ] **Step 1: Define the artifact bundle layout**

Create `docs/release/release-artifact-layout.md` with:

```markdown
# Release Artifact Layout

Expected bundle for `v1.0.0-rc1`:

- `release_candidate_requirements.txt`
- `python_runtime.txt`
- `git_commit.txt`
- `check_env.txt`
- `pytest.txt`
- `ruff.txt`
- `manual-acceptance-notes.md`
- `sample_35_summary.csv`
- `sample_41_summary.csv`
- `sample_44_summary.csv`
- `project/version.json`
- `brainfast-v1.0.0-rc1-windows.exe`
```

- [ ] **Step 2: Create `docs/release/release-evidence.md`**

Use this template:

```markdown
# Brainfast Release Evidence

## Candidate commit

- commit:
- tag:
- build date:

## Automated gates

- `check_env.py`: PASS
- `pytest project/tests -q`: PASS
- `ruff check ...`: PASS

## Manual acceptance

- detector preview: PASS / FAIL
- whole-brain sample run: PASS / FAIL
- export download: PASS / FAIL

## Attached artifacts

- requirements snapshot
- runtime snapshot
- sample summaries
- signed or unsigned EXE
```

- [ ] **Step 3: Collect the evidence on the candidate commit**

Run:

```powershell
python project/scripts/check_env.py --config project/configs/run_config.template.json *> release-check_env.txt
python -m pytest project/tests -q *> release-pytest.txt
python -m ruff check project/scripts/ project/frontend/blueprints/ project/frontend/server_context.py project/frontend/app_metadata.py project/frontend/update_checker.py *> release-ruff.txt
Copy-Item project/outputs/ml_flip_ab/sample_35/summary.csv release-sample_35_summary.csv
Copy-Item project/outputs/ml_flip_ab/sample_41/summary.csv release-sample_41_summary.csv
Copy-Item project/outputs/ml_flip_ab/sample_44/summary.csv release-sample_44_summary.csv
```

- [ ] **Step 4: Verify the evidence files are present**

Run:

```powershell
Get-ChildItem release-*.txt, release-sample_*_summary.csv
```

Expected: all six files exist and come from the same commit.

- [ ] **Step 5: Commit**

```powershell
git add docs/release/release-evidence.md docs/release/release-artifact-layout.md
git commit -m "docs: define release evidence bundle"
```

---

### Task 6: Cut the first public release without skipping the gates

**Files:**
- Modify: `README.md`
- Modify: `project/version.json`

- [ ] **Step 1: Re-run the final release gates on the exact candidate commit**

Run:

```powershell
python project/scripts/check_env.py --config project/configs/run_config.template.json
python -m pytest project/tests -q
python -m ruff check project/scripts/ project/frontend/blueprints/ project/frontend/server_context.py project/frontend/app_metadata.py project/frontend/update_checker.py
git -c core.excludesfile= status --short --untracked-files=all
```

Expected:
- all three validation commands pass
- `git status` is empty

- [ ] **Step 2: Freeze the version metadata**

Run:

```powershell
$env:GITHUB_REF_NAME="v1.0.0-rc1"
python project/scripts/build_version_json.py
Get-Content project/version.json
```

Expected: `project/version.json` reports `1.0.0-rc1` and the correct short commit.

- [ ] **Step 3: Build the Windows binary locally once before tagging**

Run:

```powershell
pyinstaller `
  --onefile `
  --name brainfast `
  --add-data "project/frontend/index.html;project/frontend" `
  --add-data "project/frontend/app.js;project/frontend" `
  --add-data "project/frontend/styles.css;project/frontend" `
  --add-data "project/version.json;project" `
  --hidden-import=flask `
  --hidden-import=scipy `
  --hidden-import=skimage `
  project/frontend/server.py
```

Expected: `dist/brainfast.exe` exists before the GitHub tag build runs.

- [ ] **Step 4: Tag only after evidence and manual acceptance are both attached**

Run:

```powershell
git tag -a v1.0.0-rc1 -m "Brainfast first release candidate"
git push origin v1.0.0-rc1
```

Expected: GitHub Actions `Release` workflow starts from the tagged commit.

- [ ] **Step 5: Promote to the first public release only after reviewing the RC artifacts**

Promotion rule:
- no missing release artifacts
- no red job in `release.yml`
- manual acceptance notes attached
- `known-limitations.md` published alongside the release

- [ ] **Step 6: Commit**

```powershell
git add README.md project/version.json
git commit -m "release: prepare v1.0.0-rc1 cutover"
```

---

## Exit Criteria

This plan is complete only when all of the following are true on the same commit:

- `python project/scripts/check_env.py --config project/configs/run_config.template.json` passes
- `python -m pytest project/tests -q` passes
- `python -m ruff check project/scripts/ project/frontend/blueprints/ project/frontend/server_context.py project/frontend/app_metadata.py project/frontend/update_checker.py` passes
- `project/version.json` is generated by the canonical writer path and matches the tested schema
- `README.md` and `docs/release/known-limitations.md` describe the actual shipped workflow boundaries
- `release.yml` validates the shipped default runtime before building the EXE
- one clean-room run and one browser acceptance run have attached evidence
- the tagged release artifact bundle is reproducible from one clean commit

Plan complete and saved to `docs/superpowers/plans/2026-04-14-v1-release-readiness-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
