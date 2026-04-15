# Stability Exit Criteria

> **Purpose:** Define the hard gates that must pass before the Brainfast pipeline
> is declared "stable for other samples" and a release candidate is cut.

---

## 1. Repo-Level Gates

All four gates must be green on the same commit:

| # | Gate | Command | Pass condition |
|---|------|---------|----------------|
| R1 | Lint | `ruff check project/` | Zero violations |
| R2 | Unit tests | `python -m pytest project/tests/unit/ -q` | All pass, 0 errors |
| R3 | Env validation | `python project/scripts/check_env.py --config <cfg>` | Exits 0 for valid config; exits non-zero for missing dep or config key |
| R4 | Default path docs | `README.md` "Cellpose-SAM" section exists | `miki_3d + cpsam` documented and reproducible |

**Current status:** R1-R4 all green (commit `081ed61`).

---

## 2. Sample-Level Gates

Progression is strictly sequential; each stage must pass before advancing.

### Stage A: Canary validation (3 buckets)

| Bucket | Data | Slices | Config | Threshold |
|--------|------|--------|--------|-----------|
| A | `data/35_C0_demo` | 111 | `run_config_35.json` | Cross-modality (Dice >= 0.30, NMI >= 1.01) |
| B | `data/35_C0_test` | 5 | `run_config_35_canary_b.json` | Sparse (Dice >= 0.01, NMI >= 1.01) |
| C | `data/35_C0_full` | 100 | `run_config_35_canary_c.json` | Cross-modality (Dice >= 0.30, NMI >= 1.01) |

**Requirement:** Each bucket must pass `validate_canary.py` **twice** (independent runs).

```bash
# Example for bucket A
python scripts/main.py --config configs/run_config_35.json --run-real-input data/35_C0_demo
python scripts/validate_canary.py --output-dir outputs/35_C0_demo --canary A
```

### Stage B: 20% cohort

- Select 20% of available samples (round up).
- Run full pipeline on each.
- All must complete without crash (Gate 1 of validate_canary).
- Registration Dice >= 0.50 for every sample.

### Stage C: 60% cohort

- Same criteria as Stage B, expanded to 60% of samples.

### Stage D: Full batch

- Run remaining samples.
- All complete; only **known, triaged** exceptions are acceptable.
- "Known" = documented in `registration-failure-taxonomy.md` with a workaround.

**Current status:** Stage A in progress (run 1 of Canary A executing).

---

## 3. Release Candidate Freeze

Once Stages A-D pass:

1. Generate exact environment snapshot:
   ```bash
   python -m pip freeze > outputs/release_candidate_requirements.txt
   ```
2. Tag the commit:
   ```bash
   git tag -a v1.0-rc1 -m "Release candidate: all canaries green, cohort validated"
   ```
3. Archive the snapshot alongside the tag in the repo.

**Current status:** `release_candidate_requirements.txt` generated (167 packages, Python 3.11).

---

## 4. Automated Gate Validation

The `validate_canary.py` script checks 6 gates per run:

| Gate | Check | Fail condition |
|------|-------|----------------|
| 1 | No crash | `pipeline_progress.json` missing or `status != "completed"` |
| 2 | QC outputs | `<output>/qc/` missing or empty |
| 3 | Registration quality | Dice < threshold, SSIM < 0.05, or NMI < 1.01 |
| 4 | Truth coverage | `cells_per_region.csv` missing or has 0 regions |
| 5 | Cell detection | `cells_mapped.csv` missing or has 0 cells |
| 6 | Orientation | Reserved (manual check for L/R flip) |

---

## 5. Rollback Policy

If any stage fails:
1. **Do not advance** to the next stage.
2. Root-cause the failure and classify it per `registration-failure-taxonomy.md`.
3. Fix on the same branch; re-run the failing bucket from scratch.
4. All previously-passed stages must be re-validated after the fix.

---

## What "done" means

The pipeline is declared stable when:
- A clean machine can `pip install -e ".[dev]"` and pass R1-R4.
- `check_env.py` correctly rejects invalid configurations.
- 3/3 canary buckets pass twice each.
- 60%+ of the sample cohort completes without new failure classes.
- A `release_candidate_requirements.txt` is frozen and tagged.

---

## 6. Release readiness gate

Before tag creation:
- clean-room install of `.[full,desktop,dev]`
- `check_env.py` green
- `pytest project/tests -q` green
- one manual browser acceptance pass recorded
- `release_candidate_requirements.txt` frozen from the candidate commit
