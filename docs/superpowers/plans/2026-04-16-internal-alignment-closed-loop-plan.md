# Brainfast Closed-Loop Internal Alignment — Implementation Plan

**Date:** 2026-04-16
**Scope:** Close the residual internal-structure registration gap (current NCC ≈ -0.11 vs Miki 0.66) via three complementary mechanisms stacked as a closed loop.
**Non-goal:** Match XY resolution gap (25 µm vs Miki 5 µm). That is a compute/RAM problem, not an algorithm problem, and this plan does not address it — it may remain the dominant NCC ceiling until hardware or pipeline design changes.

## Premise

The current pipeline produces **Dice 0.744** (silhouette + gross shape overlap matches Miki exactly). Internal NCC is -0.11 because:

1. **Intensity modality mismatch** — cleared-tissue fluorescence vs Allen average template (MRI-like). ANTs MI/CC struggles across modalities. *Addressable algorithmically.*
2. **Per-sample anatomical variation** — this particular mouse differs from Allen average in hippocampus shape, ventricle size, etc. *Addressable by per-sample human correction.*
3. **XY resolution ceiling** — 25 µm moving vs 5 µm Miki. *NOT addressable by this plan.*

## Three-layer architecture

```
┌──────────────────────────────────────────────────────────────────┐
│  γ  Active-learning prior store                                   │
│     Per-sample-class (e.g. ChATe27, PVe3) rolling averages of     │
│     correction vectors. Warm-starts phase α + β for new samples.  │
└──────────────────▲──────────────────────────────────▲─────────────┘
                   │                                  │
                   │ write correction vectors         │ read learned prior
                   │                                  │
┌──────────────────┴──────────────────────┐ ┌─────────┴──────────────┐
│  β  3D landmark liquify (human-in-loop) │ │  α  Intensity adapter  │
│     Upgraded from existing 2D landmark  │ │     Histogram matching │
│     UI. Accumulates sparse 3D point     │ │     + local contrast   │
│     pairs; drives a 3D TPS warp applied │ │     normalisation on   │
│     after Laplacian.                    │ │     moving volume.     │
└───────────────────▲─────────────────────┘ └────────▲───────────────┘
                    │                                │
                    └─────────── applies to ─────────┘
                                   │
                            ┌──────┴─────┐
                            │   ANTs     │
                            │ + Laplacian│
                            │ (existing) │
                            └────────────┘
```

The ordering on a fresh sample:
1. Load class-specific prior (γ) if it exists → pre-warp seed + intensity map
2. Intensity adaptation (α) applied to moving volume
3. ANTs SyN + Laplacian refinement (existing, already at Dice 0.744)
4. User opens 3D liquify UI (β), inspects overlays, corrects anatomically misaligned points
5. 3D TPS warp from β-corrections applied to annotation
6. User approves → corrections streamed into γ's store, class prior updated

## Scope by phase

Each phase lands independently. Exit criteria are verifiable.

### Phase α — Intensity adaptation (target: NCC +0.10 ~ +0.15)

**Files to add:**
- `project/scripts/intensity_adapter.py` — 3 functions:
  - `histogram_match_to_template(moving, template) → aligned_moving` (skimage.exposure.match_histograms)
  - `clahe_3d(vol, kernel_size, clip_limit) → vol` (per-slice CLAHE, stacked)
  - `edge_weighted_intensity(vol, edge_sigma) → weighted_vol` (multiply by gradient magnitude)

**Files to modify:**
- `project/scripts/registration_3d_volume.py` — in `build_volume_from_tiffs`, after scaling to uint16, optionally apply intensity adaptation based on new config keys (`registration.intensity_adapt.mode`: `off` | `hist_match` | `clahe` | `hist_match+clahe`).

**Tests:**
- `project/tests/unit/test_intensity_adapter.py` — 3 tests per function: shape preservation, value-range bounds, histogram divergence reduction.

**Exit criteria:**
- Full 646-slice run with `intensity_adapt.mode: "hist_match+clahe"` produces ANTs NCC ≥ -0.02 (currently -0.12).
- Visual overlay on slice_0300 (mid-brain) shows atlas annotation snapping more tightly to internal structures than the current baseline.

**Effort:** 1 session. Most risk: CLAHE on uint16 3D volume may be slow; mitigate by per-slice + numba/scipy only.

### Phase β — 3D landmark liquify (target: NCC +0.20 ~ +0.40 when user spends 5–10 min)

**Backend (new):**
- `project/scripts/scripts_3d_liquify.py` (or in an existing module):
  - `accumulate_3d_landmarks(job_id, z_index, pair_list)` → appends to `landmarks_3d.csv`
  - `compute_3d_tps_warp(source_pts, target_pts, vol_shape, spacing) → displacement_field` (uses scipy.interpolate.Rbf or the vendored Laplacian solver with sparse landmarks as Dirichlet conditions)
  - `apply_3d_warp(annotation_vol, displacement_field) → warped_annotation`
- `project/frontend/blueprints/api_liquify_3d.py` — REST endpoints:
  - `POST /align/liquify-3d/add-pair` (z, source xy, target xy)
  - `POST /align/liquify-3d/apply` (runs TPS, re-renders overlays)
  - `GET  /align/liquify-3d/state` (returns accumulated points + current metrics)
  - `DELETE /align/liquify-3d/pair/<i>` (remove misplaced correction)

**Frontend:**
- Reuse existing 2D landmark click handler; extend to record z-index per pair.
- Add a "3D liquify" tab showing:
  - Slice slider (z = 0…N)
  - Real + atlas overlay with drag-handle points per pair visible in current z ± window
  - "Apply 3D warp" button → re-runs Laplacian with the TPS-derived boundary conditions
- Preserve existing 2D landmark UI for single-slice calibration use-case.

**Tests:**
- Unit: TPS warp with known synthetic point pairs should produce expected displacement field (3 tests).
- Integration: full pipeline with prepared 5 synthetic landmark pairs should improve NCC on a tiny test volume.

**Exit criteria:**
- User adds 10 landmark pairs across 5 z-slices on the 35_C0_full_raw sample → post-warp NCC ≥ 0.15 (from α baseline).
- Corrections survive across re-run: next auto pipeline reads previous corrections and applies them as warm-start.

**Effort:** 2–3 sessions. Biggest unknown: scipy.interpolate.Rbf cost on ~50-200 3D points — if too slow, fall back to Laplacian-solver Dirichlet approach we already vendored.

### Phase γ — Class-prior closed loop (target: zero manual work on 2nd+ same-class sample)

**Data schema:**
```
project/train_data_set/
  class_priors/
    ChATe27/
      intensity_map.nii.gz    # mean histogram-matched transform from N samples
      landmark_prior.csv      # z, y, x (source), dy_mean, dx_mean, dz_mean, std
      sample_log.jsonl        # each correcting run appended
    PVe3/
      ...
  per_sample_corrections/
    35_C0/
      landmarks_3d.csv        # final set after user review
      affine_applied.npy
      pipeline_metrics.json
```

**Files to add:**
- `project/scripts/class_prior.py`:
  - `load_prior(sample_class) → PriorBundle | None`
  - `update_prior(sample_class, new_sample_corrections)` — running-mean style update
  - `apply_prior_warm_start(cfg, prior) → cfg` — merges into run config's initial_transform + intensity_adapt
- `project/configs/sample_class_registry.json` — maps sample ID prefix / metadata → class name

**Files to modify:**
- `project/scripts/main.py` — after loading config, call `class_prior.load_prior()` and merge
- `project/frontend/blueprints/api_liquify_3d.py` — on "Approve & Save" button, call `class_prior.update_prior()`

**UI:**
- In liquify-3d tab, add "Save to <class> prior" button
- On new-sample run, show "Loaded ChATe27 prior (from 4 samples)" banner if prior applies

**Tests:**
- Unit: prior update rolls in new point correctly (averaged into existing), shape of prior stable.
- Integration: saving a correction then loading it on a "new" sample (same input treated as fresh) shortens auto-reg NCC gap.

**Exit criteria:**
- After 3 user-corrected ChATe27 samples, the 4th ChATe27 sample's auto NCC starts from the level the user previously achieved manually (no re-correction needed).
- Sample-class prior files are version-tracked (schema doc + migration script).

**Effort:** 2–3 sessions after β is stable.

## Total estimated work

- α: 1 session
- β: 2–3 sessions
- γ: 2–3 sessions
- **Total: 5–7 sessions** across ~2–4 weeks wall-clock at current pace.

## Risks + mitigation

| Risk | Likelihood | Mitigation |
|---|---|---|
| Histogram matching destroys cell-count signal (we still need to find cells in the original fluorescence after reg) | Medium | Apply intensity adapt only to the registration-input copy; preserve original volume for downstream cellpose |
| 3D TPS on 50+ points scales O(N³); could be slow | Low | Fallback to Laplacian-sparse solver we already vendored |
| Class prior overfits to first few samples | High | Require ≥ 3 samples before auto-applying; show "low confidence" banner under that |
| Frontend 3D landmark UX is harder than 2D | High | Keep 2D UI as default; 3D is per-z slice-picker + z-neighbour window, not true 3D viewer |
| User changes sample prep between runs, class prior becomes stale | Medium | Include `sample_prep_version` field in class registry; prior invalidates on mismatch |

## What this plan deliberately does NOT do

- Does not resample moving volume to 5 µm xy — that's a compute budget decision, handled separately
- Does not replace ANTs SyN with an ML-learned registration model (too big a scope)
- Does not touch cell detection / Cellpose training (separate concern)
- Does not add 3D napari-style viewer (too expensive for web frontend; per-slice with slider is good enough for correction work)

## Acceptance gate per phase

- α done: a ChATe27 sample-35 full-density run at `intensity_adapt.mode=hist_match+clahe` shows NCC ≥ -0.02 and visual overlay improvement.
- β done: user can correct 10 pairs on 3 z-slices, see NCC push ≥ 0.15, and corrections persist.
- γ done: 4th ChATe27 sample runs with auto prior → NCC ≥ 0.15 out of the box without user action.
