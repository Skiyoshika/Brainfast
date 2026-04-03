# Brainfast Miki-Style 3D Whole-Brain Registration Design
**Date:** 2026-04-02  
**Status:** Draft for review  
**Goal:** Make whole-brain automatic registration use a Miki-style 3D volume-first pipeline as the system truth for visualization, QC, mapping, and counting.  
**Decision Summary:** Whole-brain mode moves from slice-first 2D registration to a native 3D pipeline. Existing 2D tools remain available only as preview and manual-correction helpers.

---

## 1. Background

The current Brainfast whole-brain flow is still centered on per-slice 2D logic:

`TIFF slices -> per-slice atlas auto-pick -> per-slice warp -> overlay -> slice QC -> mapping/counting`

This creates three structural problems:

1. Registration quality ceiling is lower than the `Sample\Miki` reference because each slice is solved mostly independently.
2. Display and QC can be misleading because current slice-level outputs mix rendering concerns with truth-generation concerns.
3. Downstream quantification still depends on 2D-derived labels instead of one globally consistent whole-brain registration result.

The `Sample\Miki` reference uses a different pattern:

`TIFF stack -> 3D volume -> hemisphere/template prep -> ANTs registration -> Laplacian refinement -> final registered annotation volume -> slice exports`

That volume-first structure is why the Miki outputs look cleaner, more stable, and more globally consistent.

---

## 2. Product Goal

For whole-brain automatic runs, Brainfast should treat the 3D registered volume as the only truth source.

That means:

- Every slice overlay is derived from the final 3D registered annotation volume.
- Whole-brain QC is based on true 3D registration metrics plus slice inspection exports.
- Cell mapping and count aggregation use 3D-derived annotations, not independent 2D slice registration.
- Existing 2D tools remain available only as auxiliary preview and manual-correction tools.

This design prioritizes registration quality over runtime. Long-running stages are acceptable as long as the UI provides clear progress, logs, and failure states.

---

## 3. Scope

### In Scope

- New default whole-brain registration architecture based on a native Miki-style 3D pipeline
- ANTs as the primary global registration dependency
- Laplacian refinement stage after ANTs
- 3D-derived truth for overlay export, QC, cell mapping, and count outputs
- UI progress model with stage-level status and logs
- Demotion of current 2D logic from primary truth-generation to auxiliary tooling

### Out of Scope

- Replacing or deleting all existing 2D tools
- Rebuilding manual correction UX in this design pass
- Supporting a second independent truth pipeline for whole-brain mode
- Optimizing for fast preview-first whole-brain runs

---

## 4. Chosen Approach

Three approaches were considered:

1. Wrap the external Miki pipeline directly
2. Rebuild a native Miki-style 3D pipeline inside Brainfast
3. Ship a temporary hybrid on top of the current `run_3d_registration.py`

The chosen approach is **Option 2: Native Miki-Style Pipeline**.

Rationale:

- It matches the target quality class more closely than tuning the current 2D workflow.
- It keeps Brainfast as a coherent product rather than a thin wrapper around an external reference folder.
- It allows one consistent truth path for visualization, QC, and quantification.

---

## 5. Target Architecture

### 5.1 Whole-Brain Automatic Truth Path

The new whole-brain path is:

`raw TIFF stack -> 3D volume build -> hemisphere/template prep -> ANTs registration -> Laplacian refinement -> final registered annotation volume`

All downstream artifacts derive from that result:

- per-slice registered labels
- per-slice overlays
- whole-brain QC summary
- cell-to-region mapping
- leaf and hierarchy count CSVs

### 5.2 2D Tools Boundary

The following must leave the whole-brain primary truth path:

- per-slice auto-pick as the final truth selector
- per-slice auto registration as the whole-brain result generator
- slice-level `best_score = 1.0` style acceptance logic
- any quantification path that depends on 2D-derived labels instead of the 3D volume result

The following remain valid as auxiliary tools:

- single-slice preview
- manual landmark correction
- liquify/local correction
- slice-level inspection of exported 3D truth

### 5.3 Mode Contract

`Whole-brain mode`:
- always uses the 3D truth path
- exports slice views from the 3D result
- drives mapping and counts from the 3D result

`Single-slice / manual mode`:
- may continue using 2D tools
- is explicitly marked as auxiliary/manual
- does not silently replace whole-brain truth outputs

---

## 6. Stage Contracts

The new whole-brain automatic flow is split into six user-visible stages.

### Stage 1. Volume Build

Input:
- raw TIFF slice directory

Outputs:
- `brain_raw.nii.gz`
- `brain_25um.nii.gz`
- `volume_metadata.json`
- preview snapshots and voxel metadata

Purpose:
- normalize the raw TIFF stack into one stable 3D input for the rest of the pipeline

### Stage 2. Template Prep

Input:
- target template assets
- hemisphere mode
- AP range derivation

Outputs:
- `template_half.nii.gz`
- `annotation_half.nii.gz`
- `template_prep.json`

Purpose:
- front-load hemisphere and atlas cropping choices instead of guessing per slice later

### Stage 3. ANTs Registration

Input:
- prepared template volume
- brain volume

Outputs:
- ANTs transform directory
- `ants_result.nii.gz`
- `registration_metrics.csv`
- `registration_summary.txt`

Purpose:
- solve the primary global 3D alignment problem

### Stage 4. Laplacian Refinement

Input:
- ANTs output

Outputs:
- deformation field artifact, such as `laplacian_deformation_field.npy`
- `final_registered.nii.gz`
- `refinement_metrics.csv`

Purpose:
- refine local contour fit and improve visual cleanliness near boundaries

### Stage 5. Truth Export

Input:
- final refined 3D registration result
- registered annotation volume

Outputs:
- `annotation_registered.nii.gz`
- per-slice `registered_label` files
- per-slice `overlay` files
- whole-brain QC panel exports

Purpose:
- derive all slice-level truth views from the final 3D truth volume

### Stage 6. Quantification

Input:
- 3D-derived registered labels
- detection outputs

Outputs:
- `cells_mapped.csv`
- `cell_counts_leaf.csv`
- `cell_counts_hierarchy.csv`
- `slice_registration_qc.csv`
- `volume_registration_qc.csv`

Purpose:
- make quantification depend on the 3D truth path rather than independent 2D registration decisions

---

## 7. UI and Progress Design

Whole-brain 3D registration will be slower than the current 2D-centered flow. The UI must make that acceptable through transparency.

### 7.1 Progress Model

The UI shows six primary stages:

1. Volume Build
2. Template Prep
3. ANTs Registration
4. Laplacian Refinement
5. Truth Export
6. Quantification

Each stage shows:

- stage name
- running/completed/failed state
- elapsed time
- latest log summary
- expected output artifact status

### 7.2 Fine-Grained Status for Slow Stages

For `ANTS Registration` and `Laplacian Refinement`, the UI also shows:

- current substage or transform step
- latest metric update if available
- last output timestamp
- heartbeat/progress signal so the run does not appear frozen

### 7.3 Result Surfaces

The whole-brain results area should expose three separate panels:

- `3D Registration Status`
- `3D QC Summary`
- `Slice Inspector`

This prevents users from confusing a slice preview with the actual source of truth.

---

## 8. QC Model

QC must be split into two layers.

### 8.1 Volume-Level QC

This is the authoritative registration evaluation layer.

Metrics should include, when available:

- NCC
- NMI
- SSIM
- Dice
- Hausdorff95

The UI and exported summaries should show:

- metrics after ANTs
- metrics after Laplacian refinement
- final chosen truth result

### 8.2 Slice-Level QC

Slice QC becomes an inspection layer, not an independent registration truth layer.

For each slice, QC should emphasize:

- raw versus exported overlay view
- tissue coverage
- visible boundary anomalies
- suspicious/flagged slices

Slice QC should no longer imply that each slice was independently auto-registered to produce the whole-brain truth.

---

## 9. Data and Output Contract Changes

### 9.1 Truth Ownership

The truth owner for whole-brain runs becomes the final 3D registered annotation volume.

### 9.2 Output Semantics

Existing outputs keep similar filenames where possible, but their meaning changes:

- `registered_slices/*_registered_label.*` means "exported from the final 3D truth"
- `registered_slices/*_overlay.*` means "rendered from raw slice plus exported 3D truth"
- `slice_registration_qc.csv` means "slice inspection and export diagnostics", not "independent slice registration success"

### 9.3 New Metadata

Whole-brain runs should also write:

- `volume_metadata.json`
- `template_prep.json`
- `volume_registration_qc.csv`
- stage/timing metadata for the full 3D run

---

## 10. Error Handling

Errors must stop at the stage boundary where failure happened and expose that boundary clearly.

Examples:

- bad TIFF stack or voxel metadata failure -> fail in `Volume Build`
- missing atlas assets or hemisphere prep failure -> fail in `Template Prep`
- ANTs dependency or transform failure -> fail in `ANTS Registration`
- refinement instability -> fail in `Laplacian Refinement`
- export mismatches between raw slices and registered volume -> fail in `Truth Export`
- mapping/counting mismatch against 3D truth -> fail in `Quantification`

The UI should never collapse these into one generic "registration failed" state.

---

## 11. Migration Strategy

### Phase 1

Add a new whole-brain 3D pipeline path that can run end-to-end and produce exported slice truth plus quantification artifacts.

### Phase 2

Switch whole-brain automatic mode to use that 3D pipeline by default.

### Phase 3

Demote legacy 2D whole-brain truth generation paths so they are no longer used by default for automatic whole-brain runs.

### Phase 4

Retain 2D tools only as auxiliary preview and manual correction surfaces.

---

## 12. Testing Strategy

Testing should cover behavior at three levels.

### 12.1 Stage Tests

- volume building from TIFF input
- template prep contract and hemisphere/AP handling
- ANTs invocation contract and artifact discovery
- Laplacian refinement output contract
- truth export from final 3D annotation
- quantification against exported labels

### 12.2 Integration Tests

- one representative whole-brain sample completes all six stages
- exported per-slice labels align with the 3D truth source
- mapping/count outputs use 3D-derived labels only
- stage progress and failure surfaces are emitted correctly

### 12.3 Regression Checks

- 2D auxiliary tools still function for preview/manual correction
- old UI areas do not falsely present 2D outputs as whole-brain truth
- slice QC does not reintroduce fake pass/fail scoring semantics

---

## 13. Acceptance Criteria

This design is considered implemented when:

1. Whole-brain automatic registration defaults to a native 3D Miki-style pipeline.
2. The final 3D registered annotation volume is the only truth source for exported slice labels, overlays, mapping, and counts.
3. The UI shows clear stage-based progress, including long-running ANTs/refinement stages.
4. QC exposes real 3D registration metrics and slice inspection outputs instead of fixed or fake pass scores.
5. Existing 2D tools still work, but only as auxiliary preview/manual-correction tools.

---

## 14. Open Implementation Constraint

This design assumes the product may become slower for whole-brain runs.

That tradeoff is accepted as long as:

- progress is visible
- logs are surfaced clearly
- stage failures are explicit
- result quality moves toward the `Sample\Miki` reference class

---

## 15. Recommendation

Proceed with a plan that makes the 3D whole-brain pipeline the product backbone instead of continuing to tune the current 2D slice-first path as if it can reach the same ceiling.

The 2D workflow should remain, but only in a subordinate role.
