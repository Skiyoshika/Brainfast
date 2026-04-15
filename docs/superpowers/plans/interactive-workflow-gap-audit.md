# Interactive Workflow Gap Audit

**Date:** 2026-04-10
**Context:** Brainfast stability rollout plan, Task 0 -- make current gaps explicit before further work.

---

## 1. 2D UI Manual Correction Exists

The frontend (`project/frontend/`) provides a working set of interactive correction tools for single-slice 2D registration:

- **Landmark correction panel:** the user can place, drag, and delete control-point pairs between the raw tissue image and the atlas overlay, then re-register with those constraints.
- **Liquify drag tool:** free-form warp of the overlay directly on the preview canvas. The user clicks and drags to deform the atlas boundary toward the tissue edge.
- **Save Calibration + Learn button:** packages the current (possibly corrected) overlay as a training pair (`Ori.tif` + `Label.tif` + `Show.png`), appends it to `train_data_set/`, and triggers `learn_from_trainset.py` to update registration parameters.

These tools are functional and have been used by at least one neurobiologist in a testing session (see `2026-04-07-ux-bugfix-neurobiologist-feedback.md`).

## 2. 3D Whole-Brain Final Truth Does Not Yet Consume Manual Corrections End-to-End

The `miki_3d` whole-brain backend (`project/scripts/run_3d_registration.py`, `registration_3d_volume.py`, `registration_3d_ants.py`) operates as a fully automatic pipeline:

1. Build a 3D input volume from the Z-stack.
2. Prepare a half-hemisphere template from the Allen atlas.
3. Run ANTs SyN registration in 3D.
4. Optional Laplacian PDE refinement.
5. Export per-slice registered labels and overlays.

At no point in this chain does the pipeline read or incorporate the 2D manual corrections (landmark edits, liquify warps, or calibration-learned parameters) that the user may have produced in the frontend. The calibration parameters produced by `learn_from_trainset.py` affect the 2D overlay preview path only.

**Consequence:** a user who carefully corrects registration in the 2D UI and then runs a whole-brain 3D pipeline will get results that ignore their corrections entirely. This is not a bug -- the two paths were built at different times -- but it is a product gap that must be closed before the workflow can be called end-to-end.

## 3. Cellpose-SAM Is Integrated but Has No Detector-Specific Manual QA Loop

Brainfast integrates Cellpose v4.1.1 with the `cpsam` (Cellpose-SAM) model via the `cellpose` Python package. The integration lives in `project/scripts/detect.py`:

- `_resolve_model_type("cpsam")` selects the SAM-augmented Cellpose model.
- `CellposeModel(pretrained_model="cpsam")` is used when the v4 API is available.
- Detection runs automatically during the pipeline; results feed into deduplication, region mapping, and aggregation.

**What is missing:** there is no UI surface for the user to:

- View individual detected cells overlaid on the tissue image and accept/reject them.
- Manually add or remove cell detections.
- Export corrected cell masks back into the pipeline for retraining or re-evaluation.
- Compare detector output quality across parameter changes.

The only detector-related feedback the user sees is the final aggregated count table and the cell chart. There is no way to distinguish "the detector found the wrong cells" from "the detector found the right cells but registration mapped them to the wrong brain region."

## 4. Save Calibration + Learn Tunes Atlas Overlay/Warp, Not Cellpose-SAM Detection

The `Save Calibration + Learn` button in the frontend triggers `learn_from_trainset.py`, which:

- Reads the training pairs in `train_data_set/`.
- Optimizes registration parameters (scale, rotation, translation, warp stiffness).
- Writes updated parameters to the calibration JSON.

This learning loop affects how the atlas overlay is placed and deformed onto the tissue. It does **not**:

- Retrain or fine-tune the Cellpose-SAM model.
- Adjust cell detection thresholds, diameter estimates, or flow parameters.
- Produce any feedback about detection quality.

**Consequence:** a user who clicks "Save Calibration + Learn" after seeing poor cell counts may believe they are improving detection, when they are actually only improving atlas placement. The two quality dimensions (registration accuracy and detection accuracy) are currently independent, and the UI does not make this distinction visible.

---

## Summary of Missing Product Pieces

| Gap | Status | Impact |
|-----|--------|--------|
| Stable single-sample interactive workflow (load, register, correct, detect, export) | Not yet completable end-to-end from the UI | Users cannot validate the tool on their own data |
| Registration-first QC gate (block downstream steps when overlay is poor) | Not implemented | Poor registration silently corrupts region-level counts |
| Cellpose-SAM manual review loop (view/edit individual detections) | Not implemented | No way to separate detector errors from registration errors |
| 3D path consuming 2D manual corrections | Not connected | Manual effort in the UI is lost when switching to whole-brain mode |

These gaps must be acknowledged and prioritized before expanding sample coverage or declaring the pipeline stable.
