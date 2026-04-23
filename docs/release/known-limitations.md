# Brainfast v1 Known Limitations

## Supported path

- Windows 10/11 primary support
- Python 3.10 or 3.11
- Default runtime: `scope=whole` + `whole_brain_backend=miki_3d` + `primary_model=cpsam`
- Cleared-tissue mouse samples using the Allen CCFv3 assets shipped in this repo
- Three learning loops are user-visible and independently closed:
  1. **Calibration learn** — `Save Calibration + Learn` tunes truth-export
     params; the next default whole-brain run consumes the learned
     `warp_params` / `fit_mode` / `edge_smooth_iter`.
  2. **Class-prior warm-start** — with ≥ `MIN_SAMPLES_FOR_APPLY` same-class
     samples, an empty same-class job auto-seeds on the 3D Liquify tab without
     a button click. Jobs with existing manual landmarks are never silently
     overwritten.
  3. **Cellpose retraining** — `Mask Editor` → `Save to Training Set` →
     `Cellpose Model Training` → `Apply Model`. Runs offline from the
     calibration and liquify loops.

## Known limitations

- 3D whole-brain registration still requires QC review before trusting
  region-level outputs. Slice-QC values (Dice, NCC) are the primary signal.
- Class-prior auto-apply only fires when the job starts empty. Jobs with
  any manual landmark pairs require the explicit `Warm-start from prior`
  button + confirmation so manual work is never clobbered.
- `ml_flip=false` remains provisional outside the tested left-hemisphere
  `right_flipped` sample class documented in
  `docs/superpowers/plans/ml-flip-matrix-summary.md`.
- GPU improves Cellpose performance but is not required for the
  fallback-free default path to start. The wizard ships with LoG fallback
  enabled by default so Cellpose OOM on low-VRAM machines does not
  dead-end the run.
- Mutable learned artifacts live under `<project>/outputs/state/` (or
  the path pointed to by `BRAINFAST_STATE_DIR`). Deleting that directory
  resets every learning loop.
- Atlas assets (`annotation_25.nii.gz` + structure graph) are fetched
  automatically by `Start_Brainfast.bat` via `download_atlas.py --ensure`.
  Users who launch `python project/frontend/server.py` directly must run
  the same command first; the amber banner at the top of the UI links to
  the recheck flow but does not yet trigger an in-UI download (planned for
  v0.5 via `POST /api/atlas/ensure`).
