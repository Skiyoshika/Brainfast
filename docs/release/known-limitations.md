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
