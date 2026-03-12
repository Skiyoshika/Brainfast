# IdleBrainUI v0.3.0 (Desktop Beta)

## What is included
- Desktop launcher: `dist/IdleBrainUI.exe`
- Discord-style UI with backend bridge
- Path validation + run/cancel/log/status APIs
- Multi-channel mode: red/green/farred
- CSV export + run history + QC preview

## Quick test checklist
1. Double-click `IdleBrainUI.exe`
2. Confirm browser opens `http://127.0.0.1:8787`
3. Fill input/atlas/structure paths
4. Run Pipeline and verify:
   - `outputs/cells_detected.csv`
   - `outputs/cells_dedup.csv`
   - `outputs/cell_counts_leaf.csv`
   - `outputs/cell_counts_hierarchy.csv`
   - `outputs/slice_qc.csv`
5. Test Cancel button during run
6. Test Run All Channels

## Known limitations
- Desktop exit flow is basic (no tray yet)
- Progress is log-derived (not true per-step engine feedback)

