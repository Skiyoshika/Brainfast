# README screenshot checklist

These are the screenshots the root `README.md` references. Capture each at **1400×900**, light theme, against the demo sample (`configs/run_config_35.json`). Save as `.png` into this folder.

## Required

| File | What to show | Capture steps |
|------|---|---|
| `hero-workflow.png` | The Registration Workflow tab with Step 1 filled in for the demo sample, showing the guided tour dismissed and the step indicator bar at top | Start `Start_Brainfast.bat` → Load `run_config_35.json` → screenshot full viewport |
| `atlas-banner.png` | The "Atlas file missing" amber banner at the top (requires the atlas file to be temporarily moved out of `project/`) | Rename `annotation_25.nii.gz` → reload → screenshot |
| `results-chart.png` | Results tab with the region-count bar/pie chart rendered after a successful run | Run the demo sample → go to Results → screenshot |
| `qc-panel.png` | QC tab showing the 12-slice whole-brain panel + reg-stats bar | After a full run → QC tab → screenshot |
| `liquify-3d.png` | 3D Liquify tab with class-prior heatmap visible | Load a post-run job → 3D Liquify → screenshot |

## Optional / v0.5

| File | What to show |
|------|---|
| `demo.gif` | 30 s screen recording of the one-click flow from config to SSIM "Excellent" result |
| `dual-channel.png` | Dual-channel overlay (560 nm + 640 nm pseudocolor) after registration |
| `model-training.png` | Model Training tab mid-training with loss curve |

## Tooling hint

Use any screenshot tool that produces PNG at the viewport size — Windows `Snipping Tool`, macOS `Cmd+Shift+4`, or `playwright codegen` for reproducibility.

For the animated demo (`demo.gif`), [ScreenToGif](https://www.screentogif.com/) on Windows or [LICEcap](https://www.cockos.com/licecap/) cross-platform work well; keep it under 5 MB.
