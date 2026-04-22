# Brainfast Registration Quality — Honest Investigation Report

**Date:** 2026-04-16
**Context:** User asked to run Brainfast end-to-end on `Sample/ChATe27/35_*_C0.tif + C1.tif` and produce a per-region dTom+ cell-count bar chart matching an n=3 reference figure. During the run, multiple code-level bugs surfaced that prevent Brainfast from reaching Miki-level registration quality.

## Reference gold standard

Miki's prior registration of the same sample (`Sample/Miki/brain_registration/20260313_133224_miki_reg_ch0/registered_output/`) is the quality baseline:
- Moving volume: `(648, 1736, 1459)` at `(0.02476, 0.005, 0.005)` mm (full xy detail)
- SyN + Laplacian refinement enabled
- Final metrics (after Laplacian):
  - NCC: **0.691**
  - NMI: 1.137
  - SSIM: 0.319
  - Dice: 0.744
  - PSNR: 13.04
  - MSE: 0.050

## Summary of findings

Five discrete issues block reaching Miki-level registration on the same raw input.

| # | Type | Location | Severity | Status |
|---|---|---|---|---|
| 1 | Pipeline bug | [`whole_brain_3d.py:491`](../../../project/scripts/whole_brain_3d.py#L491) | High (registration silently misused full atlas) | ✅ **Fixed** (2-line patch + unit test) |
| 2 | CLI bug | [`main.py:1257`](../../../project/scripts/main.py#L1257) | Medium (outputs written to wrong dir) | ⚠️ Not yet fixed |
| 3 | Hardcoded cap | [`registration_3d_volume.py:35`](../../../project/scripts/registration_3d_volume.py#L35) | **High** (forces 4× xy downsample, blocks Miki parity) | ⚠️ Not yet addressed |
| 4 | OOM crash | [`laplacian_refine_3d.py:252`](../../../project/scripts/laplacian_refine_3d.py#L252) | **High** (Laplacian can't handle full-density volumes) | ⚠️ Not yet addressed |
| 5 | Misleading config | [`configs/run_config_35.json`](../../../project/configs/run_config_35.json) `slice_spacing_um=25.0` | Medium (truth for 646 raw, wrong for 111 demo) | Clarified, not fixed |

## Bug #1 — AP range regex fails on merged filenames (FIXED)

**Code path**: `whole_brain_3d.py` AP auto-compute block reads slice filenames and extracts `z<digits>` via regex, then computes `ap_start/ap_end` for the template crop.

```python
_s_dir = Path(merged_slice_paths[0]).parent if merged_slice_paths else Path(input_dir)
_s_glob = "*.tif" if merged_slice_paths else str(input_cfg.get("slice_glob", "z*.tif"))
_s_paths = sorted(_s_dir.glob(_s_glob))
if _s_paths and reg_cfg.get("atlas_z_from_filename", False):
    _z_nums = []
    for sp in _s_paths:
        m = _re.search(r"z(\d+)", sp.stem)   # ← merged_####.tif has no "z<digits>"
        if m: _z_nums.append(int(m.group(1)))
```

**Failure mode**: `merged_slice_paths` points to `tmp_merged/merged_0000.tif…`. Regex never matches → `_z_nums` empty → `ap_start`/`ap_end` stay at default `[0, 528]` → ANTs stretches 2.8mm brain chunk across whole 12mm atlas.

**Consequence on demo run** (`outputs/_qc_before_ap_fix/35_C0_before_registration_metrics.csv`):
- NCC=0.12 vs Miki's 0.66 (5.7× worse)
- Dice=0.30 vs Miki's 0.75 (2.5× worse)
- Slice 50 overlay mis-labeled with MOB/AON (olfactory, anterior) on a mid-brain slice

**Fix**: [`whole_brain_3d.py:485-489`](../../../project/scripts/whole_brain_3d.py#L485) use `input_dir` unconditionally (source dir preserves z-number).

**Test**: [`test_whole_brain_3d.py:412`](../../../project/tests/unit/test_whole_brain_3d.py#L412) `test_ap_auto_compute_uses_source_z_numbers_when_merged_names_lack_z`.

**After fix (demo, 111 slices, before discovering bug #3)**:
- Dice 0.30 → 0.99 (silhouette only — see below)
- NCC 0.12 → -0.06 (still bad; Dice improvement was surface-only)
- Slice 50 overlay now correctly shows CA1/CA3/VPM/VAL/STR (mid-brain)

## Bug #2 — `--output-name` never wired to `run_real_input()`

[`main.py:1254-1257`](../../../project/scripts/main.py#L1254):
```python
if args.run_real_input:
    env_output_dir = os.environ.get("BRAINCOUNT_OUTPUT_DIR", "").strip()
    output_override = Path(env_output_dir) if env_output_dir else None
    run_real_input(cfg, Path(args.run_real_input), outputs_dir=output_override)
    #                                              ^^^^^^^^^^^^^^^^^^^^^^^
    # args.output_name NOT passed through.
```

`--output-name 35_C0_full` only affects `configure_logging()` at lines 1197-1201. The pipeline falls back to `project_root / "outputs"` (flat root), writing `tmp_channel/`, `ants_registration/`, `truth_export/` etc. directly into the shared outputs parent.

**Evidence**: first rerun of `35_C0_full` wrote 471MB of `tmp_channel/` into `project/outputs/tmp_channel/` instead of `project/outputs/35_C0_full/tmp_channel/`.

**Workaround**: set `BRAINCOUNT_OUTPUT_DIR` env var (used for the full-density run).

**Fix**: pass `output_override = Path(env_output_dir) if env_output_dir else (_outputs_dir if args.output_name else None)` — about 3 lines.

## Bug #3 — Hardcoded 4× xy downsample cap

[`registration_3d_volume.py:31-35`](../../../project/scripts/registration_3d_volume.py#L31):
```python
# Miki's pipeline used 5 µm pixels; we cap at 4x to stay within ANTs
# memory limits while keeping ≤ 3 µm resolution for sub-micron inputs.
raw_factor = target_um / pixel_um_xy
downsample_factor = max(1, min(4, round(raw_factor)))
```

`target_um=25` default, `pixel_um_xy=5.0` for our sample → `raw_factor=5`, clamped to `4`.

**Impact**: `build_volume_from_tiffs` writes NIfTI at 20µm xy (shape 646×409×340) instead of Miki's 5µm xy (shape 648×1736×1459). **Moving volume is 18× smaller than Miki's.** ANTs SyN has 18× less intensity detail to lock onto for the MI/CC metric, so the similarity metric never converges to a good optimum.

**Evidence**: Full-density run (646 slices, same raw as Miki but 4× downsampled xy) after Bug #1 fix:
- NCC = **-0.16** (vs Miki 0.66)
- SSIM = 0.34 (matches Miki's 0.30 — SSIM is less resolution-dependent)
- Dice = 0.97 (silhouette — see note below)

**Why Dice is misleading**: Dice on the binary tissue mask only measures outer-contour overlap. Two registrations can both produce a brain-shaped blob with high Dice while internal structure alignment is completely different. NCC and NMI are the honest internal-structure metrics.

**Fix approaches** (ranked):
1. Remove the `min(4, …)` cap → keep `round(raw_factor)` but verify RAM head-room. For 5µm→25µm that's 5× downsample, similar to Miki's input (Miki feeds 5µm xy and lets ANTs handle via multi-resolution pyramid).
2. Make the cap configurable via `registration.xy_downsample_cap` so ops can tune per machine.
3. Better: feed ANTs at native resolution and let its multi-resolution shrink factors do the work internally.

## Bug #4 — Laplacian refinement OOM on full-density volumes

[`laplacian_refine_3d.py:252`](../../../project/scripts/laplacian_refine_3d.py#L252) `_build_laplacian_3d_fast` constructs a CSR sparse Laplacian over the entire registered volume:

```
File "laplacian_refine_3d.py:252" in _build_laplacian_3d_fast
    return csr_matrix((all_vals, (all_rows, all_cols)), shape=(N, N))
numpy.core._exceptions._ArrayMemoryError: Unable to allocate 1.00 GiB 
    for an array with shape (268935552,) and data type int32
```

For 646×409×340 = 89.7M voxels, the 3D stencil produces ~269M non-zeros × 3 arrays × 4 bytes = 3.2GB contiguous per-array allocation. Windows could not satisfy even one 1GB contiguous request on a 33GB machine.

Miki's Laplacian must use a different implementation (chunked, or pre-downsampled) because Miki ran on a larger volume without crashing.

**Fix approaches**:
1. Build the Laplacian in chunks (sum incremental COO blocks) rather than pre-allocating full index arrays.
2. Downsample the registered volume before Laplacian refinement (e.g. 2× further), apply field, upsample field back.
3. Skip Laplacian entirely for Windows consumer hardware (minor NCC impact: Miki gained only 0.035 from Laplacian).

## Bug #5 (config) — `slice_spacing_um: 25.0` is misleading for demo

`configs/run_config_35.json` declares:
```json
"slice_dir": "data/35_C0_demo",
"slice_spacing_um": 25.0
```

But `data/35_C0_demo/` was built by `extract_zstack.py --every_n 5` from a raw TIFF with real z-step **24.765 µm** (ImageDescription: `spacing=24.764999`). Every-5th sampling → actual demo z-spacing is **5 × 24.765 = 123.8 µm**, not 25 µm.

**Impact**: ANTs voxel grid for demo is off by a factor of ~5× along z. With wrong spacing the affine mapping between voxel index and atlas space is distorted and SyN operates on a skewed coordinate system. This makes internal structure registration harder even without the other bugs.

**Fix**: either
- Make `extract_zstack.py` write a manifest with actual effective z-spacing and have main.py read it, or
- Document in `run_config_35.json` that the demo config assumes `every_n=1` extraction and the demo mode is for quick smoke tests only.

## Evidence artifacts

### Archived ("before AP fix" baseline)
- `project/outputs/_qc_before_ap_fix/35_C0_before_registration_metrics.csv`
- `project/outputs/_qc_before_ap_fix/35_C0_before_slice_qc.csv`
- `project/outputs/_qc_before_ap_fix/35_C0_before_volume_qc.csv`
- `project/outputs/_qc_before_ap_fix/35_C0_before_slice_0050_overlay.png` — clearly shows olfactory labels on mid-brain slice (WRONG)
- `project/outputs/_qc_before_ap_fix/ap_fix_comparison.png` — side-by-side before/after visual
- `project/outputs/_qc_before_ap_fix/ap_sweep_panel.png` — 6 slices across AP range after fix

### Post AP fix + 4× downsample (after Bug #1 only)
Earlier run (`35_C0_full`, demo 111 slices) — killed during truth-export; contaminated outputs cleaned.

### Full-density run (646 slices, Bug #1 fixed, Bug #3 still active)
`project/outputs/35_C0_full_density/`:
- `ants_registration/registration_metrics.csv`: NCC=-0.156, NMI=1.015, SSIM=0.341, Dice=0.973, PSNR=8.38
- `ants_registration/ants_result.nii.gz` — ANTs-registered moving volume (can be visualized)
- `laplacian_refinement/` — never wrote, crashed with memory error
- `truth_export/slice_0000_overlay.png` — single exported slice (pipeline died before per-slice export finished)
- `pipeline_progress.json` shows stageIndex=4 (Laplacian) when crash hit

### Fix + test committed-ready
- `project/scripts/whole_brain_3d.py` line 485-489 patch (AP regex moved to input_dir)
- `project/tests/unit/test_whole_brain_3d.py::test_ap_auto_compute_uses_source_z_numbers_when_merged_names_lack_z` — 9/9 whole_brain_3d tests + 288/288 unit suite + ruff clean

## Path forward (recommendation)

Order of work for next session, if goal is real Miki-level registration:

1. **Commit AP regex fix + unit test** (locks in verified gain).
2. **Wire `--output-name` through `run_real_input()`** (30 min, trivial but prevents future confusion).
3. **Make downsample cap configurable** and try `2×` first (should be ~360 MB volume, ANTs still tractable, ~2× more xy detail).
4. **Rewrite Laplacian sparse construction to chunk** — build COO blocks in a loop and `vstack` them at the end to avoid the 1GB contiguous allocation.
5. **Re-run 646-slice pipeline** at (`xy_cap=2` or `1`, `Laplacian ON`), verify NCC ≥ 0.5 against Miki 0.66 as success gate.
6. Only after step 5 succeeds, move on to Quantification → per-region aggregation → target bar chart.

## What NOT to do

- Don't report Dice alone as "registration success". It's silhouette, not internal.
- Don't skip Laplacian and call the result "Miki-equivalent" — Miki measures after Laplacian.
- Don't re-run Quantification on the current (Bug #3 active) ants_result — per-region counts will be geometrically misassigned and reinforce a false sense of success.
