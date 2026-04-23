# Annotation Granularity Root Cause + Fix Plan

> **Date:** 2026-04-22 · **Branch:** `v0_5-polish-2026-04-22` · **Author:** Claude Code session (triggered by user PPT review showing Xu Lab's per-slice registration produces visibly richer annotation overlay than Brainfast 3D whole-brain output).

---

## TL;DR

Brainfast's 3D whole-brain registration **reslices the Allen annotation volume into the sample's Z-downsampled shape**, dropping **34% of leaf regions that are actually present in the sampled AP range** before liquify even runs. The liquify learning loop therefore refines an already-coarsened surface and can never recover what the reslice step threw away. The UCI Xu Lab reference tool does not have this problem because it runs **2D per-slice registration** and pulls the full-resolution annotation slice directly from CCF space.

**Root cause:** annotation reslice in `whole_brain_3d.py` uses a 5× Z downsampling (528 CCF slices @25 µm → 111 sample slices @125 µm) with nearest-neighbor interpolation. Thin leaf regions that span only 1–4 CCF slices are statistically likely to fall into the gaps and be dropped.

**Fix (selected + spiked):** keep 3D ANTs SyN for volume-level anatomical alignment, but change annotation sampling from "warp whole volume into sample Z-space" to "for each sample slice, pull the full-resolution 2D annotation slice from CCF space at the corresponding atlas-Z, and let the downstream `render_overlay(prewarped_label=False)` tissue-guided 2D warp supply per-slice in-plane alignment." This preserves the full leaf granularity of the Allen atlas within the sample's AP coverage.

**Spike result (2026-04-22, on Sample 35 `35_C0_demo_run2` ANTs artifacts):** `per_slice_native` recovers **273/273 leaf region IDs** in the sampled atlas-Z range — the theoretical maximum — vs `3d_reslice`'s 181/273 (66%). The new mode is a strict superset of what the old mode produced (zero IDs that only the old mode had). Downstream overlay alignment pending E2E verification at time of writing.

---

## Evidence

### A. Atlas source is fine — Brainfast and Xu Lab use the same content

| File | Shape | Unique IDs | Non-zero voxels | Max ID |
|---|---|---|---|---|
| Brainfast `project/annotation_25.nii.gz` | (528, 320, 456) | 672 | 32,387,385 | 614,454,277 |
| UCI-ALLEN `CCF_DATA/annotation_25.nii.gz` (Xu Lab family) | (528, 320, 456) | 672 | 32,387,385 | 1,327 |

Voxel-by-voxel test (`project/annotation_25.nii.gz` > 0) **matches exactly** against the UCI-ALLEN copy. Every Brainfast region ID has a 1:1 mapping to a UCI-ALLEN Allen CCF region ID — the BBP-encoded large IDs (6×10⁸ range) are just an alternate integer encoding of the same Allen regions. Brainfast's `configs/allen_structure_tree.json` has 4669 entries and all 672 non-zero annotation IDs resolve.

**Conclusion:** the source atlas is not the problem.

### B. 34% of sample-range leaf regions disappear during ANTs 3D registration

Measured on the Sample 35 default whole-brain run in `project/outputs/35_C0_demo_run2/`:

| Stage | File | Shape | Unique IDs | Loss vs sample-range source |
|---|---|---|---|---|
| CCF full atlas | `project/annotation_25.nii.gz` | (528, 320, 456) | 672 | — (whole brain) |
| CCF half-atlas after Laplacian refine | `laplacian_refinement/annotation_refined.nii.gz` | (528, 320, 228) | 619 | — (whole brain, half-hemisphere) |
| CCF half-atlas **restricted to the sampled AP range** (atlas Z [92..251]) | — (computed in-place) | (160, 320, 228) | **273** | baseline for this sample |
| Post ANTs 3D warp to sample space | `ants_registration/annotation_registered.nii.gz` | (**111**, 409, 340) | **181** | **−34%** |
| Post 3D Liquify refinement | `annotation_refined_liquify3d.nii.gz` | (111, 409, 340) | 181 | −1 from pre-liquify (liquify itself is fine) |
| Post `per_slice_native` spike | `ants_registration/annotation_registered_per_slice_test.nii.gz` | (111, 320, 228) | **273** | **0% loss — theoretical maximum** |

Key observations:
- The 672 → 182 comparison in earlier notes was technically true but conflated "whole-brain CCF" with "atlas range this sample actually covers." The sample only spans atlas AP [92..251], where CCF itself contains 273 unique regions. The production `3d_reslice` path drops 92 of those 273 (34%) during ANTs reslice.
- **The spike-produced `per_slice_native` output contains every ID that `3d_reslice` has, plus 92 more** — zero IDs are unique to `3d_reslice`. The new mode is a strict superset on leaf-ID content.
- Liquify is not the culprit: the drop happens in the ANTs reslice step, before liquify runs.

### C. Why the drop: Z-downsampling + nearest-neighbor

The sample has 111 slices at ~125 µm spacing; Allen CCF is 528 slices at 25 µm spacing. The 3D registration reslices the annotation from (528, 320, 456) to (111, 409, 340) — a **~5× Z compression**. `whole_brain_3d.py` correctly uses `interpolator="nearestNeighbor"` (the only sane choice for discrete labels), but nearest-neighbor does not "merge" neighboring slices; it picks one source voxel per target voxel. A leaf region that spans only 1–4 CCF slices has a high probability of being skipped when the target slice lands between its source range.

### D. Xu Lab avoids the problem by not doing 3D reslice

From `D:/UCI-XuLab-RegTools/regtools/registration/registration_2d.py` and their slide (`C5_WHC_P2_#194 → CCF Slice 355 [Laplacian]`): each real slice is independently matched to one CCF slice index, then a 2D Elastix (Rigid + BSpline) or ANTs SyN registration is run in-plane only. The annotation is pulled directly as `annotation_volume[:, :, ccf_slice_index]` — a full (320, 456) slice with the complete leaf granularity — and 2D-warped into real-image coordinates via the inverse of that 2D transform. Z is never downsampled.

**Trade-off they accept:** no enforced 3D anatomical consistency across adjacent slices. For histology where adjacent slices are 50–125 µm apart and mounting already breaks strict Z continuity, this is a reasonable trade-off.

### E. Consequence for the liquify learning loop

Liquify 3D refinement is a sparse-landmark Laplacian warp applied to the **already-coarsened 182-region annotation**. It can improve landmark fit on the regions that survived but cannot re-introduce lost leaf regions. Every downstream metric — region-level cell counts, per-region Dice, user-visible overlay richness — inherits the 73% granularity loss.

User's quote that triggered this investigation:

> 我们这个工具是为了解决这个内部轮廓没有那么匹配这件事才开发的这个液化校准学习功能，但我们连他们的基础区域划分都没做到啊

Accurate diagnosis. Liquify is solving the right problem (internal-contour fit) on the wrong substrate (an atlas that has already been smoothed into parent-like blocks).

---

## Fix options (ranked)

### Fix 1 — Per-slice annotation sampling (SELECTED)

Keep 3D ANTs SyN for volume-level anatomical alignment. Change only how annotation overlay / registered-label rasters are produced: for each sample slice *i*, find the corresponding CCF Z via the forward ANTs transform applied to the slice centroid, pull `annotation_ccf[:, :, ccf_z_i]` directly, and warp that 2D slice into the sample slice's in-plane coordinates using the 2D component of the ANTs displacement.

**Why this is right:**
- Preserves full CCF Y×X leaf granularity (no Z drop).
- Reuses the 3D registration result — no new registration engine needed.
- Localized change: only the annotation-rasterization path in `whole_brain_3d.py` and its downstream consumers (truth-export, liquify source annotation).
- Additive: add as `annotation_sampling_mode = "per_slice_2d"` config switch with existing `"3d_reslice"` as default; flip default once validated.

**Risk:**
- If ANTs 3D SyN produced a non-planar deformation (annotation slice from sample-Z-i actually corresponds to a warped CCF surface rather than a flat CCF plane), picking a single CCF Z for the whole sample slice is an approximation. In practice for histology where sample Z-step ≫ CCF Z-step, this approximation is small.
- Need to pick which CCF Z to sample per sample slice: use ANTs forward transform of the slice centroid, or the Z value that minimizes in-plane registration residual. Start with the centroid choice; measure the residual delta.

### Fix 2 — Majority-vote Z-neighborhood reslice

Replace `nearestNeighbor` with a custom sampler that takes the modal label over a 5×1×1 source-Z neighborhood. Preserves thin regions better but still has a quantization floor and is more work to implement inside ANTs. Rejected in favor of Fix 1.

### Fix 3 — Multi-level annotation cache

Cache {leaf (672) / summary (300) / major (~40)} annotations and switch by zoom. Doesn't solve the core problem, only papers over it. Rejected.

### Fix 4 — 2D per-slice registration backend

Replicate Xu Lab's `whole_brain_backend=2d_per_slice` as an alternative. Largest change, new registration engine, new regression surface. Keep as a future option but not the first move.

---

## Existing code already has most of Fix 1 — just disabled

While locating the warp site (Step 1 below), the existing `project/scripts/whole_brain_3d.py:311` `_direct_z_mapping_fallback` (Strategy 3) turned out to already implement the "keep CCF native Y×X resolution" idea. Its docstring:

> "The annotation is kept at its NATIVE atlas resolution (not downsampled to the tiny input-volume grid) so that brain-region detail is preserved."

It produces a `(num_input_z, atlas_Y, atlas_X)` = `(111, 320, 456)` volume for Sample 35 — full CCF Y×X with Z sliced by offset lookup. **But it runs only as a fallback** when Strategies 1 & 2 both fail a ≥10% coverage threshold. Since Strategies 1 & 2 pass the coverage gate even while dropping 73% of leaf regions (coverage counts non-zero voxels, not unique region IDs), Strategy 3 never triggers in production.

Strategy 3 output lacks in-plane 2D alignment (no ANTs XY warp is applied to each slice, only a Z lookup). Downstream `project/scripts/truth_export_3d.py:113` hard-codes `prewarped_label=True` when calling `render_overlay`, which skips the tissue-guided 2D warp (`overlay_render.py:3481` branch). Feeding Strategy 3's native-resolution annotation into `prewarped_label=False` lets `_tissue_guided_warp` (uses `order=0` nearest-neighbor for labels at `overlay_render.py:816`) fix the in-plane alignment per slice without destroying region IDs.

Conclusion: Fix 1 is a small wiring change, not a new algorithm. Route upstream to Strategy 3 and downstream to `prewarped_label=False` behind a single config switch.

## Impl plan (Fix 1 — minimal, reuses existing code)

### Step 1 — Add config

`whole_brain_3d` section in `run_config*.json` (and the default in `whole_brain_3d.py`):

```json
{
  "whole_brain_3d": {
    "annotation_sampling_mode": "3d_reslice"  // or "per_slice_native"
  }
}
```

Default stays `"3d_reslice"` until Step 4 validates; existing tests remain untouched.

### Step 2 — Route upstream: force Strategy 3 when mode == "per_slice_native"

Thread `annotation_sampling_mode` into `_warp_annotation_volume_to_input_space`. When `per_slice_native`, skip Strategies 1 & 2 and jump directly to `_direct_z_mapping_fallback`. Stamp the chosen mode into a sidecar JSON so downstream can read it back.

### Step 3 — Route downstream: flip `prewarped_label`

Thread the mode into `export_registered_truth_slices`. When `per_slice_native`, call `render_overlay` with `prewarped_label=False` so `_tissue_guided_warp` provides per-slice in-plane alignment on the CCF-native-resolution annotation.

### Step 4 — Unit test with synthetic atlas

`project/tests/unit/test_annotation_sampling_modes.py`:
- Build a synthetic 10×10×10 source annotation with distinct labels per Z slice
- Target shape: (2, 10, 10) — 5× Z compression
- Identity or near-identity ANTs transform (use `_direct_z_mapping_fallback` directly — no ANTs runtime dependency)
- Assert: `per_slice_native` mode preserves ≥ 8 unique labels; current `3d_reslice` mode (via mocked reslice) loses ≥ 5

### Step 5 — Sample 35 E2E validation

```bash
python project/scripts/main.py --config project/configs/run_config_35.json \
  --run-real-input project/data/35_C0_demo \
  --output-name 35_C0_per_slice_test
```
with `run_config_35.json` patched to `"annotation_sampling_mode": "per_slice_native"`.

Expected: `ants_registration/annotation_registered.nii.gz` unique IDs jump from 182 → ≥ 500. Overlay PNGs in `truth_export/` show fine-grained region colors comparable to Xu Lab's "Inverse-Mapped Annotation" output.

### Step 6 — Flip default if Step 5 passes

Change the `whole_brain_3d.py` default from `"3d_reslice"` to `"per_slice_native"`. Keep both modes reachable via config.

### Step 7 — Update docs

- `docs/user_guide.md`: note the two sampling modes and which to pick
- `docs/release/known-limitations.md`: remove any wording implying 3D reslice Z-loss is intrinsic
- `docs/BRAINFAST_DEMO_AND_BATCH_RUNBOOK.md`: note the overlay quality improvement after re-run

---

## Status / next step

- **Step 1 in progress:** reading the warp site end-to-end.
- **P2 quick wins + handoff refresh** (from earlier today) stay uncommitted on the same branch; will commit after Fix 1 spike lands or separately if Fix 1 takes longer than half a day.

---

## Open questions (decide before merging)

1. Per-slice CCF Z selection: centroid-forward-transform vs per-pixel Z minimization. Start with centroid; measure residual.
2. If ANTs produced large out-of-plane deformation (e.g. 10°+ atlas tilt), per-slice sampling at a single CCF Z will still drift. Need a diagnostic that reports per-slice out-of-plane deformation magnitude and falls back to the 3D reslice for slices where it exceeds some threshold.
3. Does truth-export / liquify-3d finalize consume `annotation_registered.nii.gz` or the raw 3D warp result? Need to confirm the consumer chain before changing the producer.
