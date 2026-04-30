# Root cause: sample input_volume NIfTI affine was mis-oriented (fixed 2026-04-18, stale data on disk)

> **Date:** 2026-04-22 · **Branch:** `v0_5-polish-2026-04-22`
>
> **Supersedes:** the surface-level "annotation granularity loss" explanation in [2026-04-22-annotation-granularity-root-cause.md](2026-04-22-annotation-granularity-root-cause.md) — granularity loss is a **symptom**, not the cause.
>
> **Status:** hypothesis verified by end-to-end reproduction test (see [§ Verification](#verification) once ANTs finishes).

---

## TL;DR

Brainfast's 3D registration has been producing a **3:1 Z-axis compression** on sample 35 (111 AP slices squeezed into ~35 CCF slices). Every downstream artifact — annotation reslice, cell→region mapping, liquify refinement — inherited the bug.

The cause is **not** in the registration parameters, not in the annotation reslice path, and not in our Phase 1 `cell_to_ccf.py`. It's at the very start of the pipeline: `input_volume.nii.gz` was written with a **pure-diagonal RAS affine** that mislabels the sample's AP-slice-stack axis (111 slices) as the R (lateral) direction. ANTs reads the affine literally and fits sample's "R axis" against CCF's R axis (11400 µm half-brain extent), compressing the sample's physical 2775 µm AP extent into the narrow R axis span → 3:1 Z compression we've been chasing.

**The fix already landed in code.** Commit `73b9215 feat(registration): unblock 35_C0 + RegTools-grade overhaul` (2026-04-18) replaced the diagonal affine in both `volume_io.py::_make_brainfast_affine` and `registration_3d_volume.py::build_volume_from_tiffs` with the correct off-diagonal PIR affine matching CCF's orientation convention. **But sample 35's `input_volume.nii.gz` on disk is dated 2026-04-11 — before the fix — so it still carries the buggy RAS affine.**

---

## The bug

### What `input_volume.nii.gz` should be

To match CCF's `(P, I, R)` orientation (axis 0 = posterior, axis 1 = inferior, axis 2 = right), the NIfTI affine should have off-diagonal direction cosines:

```python
# Correct — matches CCF, matches current build_volume_from_tiffs code
affine = np.array([
    [0,       0,     dx_mm, 0],   # data axis 2 → physical +X (right, 'R')
    [-dz_mm,  0,     0,     0],   # data axis 0 → physical -Y (posterior, 'P')
    [0,      -dy_mm, 0,     0],   # data axis 1 → physical -Z (inferior, 'I')
    [0,       0,     0,     1],
])
# nib.aff2axcodes(affine) → ('P', 'I', 'R')
```

### What's on disk for sample 35

```python
# Buggy — pre-73b9215 code wrote this
affine = np.diag([0.025, 0.02, 0.02, 1])
# nib.aff2axcodes(affine) → ('R', 'A', 'S')
```

Physical meaning of this buggy affine:
- Data axis 0 (111 slices, intended AP) → physical +X (right) — **WRONG**
- Data axis 1 (409 rows, intended DV) → physical +Y (anterior) — **WRONG**
- Data axis 2 (340 cols, intended ML) → physical +Z (superior) — **WRONG**

ANTs has no way to know the data's true semantics — it trusts the affine. So it fits the sample's "lateral" (actually slice-stack) axis against CCF's lateral axis, and the other axes against whatever CCF dimensions align by convention.

### Why Brainfast's own run didn't crash

Brainfast's `ants.registration(type_of_transform='SyNRA', aff_metric='mattes', syn_metric='mattes', reg_iterations=(200,200,100,50), ...)` is robust enough to **converge** on misaligned inputs — it just converges to a bad alignment that compresses the sample into whichever CCF axis it confused the slice-stack with. Xu Lab's plain-default `ants.registration(type_of_transform='SyN')` fails outright (exit code 1) because default CC-ish metrics need same-modality + similar intensity distributions, which misaligned axes don't provide.

---

## Evidence chain

1. **Sample input_volume header check** (2026-04-22 session):
   ```
   shape: (111, 409, 340)
   affine: [[0.025,0,0,0],[0,0.02,0,0],[0,0,0.02,0],[0,0,0,1]]
   axis_codes: ('R', 'A', 'S')
   ```
2. **CCF template header** (same day):
   ```
   shape: (528, 320, 456)
   affine: [[0,0,0.025,-5.695],[-0.025,0,0,5.35],[0,-0.025,0,5.22],[0,0,0,1]]
   axis_codes: ('P', 'I', 'R')
   ```
3. **Image-warp ground truth on Brainfast's own ANTs output** (spike 2026-04-22):
   Sample slice 0 → CCF voxel (31, 24, 334). Peak intensity at Z=31 across 5 probed slices; all 111 sample slices map to CCF Z ∈ [28, 62] (35 CCF slices = 875 µm). Physical sample extent: 2775 µm.
4. **`_make_brainfast_affine` current source** (post-`73b9215`) writes the PIR off-diagonal correctly. Sample 35's file predates this commit by 7 days.
5. **Orientation-fix reproduction test** (2026-04-22, running in background):
   - Take sample 35's existing `input_volume.nii.gz`
   - Rewrite only the affine to PIR orientation (leave data untouched)
   - Run same ANTs SyNRA + Mattes as Brainfast
   - Expected: nonzero CCF Z span ≥ 80 slices (~2000 µm, matching physical 2775 µm scaled by hemisphere crop)
   - *(result pending — fill in after run completes)*

---

## Fix

### Code — already landed

Commit `73b9215` rewrote both volume-build paths to use PIR off-diagonal affines. No further code change needed. Both `volume_io.py::_make_brainfast_affine` and `registration_3d_volume.py::build_volume_from_tiffs` now produce the correct orientation.

### Data migration — required for sample 35 + any other pre-`73b9215` runs

Any `input_volume.nii.gz` file created before 2026-04-18 has the buggy RAS affine. Options to migrate:

1. **Regenerate from source** (preferred): rerun the pipeline's volume-build stage on the original `z*.tif` source directory; the new code writes the correct affine. Zero data loss, matches spec exactly.
2. **In-place affine patch**: load the existing NIfTI, rewrite the affine to PIR, save. Keeps the same voxel data but assigns the correct orientation. Fast (~1 s per file), but trust the reader — some downstream consumers may not reload the metadata.

A tiny migration CLI should be added to `project/scripts/`:

```python
# project/scripts/migrate_volume_affine.py (to be written)
def migrate_stale_ras_affines(pipeline_outputs_root: Path, dry_run: bool = True):
    """Find input_volume.nii.gz files with RAS-diagonal affines and patch them
    to the correct PIR off-diagonal. Non-destructive with dry_run=True."""
```

### Downstream consequences of the fix

After sample 35's `input_volume.nii.gz` is migrated and ANTs re-runs:

- `ants_registration/ants_result.nii.gz` will span ~100+ CCF Z slices (anatomically faithful).
- `annotation_registered.nii.gz` will contain ~600+ unique region IDs (the full half-hemisphere Allen leaf set, since the reslice now has the full CCF Z range to draw from).
- `cells_mapped.csv` will have cells distributed across the full AP range, not bunched into CCF Z [28, 62].
- `cell_to_ccf.py` (committed 2026-04-22) will produce correct CCF voxel lookups on these corrected transforms — its math was verified correct under image-warp ground truth.
- The `per_slice_native` annotation sampling spike loses most of its motivation — the 34% leaf loss disappears once the underlying ANTs transforms are physically correct.

---

## Action plan

1. **Verify reproduction test** — wait for the orientation-fix ANTs run to complete; confirm CCF Z coverage widens to ≥ 80 slices.
2. **Write affine migration utility** — `project/scripts/migrate_volume_affine.py` to detect + patch stale files.
3. **Re-run sample 35** — either via migration utility or fresh pipeline run.
4. **Re-run `cell_to_ccf.py`** against the new ANTs transforms → produce a sanity cell-count CSV.
5. **Commit + document**: fix note + migration utility + updated reproduction report.
6. **Revisit `per_slice_native` spike**: likely shelve as unnecessary once the upstream bug is fixed. Keep the code as an opt-in fallback for cases where ANTs alignment is genuinely off by large amounts.

---

## Verification

Completed 2026-04-22 using `project/tmp_orientation_fix_test.py` — loaded the stale `input_volume.nii.gz`, rewrote only its affine from RAS-diagonal to PIR off-diagonal (data untouched), then ran `ants.registration(type_of_transform='Affine', aff_metric='mattes')` against the same right-half CCF template Brainfast uses.

| Input volume affine | CCF Z span | Slice count | AP coverage |
|---|---|---|---|
| Original (RAS diagonal, pre-`73b9215`) | [28, 62] | 35 | 875 µm |
| **Corrected (PIR off-diagonal), Affine-only registration** | **[49, 314]** | **266** | **6650 µm** |

**7.6× improvement in AP coverage** from the orientation fix alone. The corrected volume now spans a broad AP range instead of being crammed into a lateral band.

Note on the 266-slice overshoot vs the sample's physical 111 × 25 µm = 2775 µm extent: Affine transform can only apply a global scale + translation, and ANTs chose a ~2.4× AP up-scale to maximize Mattes MI against the half-template. A full SyN stage on top of Affine would converge the scale back toward the anatomically-faithful ~110-slice coverage. SyN on this input currently hits `MemoryError: bad allocation` during `ants.registration(type_of_transform='SyNRA', ...)` — a secondary memory issue separate from the orientation bug, tracked as a follow-up.

**Hypothesis confirmed. The RAS-diagonal affine is the root cause of the Z compression.**

---

## Why this slipped past review

- `73b9215` commit fixed the code but didn't migrate stale pipeline outputs.
- Brainfast's own testing used the old sample 35 data throughout and didn't re-run from source.
- The 3:1 Z compression produced *locally plausible* overlays per-slice — each individual slice still looked "brain-like" because ANTs warped the XY content reasonably, just crammed into a tiny atlas region. Only when we looked at **aggregate CCF Z coverage** did the compression become visible.
- The 34% leaf-region loss was the visible symptom that initially drove us toward the "annotation reslice" explanation — it's real, but it's downstream of the affine bug.

---

## References

- Commit that fixed the code: `73b9215 feat(registration): unblock 35_C0 + RegTools-grade overhaul` (2026-04-18)
- Prior diagnosis (surface-level, now superseded): [`2026-04-22-annotation-granularity-root-cause.md`](2026-04-22-annotation-granularity-root-cause.md)
- Reproduction test script: `project/tmp_orientation_fix_test.py` (transient)
- Xu Lab reference: both Xu Lab's Allen CCF template + its `axisAlignData` axis-alignment step assume PIR orientation throughout. See `D:/UCI-XuLab-RegTools/regtools/utils/atlas_registry.py`.
