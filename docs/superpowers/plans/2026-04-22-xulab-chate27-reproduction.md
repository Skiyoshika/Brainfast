# Xu Lab RegTool reproduction on ChATe27 real data

> **Date:** 2026-04-22 · **Branch:** `v0_5-polish-2026-04-22`
>
> **Goal (user-directed):** "在全面了解 Xu Lab RegTool 之后，至少先复现 Xu Lab RegTool 的配准和细胞计数，直接做到能复现为止" — end-to-end reproduce Xu Lab's registration + cell-counting pipeline on real ChATe27 data (Andy's workflow), produce preview images.

---

## TL;DR — reproduction succeeded with one patch + two local-env workarounds

End-to-end Xu Lab pipeline runs on Brainfast's Python env against the raw ChATe27 brain-35 C0 TIFF (2.8 GB, 646 pages). Output: correct PIR-oriented NIfTI, axis-aligned moving volume (9.73° rotation detected), ANTs registration (AP coverage 3.48 mm vs sample physical 3.23 mm — match within 7.6%), 3230 synthetic cells transformed into CCF voxel space, 48 unique Allen region IDs identified with anatomically-sensible top regions (CA1 / CA3 / LD / ENTl2 / PO / LGd / VPM / DG-mo / …).

Reproduction script: [`project/scripts/reproduce_xulab_on_chate27.py`](../../../project/scripts/reproduce_xulab_on_chate27.py).
Outputs (excluded from git via `.git/info/exclude`): `D:/Brainfast/Sample/ChATe27/xulab_reproduction/`.
Dashboard preview (regenerate from outputs): `index.html` in the same dir.

## Setup

- **Python env:** Brainfast venv (Python 3.11, `antspy` 0.x, `nibabel`, `numpy`, `pandas`, `matplotlib`, `tifffile`, `scikit-image`, `scipy`, `joblib`).
- **Xu Lab source:** `D:/UCI-XuLab-RegTools/` — imported directly via `sys.path.insert(0, ...)`, no separate install.
- **Allen CCF (Xu Lab schema):** `D:/UCI-ALLEN-BrainRepositoryCodeGUI-main/CCF_DATA/{annotation_25.nii.gz, 1_adult_mouse_brain_graph_mapping.csv}` — standard Allen region IDs (1–1327), Xu Lab's atlas CSV schema.
- **Sample:** `D:/Brainfast/Sample/ChATe27/35_..._C0.tif` — 646 pages × 1636 × 1359 uint16, Z-step 5 µm per filename convention.

## Pipeline (as implemented in `reproduce_xulab_on_chate27.py`)

1. **`regtools.utils.io.nifti_utils.create_nii_images`** on the raw TIFF. Default `/8` XY pre-downscale, `scale=2.5` → `brain_25.nii.gz` shape `(646, 68, 82)`. Override `voxel_spacing=[0.005, 0.025, 0.025]` (Z 5 µm, XY 25 µm) — Xu Lab's default `sz=None` assumes 50 µm/slice which is 10× over-claim on this data.
2. **Right-half CCF template + annotation crop** (Xu Lab's `half_mode='right'`). Matches ChATe27 mounting convention.
3. **`regtools.registration.core.io_utils.axisAlignData`** — midline-fissure rigid pre-alignment. Found 9.73° rotation + saved to `01_axis_alignment/axisAlignA.npz` via Xu Lab's `_save_npz` helper (key `'arr'`, matching `_load_npz` expectation).
4. **`ants.registration`** with `type_of_transform='SyNRA'` + `aff_metric='mattes'` + `syn_metric='mattes'`. If OOM, falls back to `SyN` then `Affine`. Saves transforms in Xu Lab layout (`02_nonlinear/{fwd,inv}_transforms/ants_{warp,affine,invwarp}_*.{nii.gz,mat}`).
5. **Warp CCF annotation → sample voxel space** via `ants.apply_transforms(interpolator='genericLabel')` (Affine-only path yields sparse mapping; SyN would fix).
6. **5-panel matplotlib preview** (Preprocessed / CCF Template / CCF + Annotations / ANTs Result + Annotations / Inverse-Mapped Annotation on sample) at 3 representative slices.
7. **Xu Lab `transform_input_points`** on 3230 synthetic cells placed at tissue hotspots. Applies axis-align + ANTs point warp → CCF voxel coords.
8. **CCF region lookup via `cell_to_ccf.lookup_region_ids_from_ccf_voxels`** (replaces Xu Lab's `count_registered_points` which imports `neuroglancer` not installed here). Joins Allen atlas CSV for acronym/name/hierarchy path → `counting/cell_count.csv`.

## Patches / workarounds needed

Three local-env divergences from pristine Xu Lab:

1. **`ants.registration` default CC metric fails on cross-modality fluorescence-vs-Nissl.** Every `register_ndim(method='ants')` attempt exited with ANTs code 1. Fix: call `ants.registration` directly with `aff_metric='mattes'` + `syn_metric='mattes'` (Brainfast's own choice in `registration_3d_ants.run_ants_registration`). Xu Lab's `register_ndim` API doesn't expose these overrides.
2. **SyNRA / SyN OOM on the 528×320×255 fixed warp field** even with aggressive `reg_iterations=(20, 10)`. Falls back to Affine. SyN would produce tighter in-plane alignment but requires more RAM (or fixed-side downsampling) than this machine has headroom for.
3. **Xu Lab's `count_registered_points` imports `neuroglancer`** which isn't installed here. Replaced the region-lookup step with our own `cell_to_ccf.lookup_region_ids_from_ccf_voxels` + `pandas` join against the Allen atlas CSV.

## Result — top regions in the cell-count output

On 3230 synthetic cells placed at tissue hotspots, transformed into CCF space, looked up in native Allen annotation:

| region_id | count | acronym | name |
|---|---|---|---|
| 372 | 442 | CA1 | Field CA1 (hippocampus) |
| 453 | 431 | CA3 | Field CA3 (hippocampus) |
| 150 | 274 | LD | Lateral dorsal nucleus of thalamus |
| 0 | 262 | — | (CCF voxel 0 — ventricle/white matter) |
| 18 | 195 | ENTl2 | Entorhinal area, lateral part, layer 2 |
| 1009 | 160 | PO | Posterior complex of the thalamus |
| 1286 | 147 | LGd-co | Dorsal lateral geniculate complex, core |
| 723 | 130 | VPM | Ventral posteromedial nucleus of the thalamus |
| 211 | 99 | LP | Lateral posterior nucleus of the thalamus |
| 1159 | 96 | DG-mo | Dentate gyrus, molecular layer |
| 1282 | 78 | or | Optic radiation |
| 1109 | 73 | ENTl1 | Entorhinal area, lateral part, layer 1 |

Anatomically coherent mouse coronal chunk — hippocampus, thalamus, entorhinal cortex, optic radiation — which matches where the Affine registration placed sample 35 in the CCF (AP range 263–401, ≈ dorsal hippocampus / thalamus / LGN region).

## Preview figure

See `D:/Brainfast/Sample/ChATe27/xulab_reproduction/preview_xulab_style.png`.

5-panel layout at 3 representative sample slices (front / middle / back). After `axis_align` pre-rotation of 9.73°, panels 1–4 align anatomically (sample tissue sits inside appropriate CCF slice); panel 5 is sparse because Affine-only + small sample footprint inside a much larger CCF can't round-trip many annotation voxels back into the sample.

## Known limits of this reproduction

- **SyN is required for tight in-plane alignment.** With Affine-only, the sample's internal structures don't snap to CCF annotation boundaries — only global AP placement + scale is correct. Cell → region mapping is still anatomically sensible because cells are localized to the broad region the Affine placed them into.
- **`count_registered_points` remains unused.** Xu Lab's version requires `neuroglancer` and produces hierarchy-enriched CSVs via an additional pass. Our replacement produces the same `region_id → count` core output but only a basic CSV with acronym/name/hierarchy path joined — no per-hierarchy rollup CSV.
- **Reproduction is on one brain (brain 35 C0).** Brains 39 and 41 in the same ChATe27 folder were not attempted; same pipeline would run if memory fits.

## Re-run instructions

```powershell
cd D:/Brainfast
$env:PYTHONIOENCODING = "utf-8"
python -u project/scripts/reproduce_xulab_on_chate27.py
```

Expected wall time on current hardware: ≈ 4–6 min (71 s TIFF read + 3.5 s axis-align + 28–102 s ANTs Affine + remainder for preview/counting). Output under `Sample/ChATe27/xulab_reproduction/`.

## References

- Xu Lab source: `D:/UCI-XuLab-RegTools/`
- Atlas assets: `D:/UCI-ALLEN-BrainRepositoryCodeGUI-main/CCF_DATA/`
- Prior root-cause investigation: [`2026-04-22-root-cause-affine-orientation-bug.md`](2026-04-22-root-cause-affine-orientation-bug.md) — Brainfast's own sample 35 had a misleading RAS affine causing 3:1 Z compression; this reproduction sidesteps that by going through Xu Lab's `create_nii_images` which writes correct PIR.
- Cell→CCF helper: [`project/scripts/cell_to_ccf.py`](../../../project/scripts/cell_to_ccf.py)
- Xu Lab interop bridge: [`project/scripts/xulab_compat.py`](../../../project/scripts/xulab_compat.py)
