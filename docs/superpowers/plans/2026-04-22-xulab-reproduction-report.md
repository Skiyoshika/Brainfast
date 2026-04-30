# Xu Lab RegTool reproduction on Brainfast sample 35

> **Date:** 2026-04-22 · **Branch:** `v0_5-polish-2026-04-22` · **Author:** Claude Code autonomous session
>
> **Goal (user-directed):** "在全面了解 Xu Lab RegTool 之后，至少先复现 Xu Lab RegTool 的配准和细胞计数，直接做到能复现为止。"

---

## TL;DR — WORK IN PROGRESS

*(Fill in once reproduction run completes — placeholder while ANTs is cranking.)*

- Brainfast sample 35 `input_volume.nii.gz` (111 slices @25µm, physical Z=2.775 mm) was passed into Xu Lab's `register_ndim(ndim=3, method='ants', half_mode='right', skip_laplacian=True)` via direct Python import (no separate Xu Lab install needed — both tools live in same Brainfast venv + antspy).
- *(result pending)*
- *(finding pending)*

---

## Why this reproduction exists

Brainfast's own 3D ANTs registration produces a **3:1 Z-axis compression**: sample 35's 111 slices (physical 2.775 mm) are mapped to ~35 CCF slices (~875 µm) instead of the anatomically-expected ~111 slices (~2.775 mm matching physical extent). This breaks every downstream step:

- `annotation_registered.nii.gz` drops 34% of leaf regions in the AP range ([`2026-04-22-annotation-granularity-root-cause.md`](2026-04-22-annotation-granularity-root-cause.md)).
- Cell → region mapping is off — every cell's CCF Z is pulled to the compressed range.
- Liquify correction operates on an already-corrupted base.

The [cell_to_ccf.py](../../../project/scripts/cell_to_ccf.py) Phase 1 scaffolding committed earlier is mathematically correct (verified via image-warp ground truth, matches within Δ=0.3 voxel), but because the underlying ANTs transforms are Z-compressed, the output is compressed too.

**This reproduction is the first question**: does Xu Lab's plain `ants.registration(type_of_transform='SyN')` produce the anatomically-correct Z coverage, while Brainfast's `SyNRA` + custom `reg_iterations` + Mattes MI overrides produce the compression? If yes, Brainfast just needs to roll back to Xu Lab-aligned ANTs params.

---

## Setup

- **Python env:** Brainfast venv at `D:/Brainfast` (Python 3.11, antspy 0.x installed)
- **Xu Lab source:** `D:/UCI-XuLab-RegTools/` (not separately installed — imported directly via `sys.path.insert(0, ...)`)
- **Fixed (CCF template):** `project/configs/allen_ref_cache/average_template_25.nii.gz` (528×320×456, 25µm iso)
- **Moving (sample):** `project/outputs/35_C0_demo_run2/volume/input_volume.nii.gz` (111×409×340, 25×20×20 µm)
- **Annotation:** `project/annotation_25.nii.gz` (672 unique leaf regions)
- **Atlas CSV:** `D:/UCI-ALLEN-BrainRepositoryCodeGUI-main/CCF_DATA/1_adult_mouse_brain_graph_mapping.csv` (Xu Lab-compatible schema)
- **Cells:** `project/outputs/35_C0_demo_run2/cells_dedup.csv` (deduped detector output from the same ANTs run)

### ANTs registration parameter diff

| Parameter | Brainfast | Xu Lab default |
|---|---|---|
| `type_of_transform` | **SyNRA** (Rigid + Affine + SyN) | **SyN** (SyN only) |
| `aff_metric` | **mattes** (explicit) | default (CC / mutual info) |
| `syn_metric` | **mattes** (explicit) | default |
| `syn_sampling` | **32** (explicit) | default |
| `reg_iterations` | **(200, 200, 100, 50)** (explicit 4-level) | default multi-res schedule |
| `rescale_moving` / `rescale_fixed` | — (no rescaling) | (passable, used `True` in this repro for uint16→float32) |

**Hypothesis:** Brainfast's aggressive SyNRA rigid+affine front-end mis-aligns the sample in Z before SyN has a chance to correct, producing the 3:1 compression. Xu Lab's plain SyN starts from a less-aggressive initial pose and should produce anatomically-faithful Z coverage.

---

## Run command

```python
# See project/tmp_xulab_reproduce.py for the full script
register_ndim(
    ndim=3,
    fixed_path=str(CCF_TEMPLATE),
    moving_path=str(SAMPLE_VOLUME),
    output_dir=str(OUT),
    axis_align=False,
    half_mode="right",
    atlas="allen_ccf",
    method="ants",
    skip_laplacian=True,
    random_seed=42,
    rescale_moving=True,
    rescale_fixed=True,
    verbose=True,
)
```

---

## Results (to be filled)

### Registration Z coverage

| Source | Nonzero CCF-Z range | Slice count | Physical range |
|---|---|---|---|
| Brainfast `ants_result.nii.gz` | [28, 62] approx | ~35 | ~875 µm (3:1 compression) |
| Xu Lab reproduction `02_nonlinear/result.nii.gz` | *(pending)* | *(pending)* | *(pending)* |
| Expected (anatomically-faithful) | ~[92, 202] | ~111 | ~2.775 mm |

### Cell → region mapping

| Source | Cells | Unmapped | Unique region IDs | Top 5 regions |
|---|---|---|---|---|
| Brainfast `cells_mapped.csv` | 59,290 | 1,001 | 147 | *(see CSV)* |
| Xu Lab reproduction `cell_count.csv` | 59,290 | *(pending)* | *(pending)* | *(pending)* |

---

## Conclusion (to be filled)

One of:

### Scenario A — Xu Lab runs, good Z coverage

Brainfast's SyNRA + custom params are the bug. Fix: roll back `whole_brain_3d.py` / `registration_3d_ants.py` to Xu Lab-aligned defaults. Expected diff:

```python
# registration_3d_ants.py:160
reg = ants.registration(
    fixed=fixed_img,
    moving=moving_img,
    type_of_transform="SyN",          # was "SyNRA"
    random_seed=int(random_seed),
    verbose=False,
    # Remove: aff_metric, syn_metric, syn_sampling, reg_iterations
)
```

### Scenario B — Xu Lab runs, same Z compression

Params are innocent. Issue is deeper — sample `input_volume.nii.gz` metadata, Allen CCF template geometry, or ANTs version behavior. Next debugging step: compare a known-good UCI-ALLEN sample against CCF using same code path.

### Scenario C — Xu Lab also fails

Environment / antspy issue. Next: run `ants.registration` in complete isolation (minimal repro), upgrade/downgrade antspy, rebuild venv.

---

## Next actions after this report

1. If Scenario A: write `fix(3d-registration): align ANTs params with Xu Lab` commit. Re-run sample 35 end-to-end. Verify annotation_registered has 528→~110 nonzero Z span.
2. If Scenario B: spike per-slice 2D registration pipeline (Xu Lab `registration_batch_2d`) which sidesteps 3D warp entirely.
3. If Scenario C: produce a minimal antspy repro and document the env incompatibility. Consider pinning antspy version in `pyproject.toml`.
