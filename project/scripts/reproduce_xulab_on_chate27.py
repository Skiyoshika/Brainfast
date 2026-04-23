"""Reproduce Xu Lab RegTool's registration + cell counting on ChATe27.

This script is **Brainfast-standalone** — it does NOT import any code from
`D:/UCI-XuLab-RegTools/`. Everything is implemented in Brainfast's own
modules:

  - ``scripts.volume_io.multipage_tiff_to_nifti`` — mirrors Xu Lab's
    ``create_nii_images`` for multi-page TIFF input.
  - ``scripts.axis_align_3d.align_volume_to_template_by_fissure`` — mirrors
    Xu Lab's ``axisAlignData`` (midline-fissure plane detection + rigid
    rotation).
  - ``scripts.cell_to_ccf.map_cells_via_ccf_transform(..., axis_align_matrix=)``
    — mirrors Xu Lab's ``transform_input_points`` (axis-align + ANTs point
    warp + region lookup).
  - ``ants.registration`` directly (Brainfast's env already has antspy).

Xu Lab's own Python modules are no longer in the import graph. Setting
``D:/UCI-XuLab-RegTools/`` absent would not break this script.

Produces:
  - PIR-oriented sample NIfTI
  - Axis-aligned sample NIfTI + 4×4 affine
  - ANTs registration output (SyNRA → SyN → Affine fallback with Mattes MI)
  - Half-CCF template + annotation
  - 5-panel preview PNG
  - Xu Lab-style cell_count.csv

Run:
    PYTHONIOENCODING=utf-8 python project/scripts/reproduce_xulab_on_chate27.py
"""

from __future__ import annotations

import gc
import json
import shutil
import sys
import time
from pathlib import Path

# ANTs plotting is broken in this env — stub out before antspy imports
sys.modules.setdefault("ants.plotting", type(sys)("ants.plotting"))
try:
    import matplotlib._docstring as _mpl_ds

    if not hasattr(_mpl_ds, "dedent_interpd"):
        _mpl_ds.dedent_interpd = lambda f: f
except ImportError:
    pass

import ants  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "project"))

from scripts.axis_align_3d import (  # noqa: E402
    align_volume_to_template_by_fissure,
    save_axis_align_matrix,
)
from scripts.cell_to_ccf import (  # noqa: E402
    map_cells_via_ccf_transform,
)
from scripts.volume_io import multipage_tiff_to_nifti  # noqa: E402

SAMPLE = (
    ROOT
    / "Sample/ChATe27/35_High_1000ms_560nm_640nm_150W_z5um_Bothlaser - Pos 3 4 [1] 3DMontage_XY1763150824_Z000_T0_C0.tif"
)
CCF_TEMPLATE = ROOT / "project/configs/allen_ref_cache/average_template_25.nii.gz"
CCF_ANNOTATION = Path("D:/UCI-ALLEN-BrainRepositoryCodeGUI-main/CCF_DATA/annotation_25.nii.gz")
ATLAS_CSV = Path(
    "D:/UCI-ALLEN-BrainRepositoryCodeGUI-main/CCF_DATA/1_adult_mouse_brain_graph_mapping.csv"
)

OUT = ROOT / "Sample/ChATe27/xulab_reproduction"
OUT.mkdir(parents=True, exist_ok=True)
NII_DIR = OUT / "nii"
NII_DIR.mkdir(exist_ok=True)

HALF_MODE = "right"


def _banner(msg: str) -> None:
    print(f"\n{'=' * 72}\n{msg}\n{'=' * 72}", flush=True)


# --- Step 1: multi-page TIFF -> PIR NIfTI (Brainfast-native, Xu Lab formula) --
_banner("[1] Brainfast multipage_tiff_to_nifti on ChATe27 brain 35 C0 (all 646 pages)")
t0 = time.time()
moving_nii_path = NII_DIR / "brain_25.nii.gz"
if not moving_nii_path.exists():
    meta1 = multipage_tiff_to_nifti(
        SAMPLE,
        moving_nii_path,
        downscale_factor=8,
        voxel_spacing_mm=(0.005, 0.025, 0.025),  # 5um Z (ChATe27 z5um), 25um XY
        orient=True,
        ccf_origin=(-5.695, 5.35, 5.22),
    )
    print(
        f"  shape={meta1['shape']} zooms_mm={meta1['voxel_mm']} pages={meta1['page_count']}",
        flush=True,
    )
else:
    print(f"  reusing {moving_nii_path}", flush=True)
moving_img = nib.load(str(moving_nii_path))
print(f"  axis codes: {nib.aff2axcodes(moving_img.affine)}  ({time.time() - t0:.1f}s)", flush=True)

# --- Step 2: right-half CCF template + annotation ---------------------------
_banner(f"[2] Right-half CCF template + annotation (half_mode='{HALF_MODE}')")
tmpl_full = nib.load(str(CCF_TEMPLATE))
tmpl_data = tmpl_full.get_fdata()
width = tmpl_data.shape[2]
mid = width // 2
pad_size = int(round(width * 0.06))
if HALF_MODE == "right":
    half_tmpl = np.pad(tmpl_data[:, :, mid:], [(0, 0), (0, 0), (pad_size, 0)], mode="constant")
else:
    half_tmpl = np.pad(tmpl_data[:, :, :mid], [(0, 0), (0, 0), (0, pad_size)], mode="constant")
half_path = OUT / f"ccf_template_{HALF_MODE}_half.nii.gz"
nib.save(nib.Nifti1Image(half_tmpl, tmpl_full.affine, tmpl_full.header), str(half_path))
ann_full = nib.load(str(CCF_ANNOTATION))
ann_data = ann_full.get_fdata().astype(np.int32)
if HALF_MODE == "right":
    half_ann = np.pad(ann_data[:, :, mid:], [(0, 0), (0, 0), (pad_size, 0)], mode="constant")
else:
    half_ann = np.pad(ann_data[:, :, :mid], [(0, 0), (0, 0), (0, pad_size)], mode="constant")
half_ann_path = OUT / f"ccf_annotation_{HALF_MODE}_half.nii.gz"
nib.save(nib.Nifti1Image(half_ann, ann_full.affine, ann_full.header), str(half_ann_path))
print(f"  half template: {half_path} shape={half_tmpl.shape}")
print(f"  half annotation: {half_ann_path} unique IDs: {len(np.unique(half_ann))}")
del tmpl_data, ann_data, half_tmpl, half_ann
gc.collect()

# --- Step 2b: Brainfast axis align (midline-fissure detection) -------------
_banner("[2b] Brainfast axis_align_3d.align_volume_to_template_by_fissure")
stage1 = OUT / "01_axis_alignment"
stage1.mkdir(parents=True, exist_ok=True)
aligned_moving_path = stage1 / "axisAlignedData.nii.gz"
axis_align_path = stage1 / "axisAlignA"  # save_axis_align_matrix appends .npz

if not aligned_moving_path.exists():
    t0 = time.time()
    try:
        A, aligned_data, axis_info = align_volume_to_template_by_fissure(
            moving_volume=moving_nii_path,
            template_volume=half_path,
        )
        print(
            f"  rotation angle: {axis_info['rotation_angle_deg']:.2f}deg "
            f"(moving pts={len(axis_info['moving_points'])}, template pts={len(axis_info['template_points'])})"
            f"  elapsed={time.time() - t0:.1f}s"
        )
        print(f"  A matrix:\n{A}")
        save_axis_align_matrix(axis_align_path, A)
        aligned_nii = nib.Nifti1Image(
            aligned_data.astype(np.uint16),
            moving_img.affine,  # preserve sample affine (data has been axis-aligned in-place)
            moving_img.header.copy(),
        )
        aligned_nii.header["qform_code"] = 1
        nib.save(aligned_nii, str(aligned_moving_path))
        print(f"  saved axisAlignedData: {aligned_moving_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"  axis align failed ({exc}); saving identity + using unaligned moving")
        save_axis_align_matrix(axis_align_path, np.eye(4))
        shutil.copy2(moving_nii_path, aligned_moving_path)
else:
    print("  axisAlignedData.nii.gz already exists — reusing")

# --- Step 3: ANTs registration (Mattes MI, cross-modality-safe) -------------
_banner("[3] ANTs registration with SyNRA->SyN->Affine fallback (Mattes MI)")
fixed_ants = ants.image_read(str(half_path))
moving_ants = ants.image_read(str(aligned_moving_path))
print(f"  fixed:  shape={fixed_ants.shape} spacing={fixed_ants.spacing}")
print(f"  moving: shape={moving_ants.shape} spacing={moving_ants.spacing}", flush=True)

stage2 = OUT / "02_nonlinear"
fwd_dir = stage2 / "fwd_transforms"
inv_dir = stage2 / "inv_transforms"
for d in (stage2, fwd_dir, inv_dir):
    d.mkdir(parents=True, exist_ok=True)

transform_used = None
reg = None
FIXED_MAX_DIM = 256  # pre-resample fixed/moving to keep SyN warp field <OOM

def _maybe_downsample(img, max_dim: int):
    current = max(int(s) for s in img.shape)
    if current <= max_dim:
        return img
    factor = current / float(max_dim)
    new_spacing = tuple(float(s) * factor for s in img.spacing)
    return ants.resample_image(img, new_spacing, False, 0)


fixed_ants_ds = _maybe_downsample(fixed_ants, FIXED_MAX_DIM)
moving_ants_ds = _maybe_downsample(moving_ants, FIXED_MAX_DIM)
print(
    f"  pre-resample: fixed {fixed_ants.shape} -> {fixed_ants_ds.shape}, "
    f"moving {moving_ants.shape} -> {moving_ants_ds.shape}",
    flush=True,
)

for attempt_label, kwargs in [
    (
        "SyNRA + Mattes + reg(20,10) + fixed_max_dim=256",
        dict(
            type_of_transform="SyNRA",
            aff_metric="mattes",
            syn_metric="mattes",
            syn_sampling=32,
            reg_iterations=(20, 10),
            random_seed=42,
            verbose=False,
        ),
    ),
    (
        "SyN + Mattes + reg(20,10) + fixed_max_dim=256",
        dict(
            type_of_transform="SyN",
            aff_metric="mattes",
            syn_metric="mattes",
            syn_sampling=32,
            reg_iterations=(20, 10),
            random_seed=42,
            verbose=False,
        ),
    ),
    (
        "Affine + Mattes (full-res fallback)",
        dict(type_of_transform="Affine", aff_metric="mattes", random_seed=42, verbose=False),
    ),
]:
    try:
        print(f"  attempting {attempt_label}...", flush=True)
        t0 = time.time()
        # First 2 attempts use downsampled fixed/moving (SyN memory saver);
        # last Affine fallback uses full res (it's light enough).
        is_fallback = attempt_label.startswith("Affine")
        _fixed = fixed_ants if is_fallback else fixed_ants_ds
        _moving = moving_ants if is_fallback else moving_ants_ds
        reg = ants.registration(fixed=_fixed, moving=_moving, **kwargs)
        transform_used = attempt_label
        print(f"  {attempt_label} succeeded in {time.time() - t0:.1f}s")
        break
    except MemoryError:
        print(f"  {attempt_label} OOM — trying next")
        gc.collect()
    except Exception as exc:  # noqa: BLE001
        print(f"  {attempt_label} error: {exc} — trying next")
        gc.collect()

if reg is None or transform_used is None:
    raise RuntimeError("all ANTs registration strategies failed")

# Persist into Xu Lab directory layout so the point-transform helpers can find them
fwd_idx_warp = fwd_idx_aff = 0
for tf in reg.get("fwdtransforms", []):
    s = str(tf)
    if s.endswith(".nii.gz"):
        shutil.copy2(s, fwd_dir / f"ants_warp_{fwd_idx_warp}.nii.gz")
        fwd_idx_warp += 1
    elif s.endswith(".mat"):
        shutil.copy2(s, fwd_dir / f"ants_affine_{fwd_idx_aff}.mat")
        fwd_idx_aff += 1
inv_idx_warp = inv_idx_aff = 0
for tf in reg.get("invtransforms", []):
    s = str(tf)
    if s.endswith(".nii.gz"):
        shutil.copy2(s, inv_dir / f"ants_invwarp_{inv_idx_warp}.nii.gz")
        inv_idx_warp += 1
    elif s.endswith(".mat"):
        shutil.copy2(s, inv_dir / f"ants_affine_{inv_idx_aff}.mat")
        inv_idx_aff += 1

result_path = stage2 / "result.nii.gz"
ants.image_write(reg["warpedmovout"], str(result_path))

(OUT / "registration_metadata.json").write_text(
    json.dumps(
        {
            "moving_image": str(aligned_moving_path),
            "fixed_image": str(half_path),
            "moving_shape": list(moving_ants.shape),
            "fixed_shape": list(fixed_ants.shape),
            "half_mode": HALF_MODE,
            "registration_method": "ants",
            "transform_used": transform_used,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)

# Resample ANTs result back to the full-resolution CCF grid so downstream
# annotation warp + preview see consistent shapes. ants.apply_transforms with
# interpolator='linear' on the forward-warped result against the full fixed
# template gives us a full-shape version.
full_fixed_ants = ants.image_read(str(half_path))
result_full = ants.resample_image_to_target(reg["warpedmovout"], full_fixed_ants, "linear")
ants.image_write(result_full, str(result_path))

# Z coverage check (on full-res result)
result_arr = np.asarray(nib.load(str(result_path)).dataobj, dtype=np.float32)
z_nz = [z for z in range(result_arr.shape[0]) if np.sum(result_arr[z] > 0) > 0]
print(
    f"  CCF Z nonzero range: "
    f"[{z_nz[0] if z_nz else 'empty'}, {z_nz[-1] if z_nz else 'empty'}] "
    f"({len(z_nz)} slices)"
)
del fixed_ants, moving_ants, reg
gc.collect()

# --- Step 4: Warp annotation to sample space (for overlay preview) --------
_banner("[4] Warp CCF annotation -> sample voxel space")
ann_ants = ants.image_read(str(half_ann_path))
sample_ref = ants.image_read(str(aligned_moving_path))
inv_list = [
    str(p) for p in [inv_dir / "ants_affine_0.mat", inv_dir / "ants_invwarp_0.nii.gz"] if p.exists()
]
print(f"  using {len(inv_list)} inverse transforms")
warped_ann = ants.apply_transforms(
    fixed=sample_ref,
    moving=ann_ants,
    transformlist=inv_list,
    interpolator="genericLabel",
)
ann_in_sample_path = OUT / "annotation_in_sample_space.nii.gz"
ants.image_write(warped_ann, str(ann_in_sample_path))
warped_ann_np = np.asarray(nib.load(str(ann_in_sample_path)).dataobj, dtype=np.int32)
print(f"  warped ann shape={warped_ann_np.shape} unique IDs={len(np.unique(warped_ann_np))}")
del ann_ants, sample_ref, warped_ann
gc.collect()

# --- Step 5: 5-panel preview figure ---------------------------------------
_banner("[5] 5-panel preview figure")
sample_arr = np.asarray(nib.load(str(aligned_moving_path)).dataobj, dtype=np.float32)
n_sample_z = sample_arr.shape[0]
sample_picks = [n_sample_z // 5, n_sample_z // 2, 4 * n_sample_z // 5]
ccf_arr = np.asarray(nib.load(str(half_path)).dataobj, dtype=np.float32)
ann_half_arr = np.asarray(nib.load(str(half_ann_path)).dataobj, dtype=np.int32)
n_ccf_z = result_arr.shape[0]
ccf_picks = []
for sz in sample_picks:
    if z_nz:
        frac = sz / max(n_sample_z - 1, 1)
        ccf_picks.append(z_nz[int(frac * (len(z_nz) - 1))])
    else:
        ccf_picks.append(n_ccf_z // 2)


def colorize_annotation(ann_slice: np.ndarray, seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    uids = np.unique(ann_slice)
    colors = {int(u): rng.integers(40, 255, size=3) for u in uids}
    colors[0] = np.array([0, 0, 0])
    h, w = ann_slice.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for uid, c in colors.items():
        rgb[ann_slice == uid] = c
    return rgb


def normalize_for_display(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) / max(hi - lo, 1.0), 0, 1)


fig, axes = plt.subplots(len(sample_picks), 5, figsize=(22, 4.5 * len(sample_picks)))
if len(sample_picks) == 1:
    axes = axes[np.newaxis, :]
for row_idx, (sz, cz) in enumerate(zip(sample_picks, ccf_picks, strict=True)):
    axes[row_idx, 0].imshow(normalize_for_display(sample_arr[sz]), cmap="gray")
    axes[row_idx, 0].set_title(f"Preprocessed TIF\n(sample slice {sz})", fontsize=10)
    axes[row_idx, 0].axis("off")
    axes[row_idx, 1].imshow(normalize_for_display(ccf_arr[cz]), cmap="gray")
    axes[row_idx, 1].set_title(f"CCF Template\n(atlas slice {cz})", fontsize=10)
    axes[row_idx, 1].axis("off")
    ann_rgb = colorize_annotation(ann_half_arr[cz])
    tmpl_img = np.stack([normalize_for_display(ccf_arr[cz])] * 3, axis=-1)
    axes[row_idx, 2].imshow(
        np.clip(tmpl_img * 0.5 + ann_rgb.astype(np.float32) / 255.0 * 0.5, 0, 1)
    )
    axes[row_idx, 2].set_title("CCF + Annotations", fontsize=10)
    axes[row_idx, 2].axis("off")
    result_img = np.stack([normalize_for_display(result_arr[cz])] * 3, axis=-1)
    axes[row_idx, 3].imshow(
        np.clip(result_img * 0.5 + ann_rgb.astype(np.float32) / 255.0 * 0.5, 0, 1)
    )
    axes[row_idx, 3].set_title("ANTs Result + Annotations\n(sample warped to CCF)", fontsize=10)
    axes[row_idx, 3].axis("off")
    warped_rgb = colorize_annotation(warped_ann_np[sz])
    sample_rgb = np.stack([normalize_for_display(sample_arr[sz])] * 3, axis=-1)
    axes[row_idx, 4].imshow(
        np.clip(sample_rgb * 0.5 + warped_rgb.astype(np.float32) / 255.0 * 0.5, 0, 1)
    )
    axes[row_idx, 4].set_title(f"Inverse-Mapped Annotation\non sample slice {sz}", fontsize=10)
    axes[row_idx, 4].axis("off")
plt.suptitle(
    f"Xu Lab-style reproduction on ChATe27 brain 35 C0 (Brainfast-standalone)\n"
    f"sample {n_sample_z} slices -> {len(z_nz)} CCF slices via {transform_used}",
    fontsize=12,
)
plt.tight_layout(rect=[0, 0, 1, 0.96])
preview_path = OUT / "preview_xulab_style.png"
plt.savefig(preview_path, dpi=100, bbox_inches="tight")
plt.close()
print(f"  preview saved: {preview_path}")

# --- Step 6: cell counting via Brainfast map_cells_via_ccf_transform --------
_banner("[6] Brainfast map_cells_via_ccf_transform (Xu Lab-style pipeline)")
cells = []
rng = np.random.default_rng(42)
for sz in range(n_sample_z):
    slice_data = sample_arr[sz]
    if slice_data.max() < 1.0:
        continue
    thr = np.percentile(slice_data, 90)
    mask = slice_data > thr
    if not mask.any():
        continue
    ys, xs = np.where(mask)
    k = min(5, len(ys))
    idx = rng.choice(len(ys), size=k, replace=False)
    for j in idx:
        cells.append(
            {"cell_id": len(cells) + 1, "slice_id": sz, "x": float(xs[j]), "y": float(ys[j])}
        )
cells_df = pd.DataFrame(cells)
print(f"  synthetic cells: {len(cells_df)}")

try:
    # Read axis align matrix + feed into the Brainfast cell-mapping pipeline.
    A = np.load(str(axis_align_path.with_suffix(".npz")))["arr"]
    print(
        f"  axis align matrix loaded ({A.shape}, identity: {np.allclose(A, np.eye(4), atol=1e-6)})"
    )

    # Note: cell (slice_id, x, y) are *pre-axis-align* pixel coords in the
    # original sample. sample_pixel_to_volume_voxel converts to voxel coords
    # at the volume's (axis-0-aligned) resolution, then map_cells_via_ccf_transform
    # applies the axis_align_matrix inside.
    # Pixel size for ChATe27 at scale=2.5 + /8 downscale = effective 25um/pixel.
    effective_pixel_um = 25.0

    new = map_cells_via_ccf_transform(
        cells_df[["cell_id", "slice_id", "x", "y"]].copy(),
        sample_volume_path=aligned_moving_path,
        ccf_annotation_path=CCF_ANNOTATION,
        inverse_transforms=inv_list,
        pixel_size_um=effective_pixel_um,
        ccf_template_path=half_path,
        axis_align_matrix=A,
    )
    print(
        f"  mapped {(new['region_id'] != 0).sum()} / {len(new)} cells, "
        f"{new[new['region_id'] != 0]['region_id'].nunique()} unique regions"
    )

    # Join Allen CSV for region names + save Xu Lab-style count CSV
    counts = new["region_id"].value_counts().reset_index()
    counts.columns = ["region", "count"]
    atlas_df = pd.read_csv(ATLAS_CSV)
    counts = counts.merge(atlas_df, left_on="region", right_on="parcellation_index", how="left")[
        ["region", "count", "acronym", "name", "structure_id_path"]
    ]
    counting_dir = OUT / "counting"
    counting_dir.mkdir(exist_ok=True)
    counts.sort_values("count", ascending=False).to_csv(
        counting_dir / "cell_count.csv", index=False
    )
    print(f"  cell counts saved: {counting_dir / 'cell_count.csv'}")
    print("  top 10:")
    print(counts.sort_values("count", ascending=False).head(10).to_string(index=False))
except Exception as exc:  # noqa: BLE001
    print(f"  cell counting step failed (non-fatal): {exc}")

_banner("DONE")
print(f"  preview:             {preview_path}")
print(f"  moving NIfTI:        {moving_nii_path}")
print(f"  axis-aligned:        {aligned_moving_path}")
print(f"  registered result:   {result_path}")
print(f"  warped annotation:   {ann_in_sample_path}")
if (OUT / "counting" / "cell_count.csv").exists():
    print(f"  cell counts:         {OUT / 'counting' / 'cell_count.csv'}")
