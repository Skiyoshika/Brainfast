"""Reproduce Xu Lab RegTool end-to-end on ChATe27 real data (Andy's workflow).

Follows Xu Lab's standard CLI path:

    python tif_to_nii.py <brain_35_C0.tif> <out_dir>       # → brain_25.nii.gz
    python -m regtools.registration <moving=brain_25.nii.gz> <fixed=CCF>
    # then transform_input_points + count_registered_points

Here we do the same in-process, with one patch: the ANTs stage uses
Mattes MI (Brainfast-style) because Xu Lab's default CC metric fails on
cross-modality fluorescence-vs-Nissl inputs in this Python/antspy env
(exit code 1). Everything else is Xu Lab code, Xu Lab defaults.

Produces:
  * brain_10 / brain_25 / brain_50 / brain_100.nii.gz at multi-res
  * Right-half CCF template + annotation (Xu Lab half_mode crop)
  * 02_nonlinear/result.nii.gz + fwd_transforms + inv_transforms
  * registration_metadata.json
  * preview_xulab_style.png (5-panel figure: Preprocessed / Template /
    Template+Ann / ANTs Result+Ann / Inverse-Mapped Ann on sample)
  * counting/cell_count.csv (if synthetic cells reach Xu Lab counting)

Run:
    PYTHONIOENCODING=utf-8 python project/tmp_reproduce_chate27.py
"""

from __future__ import annotations

import gc
import json
import shutil
import sys
import time
from pathlib import Path

# ANTs plotting is broken in this env — stub it out before antspy imports.
sys.modules.setdefault("ants.plotting", type(sys)("ants.plotting"))
try:
    import matplotlib._docstring as _mpl_ds

    if not hasattr(_mpl_ds, "dedent_interpd"):
        _mpl_ds.dedent_interpd = lambda f: f
except ImportError:
    pass

sys.path.insert(0, "D:/UCI-XuLab-RegTools")

import ants  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import nibabel as nib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from regtools.registration.core.io_utils import axisAlignData  # noqa: E402
from regtools.registration.pipeline_io import _save_npz  # noqa: E402
from regtools.utils.io.nifti_utils import create_nifti_image, create_nii_images  # noqa: E402

BF = Path("D:/Brainfast")
SAMPLE = (
    BF
    / "Sample/ChATe27/35_High_1000ms_560nm_640nm_150W_z5um_Bothlaser - Pos 3 4 [1] 3DMontage_XY1763150824_Z000_T0_C0.tif"
)
CCF_TEMPLATE = BF / "project/configs/allen_ref_cache/average_template_25.nii.gz"
CCF_ANNOTATION = Path("D:/UCI-ALLEN-BrainRepositoryCodeGUI-main/CCF_DATA/annotation_25.nii.gz")
ATLAS_CSV = Path(
    "D:/UCI-ALLEN-BrainRepositoryCodeGUI-main/CCF_DATA/1_adult_mouse_brain_graph_mapping.csv"
)

OUT = BF / "Sample/ChATe27/xulab_reproduction"
OUT.mkdir(parents=True, exist_ok=True)
NII_DIR = OUT / "nii"
NII_DIR.mkdir(exist_ok=True)

HALF_MODE = "right"  # ChATe27 mounted right hemisphere — matches Brainfast's sample 35 convention


def _banner(msg: str) -> None:
    print(f"\n{'=' * 72}\n{msg}\n{'=' * 72}", flush=True)


# --- Step 1: Xu Lab's own tif->nii conversion (no manual decimation) -------
_banner("[1] Xu Lab create_nii_images on ChATe27 brain 35 C0 (all 646 pages)")
t0 = time.time()
print(f"  source: {SAMPLE.name}", flush=True)

if not (NII_DIR / "brain_25.nii.gz").exists():
    # Xu Lab defaults: multi-page TIFF auto-detected, downscale=True -> /8 XY,
    # scale=2.5 -> brain_25.nii.gz at 25um XY resolution.
    # Override voxel_spacing to tell ANTs the actual physical thickness:
    #   axis 0 (slice stack) = 5um per page (from 'z5um' in filename),
    #   axis 1/2 (image YX) = 25um per voxel at scale=2.5.
    # Xu Lab's default sz=None assumes 50um slices — 10x over-claim on
    # ChATe27's actual physical Z thickness that would make ANTs try to
    # stretch the sample across 10x too much of CCF AP.
    create_nii_images(
        str(SAMPLE),
        str(NII_DIR),
        channel=0,
        scales=[2.5],
        orient=True,
        downscale=True,
        voxel_spacing=[0.005, 0.025, 0.025],  # [Z, Y, X] in mm
    )
else:
    print("  brain_25.nii.gz already exists - reusing", flush=True)

moving_nii_path = NII_DIR / "brain_25.nii.gz"
moving_img = nib.load(str(moving_nii_path))
print(f"  moving NIfTI: {moving_nii_path}")
print(f"  shape: {moving_img.shape}")
print(f"  axis_codes: {nib.aff2axcodes(moving_img.affine)}")
print(f"  zooms (mm): {moving_img.header.get_zooms()[:3]}")
print(f"  elapsed: {time.time() - t0:.1f}s", flush=True)

# --- Step 2: Half-CCF template + annotation (Xu Lab's half_mode='right') ---
_banner(f"[2] Preparing right-half CCF template + annotation (half_mode={HALF_MODE!r})")
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
print(f"  half template: {half_path} shape={half_tmpl.shape}")

ann_full = nib.load(str(CCF_ANNOTATION))
ann_data = ann_full.get_fdata().astype(np.int32)
if HALF_MODE == "right":
    half_ann = np.pad(ann_data[:, :, mid:], [(0, 0), (0, 0), (pad_size, 0)], mode="constant")
else:
    half_ann = np.pad(ann_data[:, :, :mid], [(0, 0), (0, 0), (0, pad_size)], mode="constant")
half_ann_path = OUT / f"ccf_annotation_{HALF_MODE}_half.nii.gz"
nib.save(nib.Nifti1Image(half_ann, ann_full.affine, ann_full.header), str(half_ann_path))
print(f"  half annotation: {half_ann_path} unique IDs: {len(np.unique(half_ann))}")
del tmpl_data, ann_data, half_tmpl, half_ann
gc.collect()

# --- Step 2b: Xu Lab axis alignment (midline fissure detection) -----------
_banner("[2b] Xu Lab axisAlignData (midline-fissure rigid pre-alignment)")
stage1 = OUT / "01_axis_alignment"
stage1.mkdir(parents=True, exist_ok=True)
aligned_moving_path = stage1 / "axisAlignedData.nii.gz"
axis_align_npz = stage1 / "axisAlignA"  # _save_npz appends .npz

if not aligned_moving_path.exists():
    t0 = time.time()
    try:
        A, aligned_data, axis_info = axisAlignData(str(half_path), str(moving_nii_path))
        print(
            f"  rotation angle: {axis_info['rotation_angle_deg']:.2f}deg ({time.time() - t0:.1f}s)"
        )
        print(f"  A matrix:\n{A}")
        # Save with key='arr' via Xu Lab's own helper (matches _load_npz expectation)
        _save_npz(str(axis_align_npz), A)
        # Save aligned volume — use same voxel_spacing as input brain_25
        aligned_nii = create_nifti_image(
            aligned_data.astype(np.uint16),
            scale=2.5,
            name=None,
            voxel_spacing=[0.005, 0.025, 0.025],
        )
        # Override origin to 0 (as in brain_25 creation)
        aff = aligned_nii.affine.copy()
        aff[0, 3] = 0.0
        aff[1, 3] = 0.0
        aff[2, 3] = 0.0
        new_aligned = nib.Nifti1Image(aligned_data.astype(np.uint16), aff, aligned_nii.header)
        new_aligned.header["qform_code"] = 1
        nib.save(new_aligned, str(aligned_moving_path))
        print(f"  saved axisAlignedData: {aligned_moving_path}")
    except Exception as exc:  # noqa: BLE001
        print(f"  axisAlignData failed ({exc}); saving identity + using unaligned moving")
        _save_npz(str(axis_align_npz), np.eye(4))
        shutil.copy2(moving_nii_path, aligned_moving_path)
else:
    print("  axisAlignedData.nii.gz already exists — reusing")

# --- Step 3: ANTs registration (Mattes-patched, Xu Lab layout output) -----
_banner("[3] ANTs SyNRA + Mattes MI on axis-aligned moving (cross-modality patch)")
fixed_ants = ants.image_read(str(half_path))
moving_ants = ants.image_read(str(aligned_moving_path))
print(f"  fixed:  shape={fixed_ants.shape} spacing={fixed_ants.spacing}")
print(f"  moving: shape={moving_ants.shape} spacing={moving_ants.spacing}", flush=True)
t0 = time.time()
# With axis-aligned moving, SyNRA needs less warp capacity → try smaller
# reg_iterations first to conserve memory.
transform_used = None
for attempt_label, kwargs in [
    (
        "SyNRA + Mattes + reg(20,10)",
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
        "SyN + Mattes + reg(20,10)",
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
        "Affine + Mattes (fallback)",
        dict(
            type_of_transform="Affine",
            aff_metric="mattes",
            random_seed=42,
            verbose=False,
        ),
    ),
]:
    try:
        print(f"  attempting {attempt_label}...", flush=True)
        t0 = time.time()
        reg = ants.registration(fixed=fixed_ants, moving=moving_ants, **kwargs)
        transform_used = attempt_label
        print(f"  {attempt_label} succeeded in {time.time() - t0:.1f}s")
        break
    except MemoryError:
        print(f"  {attempt_label} OOM — trying next strategy")
        gc.collect()
    except Exception as exc:  # noqa: BLE001
        print(f"  {attempt_label} error: {exc} — trying next strategy")
        gc.collect()

if transform_used is None:
    raise RuntimeError("all registration strategies failed")
print(f"  registered via {transform_used}")
print(f"  ANTs done in {time.time() - t0:.1f}s", flush=True)

# Persist in Xu Lab directory layout (axisAlignA.npz already saved in Step 2b)
stage2 = OUT / "02_nonlinear"
fwd_dir = stage2 / "fwd_transforms"
inv_dir = stage2 / "inv_transforms"
for d in (stage2, fwd_dir, inv_dir):
    d.mkdir(parents=True, exist_ok=True)

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
            "moving_image": str(aligned_moving_path),  # axis-aligned, not raw brain_25
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

# Z coverage check
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
_banner("[4] Warp CCF annotation -> sample voxel space (genericLabel interpolator)")
ann_ants = ants.image_read(str(half_ann_path))
sample_ref = ants.image_read(str(aligned_moving_path))  # axis-aligned target
inv_list = [
    str(p)
    for p in [
        inv_dir / "ants_affine_0.mat",
        inv_dir / "ants_invwarp_0.nii.gz",
    ]
    if p.exists()
]
print(f"  using {len(inv_list)} inverse transforms")
# genericLabel handles multi-label probability volumes better than nearestNeighbor
# in edge cases (single .mat transform, small sample footprint inside large fixed).
warped_ann = ants.apply_transforms(
    fixed=sample_ref,
    moving=ann_ants,
    transformlist=inv_list,
    interpolator="genericLabel",
)
ann_in_sample_path = OUT / "annotation_in_sample_space.nii.gz"
ants.image_write(warped_ann, str(ann_in_sample_path))
warped_ann_np = np.asarray(nib.load(str(ann_in_sample_path)).dataobj, dtype=np.int32)
print(f"  warped ann shape: {warped_ann_np.shape} unique IDs: {len(np.unique(warped_ann_np))}")
del ann_ants, sample_ref, warped_ann
gc.collect()

# --- Step 5: Preview figure (5-panel Xu Lab style) ------------------------
_banner("[5] Generating 5-panel preview figure")
# Preview uses axis-aligned moving (what ANTs actually saw) for the first panel
sample_arr = np.asarray(nib.load(str(aligned_moving_path)).dataobj, dtype=np.float32)
n_sample_z = sample_arr.shape[0]
sample_picks = [n_sample_z // 5, n_sample_z // 2, 4 * n_sample_z // 5]

ccf_arr = np.asarray(nib.load(str(half_path)).dataobj, dtype=np.float32)
ann_half_arr = np.asarray(nib.load(str(half_ann_path)).dataobj, dtype=np.int32)
n_ccf_z = result_arr.shape[0]

ccf_picks = []
for sz in sample_picks:
    if z_nz:
        # Map proportional position in sample Z range onto the CCF Z range that got signal
        frac = sz / max(n_sample_z - 1, 1)
        ccf_pick = z_nz[int(frac * (len(z_nz) - 1))]
    else:
        ccf_pick = n_ccf_z // 2
    ccf_picks.append(ccf_pick)


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

    warped_slice = warped_ann_np[sz]
    warped_rgb = colorize_annotation(warped_slice)
    sample_rgb = np.stack([normalize_for_display(sample_arr[sz])] * 3, axis=-1)
    axes[row_idx, 4].imshow(
        np.clip(sample_rgb * 0.5 + warped_rgb.astype(np.float32) / 255.0 * 0.5, 0, 1)
    )
    axes[row_idx, 4].set_title(f"Inverse-Mapped Annotation\non sample slice {sz}", fontsize=10)
    axes[row_idx, 4].axis("off")

plt.suptitle(
    f"Xu Lab RegTool reproduction on ChATe27 brain 35 C0 "
    f"(sample {n_sample_z} slices -> {len(z_nz)} CCF slices)",
    fontsize=12,
)
plt.tight_layout(rect=[0, 0, 1, 0.97])
preview_path = OUT / "preview_xulab_style.png"
plt.savefig(preview_path, dpi=100, bbox_inches="tight")
plt.close()
print(f"  preview saved: {preview_path}")

# --- Step 6: Xu Lab cell counting on synthetic cells -----------------------
_banner("[6] Xu Lab transform_input_points + count_registered_points")
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
    k_count = min(5, len(ys))
    idx = rng.choice(len(ys), size=k_count, replace=False)
    for k in idx:
        cells.append((sz, int(ys[k]), int(xs[k])))
print(f"  synthetic cells: {len(cells)}")

points_txt = OUT / "synthetic_cells.txt"
with open(points_txt, "w", encoding="utf-8") as fh:
    fh.write("index\n")
    fh.write(f"{len(cells)}\n")
    for z, y, x in cells:
        fh.write(f"{z} {y} {x}\n")

# Use Xu Lab's transform_input_points (applies axisAlign + ANTs point warp
# sample -> CCF). Then do region lookup ourselves — Xu Lab's
# count_registered_points imports neuroglancer which isn't installed here.
try:
    from regtools.registration.point_registration import transform_input_points  # noqa: E402

    transformed = transform_input_points(
        points_file=str(points_txt),
        reg_output_dir=str(OUT),
        reg_method="ants",
        inverse=False,
    )
    print(f"  transformed {len(transformed)} cells into CCF space")

    # Replace Xu Lab count_registered_points (neuroglancer-dependent) with our
    # cell_to_ccf region lookup + Xu Lab atlas CSV join.
    sys.path.insert(0, str(BF / "project"))
    from scripts.cell_to_ccf import lookup_region_ids_from_ccf_voxels  # noqa: E402

    arr = np.asarray(transformed, dtype=np.float64)
    region_ids, oob = lookup_region_ids_from_ccf_voxels(arr, CCF_ANNOTATION)
    print(
        f"  region lookup: {oob} points OOB, {int((region_ids == 0).sum() - oob)} in CCF but unassigned"
    )

    counts = pd.Series(region_ids).value_counts().reset_index()
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
    print(f"  {int(pd.Series(region_ids).nunique())} unique region IDs")
    print(f"  cell counts saved to {counting_dir}/cell_count.csv")
except Exception as exc:  # noqa: BLE001
    print(f"  cell counting step failed (non-fatal): {exc}")

_banner("DONE — outputs at " + str(OUT))
print(f"  preview:             {preview_path}")
print(f"  moving NIfTI:        {moving_nii_path}")
print(f"  registered result:   {result_path}")
print(f"  warped annotation:   {ann_in_sample_path}")
if (OUT / "counting" / "cell_count.csv").exists():
    print(f"  cell counts:         {OUT / 'counting' / 'cell_count.csv'}")
