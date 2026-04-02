"""
3D volumetric registration pipeline for Brainfast using Elastix.

Pipeline:
  1. Stack TIF slices -> 3D NIfTI (downsampled to 25um)
  2. Crop Allen CCF template to matching AP range + hemisphere
  3. Elastix: Rigid + BSpline registration (template -> brain)
  4. Transformix: warp annotation into brain space
  5. Save registered annotation NIfTI for per-slice overlay

Usage (from project/ dir):
    python scripts/run_3d_registration.py --config configs/run_config_35.json
    python scripts/run_3d_registration.py --skip-to-transformix
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.paths import bootstrap_sys_path
from scripts.registration_3d_volume import (
    build_volume_from_tiffs,
    prepare_half_template_inputs,
)

PROJECT_DIR = bootstrap_sys_path()

UCI_DIR = Path("d:/UCI-ALLEN-BrainRepositoryCodeGUI-main")
ELASTIX_EXE = UCI_DIR / "elastix" / "elastix.exe"
TRANSFORMIX_EXE = UCI_DIR / "elastix" / "transformix.exe"
CCF_DATA = UCI_DIR / "CCF_DATA"
TEMPLATE_25 = CCF_DATA / "average_template_25.nii.gz"
PARAMS_RIGID = UCI_DIR / "001_parameters_Rigid.txt"
PARAMS_BSPLINE = UCI_DIR / "002_parameters_BSpline.txt"
ANNOTATION = PROJECT_DIR / "annotation_25.nii.gz"


def run_elastix(fixed, moving, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(ELASTIX_EXE),
        "-f",
        str(fixed),
        "-m",
        str(moving),
        "-out",
        str(out_dir),
        "-p",
        str(PARAMS_RIGID),
        "-p",
        str(PARAMS_BSPLINE),
        "-threads",
        "8",
    ]
    print(f"  Fixed:  {fixed.name}  {nib.load(str(fixed)).shape}")
    print(f"  Moving: {moving.name}  {nib.load(str(moving)).shape}")
    t0 = time.time()
    r = subprocess.run(cmd)
    print(f"  Done in {(time.time() - t0) / 60:.1f} min  (exit {r.returncode})")
    if r.returncode != 0:
        raise RuntimeError("Elastix failed - check elastix log above")
    return out_dir


def run_transformix(annotation, transform_params, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    # Patch to nearest-neighbor for label images
    tp = transform_params.read_text(encoding="utf-8", errors="replace")
    tp = tp.replace("(FinalBSplineInterpolationOrder 3)", "(FinalBSplineInterpolationOrder 0)")
    tp = tp.replace(
        '(ResampleInterpolator "FinalBSplineInterpolator")',
        '(ResampleInterpolator "FinalNearestNeighborInterpolator")',
    )
    patched = out_dir / "TransformParameters_NN.txt"
    patched.write_text(tp, encoding="utf-8")
    cmd = [
        str(TRANSFORMIX_EXE),
        "-in",
        str(annotation),
        "-out",
        str(out_dir),
        "-tp",
        str(patched),
        "-threads",
        "8",
    ]
    t0 = time.time()
    r = subprocess.run(cmd)
    print(f"  Transformix done in {time.time() - t0:.1f}s  (exit {r.returncode})")
    if r.returncode != 0:
        raise RuntimeError("Transformix failed")
    for name in ["result.nii", "result.nii.gz", "result.mhd"]:
        p = out_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Transformix output not found in {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/run_config_35.json")
    ap.add_argument("--skip-to-transformix", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = json.loads((PROJECT_DIR / args.config).read_text())
    reg = cfg["registration"]
    inp = cfg["input"]

    slice_dir = PROJECT_DIR / inp["slice_dir"]
    pixel_um_xy = float(inp["pixel_size_um_xy"])
    z_spacing = float(inp["slice_spacing_um"])
    hemisphere = str(reg.get("atlas_hemisphere", "right_flipped"))
    z_scale = float(reg.get("atlas_z_z_scale", 0.2))
    z_offset = int(reg.get("atlas_z_offset", 150))
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_DIR / "outputs" / "registration_3d"

    print("=" * 60)
    print("Brainfast 3D Registration (Elastix)")
    print("=" * 60)

    # 1. Brain NIfTI
    brain_nii = out_dir / "brain_25um.nii.gz"
    if not brain_nii.exists():
        print("\n[1/4] Stacking TIFs -> NIfTI...")
        brain_meta = build_volume_from_tiffs(slice_dir, brain_nii, pixel_um_xy, z_spacing)
        brain_shape = tuple(brain_meta["shape"])
        print(
            f"  {brain_meta['slice_count']} slices, XY downsample x{brain_meta['downsample_factor']}"
            f" -> {brain_meta['voxel_mm'][1] * 1000:.0f}um/px"
        )
        print(f"  Volume shape: {brain_shape}")
        mb = brain_nii.stat().st_size // 1024 // 1024
        print(f"  Saved -> {brain_nii}  ({mb} MB)")
    else:
        brain_shape = nib.load(str(brain_nii)).shape
        print(f"\n[1/4] Brain NIfTI exists: {brain_nii}  shape={brain_shape}")

    # Compute AP range from filenames
    slices = sorted(slice_dir.glob("z*.tif"))
    z_first = int(slices[0].stem[1:])
    z_last = int(slices[-1].stem[1:])
    ap_start = max(0, int(z_first * z_scale) + z_offset - 5)
    ap_end = min(528, int(z_last * z_scale) + z_offset + 5)
    print(f"  AP range in atlas: [{ap_start}, {ap_end}]  ({ap_end - ap_start} slices)")

    # 2. Cropped template + annotation
    print(f"\n[2/4] Cropping template + annotation to AP range + {hemisphere}...")
    half_meta = prepare_half_template_inputs(
        TEMPLATE_25, ANNOTATION, hemisphere, ap_start, ap_end, out_dir
    )
    tmpl_cropped = half_meta["template_path"]
    ann_cropped = half_meta["annotation_path"]
    print(f"  Saved template_half.nii.gz  shape={tuple(half_meta['shape'])}")
    print(f"  Saved annotation_half.nii.gz  shape={tuple(half_meta['shape'])}")

    # 3. Elastix
    elastix_out = out_dir / "elastix"
    tp_file = elastix_out / "TransformParameters.1.txt"
    if not args.skip_to_transformix:
        print("\n[3/4] Running Elastix (Rigid + BSpline) ...")
        print("  This takes ~10-15 minutes...")
        run_elastix(tmpl_cropped, brain_nii, elastix_out)
    else:
        print(f"\n[3/4] Skipping Elastix (existing: {tp_file})")
        if not tp_file.exists():
            raise FileNotFoundError(f"No transform at: {tp_file}")

    # 4. Transformix on annotation
    print("\n[4/4] Warping annotation into brain space...")
    transformix_out = out_dir / "registered_annotation"
    result_file = run_transformix(ann_cropped, tp_file, transformix_out)

    final = out_dir / "annotation_registered.nii.gz"
    # Re-save with nibabel to ensure proper gzip compression
    img = nib.load(str(result_file))
    nib.save(img, str(final))
    print(f"\n{'=' * 60}")
    print(f"DONE! Registered annotation -> {final}")
    print("Run test_single_slice.py to view overlay.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
