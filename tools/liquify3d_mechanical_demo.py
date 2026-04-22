"""Mechanical end-to-end demo for Phase β 3D liquify closed loop.

Given an already-registered run (e.g. ``outputs/35_C0_full_density``), this
script:

1. Synthesises a set of landmark pairs that asks the annotation at a chosen
   z to be shifted by a known ``dx`` along axis 2.
2. POSTs the pairs (or calls the module directly) → LandmarkStore → Laplacian
   solver → writes ``annotation_refined_liquify3d.nii.gz``.
3. Re-exports the overlay PNG for the chosen z from the refined annotation.
4. Produces a side-by-side before/after PNG and prints the per-voxel
   displacement statistics so one can verify the warp actually moved things.

This does NOT claim the landmarks are anatomically correct — they are a
mechanical probe of the pipeline. Real β verification requires opening the
UI and picking actual anatomical correspondences.

Usage:
    python tools/liquify3d_mechanical_demo.py \\
        --outputs-dir project/outputs/35_C0_full_density \\
        --z 349 --shift-vx 20 --n-pairs 5 \\
        --out-png project/outputs/35_C0_full_density/liquify3d_demo.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PROJECT_ROOT = REPO_ROOT / "project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.liquify_3d import (  # noqa: E402
    LandmarkStore,
    refine_annotation_with_landmarks,
)
from scripts.truth_export_3d import export_registered_truth_slices  # noqa: E402


def _synth_pairs(annotation_shape, z: int, shift_vx: float, n_pairs: int):
    """Pick *n_pairs* landmark anchors evenly over the annotated foreground
    of slice *z* and request a uniform ``(0, +shift_vx)`` displacement.
    """
    _d, h, w = annotation_shape
    ys = np.linspace(h * 0.25, h * 0.75, n_pairs)
    xs_atlas = np.full(n_pairs, w * 0.50)
    xs_real = xs_atlas + shift_vx
    pairs = []
    for y, ax, rx in zip(ys, xs_atlas, xs_real, strict=True):
        pairs.append((int(z), float(y), float(ax), float(y), float(rx)))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs-dir", required=True, type=Path)
    ap.add_argument("--slice-dir", type=Path, help="Source fluorescence slice dir",
                    default=None)
    ap.add_argument("--slice-glob", default="z*.tif")
    ap.add_argument("--pixel-size-um", type=float, default=5.0)
    ap.add_argument("--z", type=int, default=349,
                    help="z index at which to synthesise landmarks + render overlays")
    ap.add_argument("--shift-vx", type=float, default=20.0,
                    help="synthetic displacement along axis 2 (in annotation voxels)")
    ap.add_argument("--n-pairs", type=int, default=5)
    ap.add_argument("--out-png", required=True, type=Path)
    args = ap.parse_args()

    outputs_dir = args.outputs_dir.resolve()
    ann_path = outputs_dir / "ants_registration" / "annotation_registered.nii.gz"
    assert ann_path.exists(), f"no annotation at {ann_path}"

    ann_img = nib.load(str(ann_path))
    print(f"annotation shape={ann_img.shape}")

    # 1. Write synthesised landmark pairs
    landmarks_csv = outputs_dir / "landmarks_3d_demo.csv"
    if landmarks_csv.exists():
        landmarks_csv.unlink()
    store = LandmarkStore(landmarks_csv)
    pairs = _synth_pairs(ann_img.shape, args.z, args.shift_vx, args.n_pairs)
    for z, ay, ax, ry, rx in pairs:
        store.add_pair(z=z, real=(ry, rx), atlas=(ay, ax))
    print(f"seeded {len(pairs)} landmark pairs at z={args.z}, dx={args.shift_vx} vx")

    # 2. Apply Laplacian warp
    refined_path = outputs_dir / "annotation_refined_liquify3d_demo.nii.gz"
    meta = refine_annotation_with_landmarks(
        annotation_path=ann_path,
        landmarks_csv=landmarks_csv,
        output_path=refined_path,
    )
    print(f"refine complete: {meta}")

    # 3. Re-export the overlay for this z from the refined annotation
    slice_dir = args.slice_dir or (REPO_ROOT / "project" / "data" / "35_C0_full_raw")
    real_slices = sorted(slice_dir.glob(args.slice_glob))
    if not real_slices:
        raise SystemExit(f"no slices in {slice_dir}")

    tmp_refined_dir = outputs_dir / "truth_export_liquify3d_demo"
    tmp_refined_dir.mkdir(parents=True, exist_ok=True)
    # Only bother exporting the target z (plus one neighbour to satisfy the
    # export function, which wants to walk the whole volume).  Simpler: export
    # all slices but only one is needed for the panel.
    print(f"re-exporting {len(real_slices)} overlay(s) from refined annotation...")
    export_registered_truth_slices(
        real_slice_paths=real_slices,
        annotation_volume_path=refined_path,
        out_dir=tmp_refined_dir,
        pixel_size_um=args.pixel_size_um,
        slicing_plane="coronal",
    )

    # 4. Build before/after panel
    before_png = outputs_dir / "truth_export" / f"slice_{args.z:04d}_overlay.png"
    after_png = tmp_refined_dir / f"slice_{args.z:04d}_overlay.png"
    if not before_png.exists() or not after_png.exists():
        raise SystemExit(
            f"missing overlay(s): before={before_png.exists()}, after={after_png.exists()}"
        )

    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    axes[0].imshow(mpimg.imread(before_png))
    axes[0].set_title(f"BEFORE liquify — z={args.z}\n(annotation_registered.nii.gz)", fontsize=11)
    axes[0].axis("off")
    axes[1].imshow(mpimg.imread(after_png))
    axes[1].set_title(
        f"AFTER liquify — z={args.z}\n({args.n_pairs} pairs × dx=+{args.shift_vx:.0f} voxels)",
        fontsize=11,
    )
    axes[1].axis("off")
    fig.suptitle(
        "Phase β mechanical end-to-end demo — annotation shifted by synthetic landmarks",
        fontsize=13, fontweight="bold",
    )
    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(args.out_png), dpi=130, bbox_inches="tight")
    print(f"\nbefore/after panel saved: {args.out_png}")

    # 5. Quantitative probe: compare label content around the landmark's z
    a_before = np.asarray(nib.load(str(ann_path)).dataobj, dtype=np.int64)
    a_after = np.asarray(nib.load(str(refined_path)).dataobj, dtype=np.int64)
    diff_slice = (a_before[args.z] != a_after[args.z]).sum()
    print(
        f"\nslice z={args.z}: {diff_slice} voxels changed label out of "
        f"{a_before[args.z].size} ({100*diff_slice/a_before[args.z].size:.1f}%)"
    )


if __name__ == "__main__":
    main()
