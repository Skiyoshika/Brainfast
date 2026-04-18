"""Re-export truth overlay slices from a pipeline run's existing ANTs +
Laplacian artifacts.

Use case: a previous pipeline run was killed during truth_export (or before
quantification). Rather than re-running ANTs + Laplacian (slow), reuse the
on-disk annotation_refined.nii.gz and regenerate the slice-level overlays.

Usage:
    python tools/reexport_truth_slices.py \\
        --outputs-dir project/outputs/35_C0_full_density \\
        --slice-dir   project/data/35_C0_full_raw \\
        --slice-glob "z*.tif" \\
        --pixel-size-um 5.0
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PROJECT_ROOT = REPO_ROOT / "project"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.truth_export_3d import export_registered_truth_slices  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs-dir", required=True, type=Path)
    ap.add_argument("--slice-dir", required=True, type=Path)
    ap.add_argument("--slice-glob", default="z*.tif")
    ap.add_argument("--pixel-size-um", type=float, default=5.0)
    ap.add_argument("--slicing-plane", default="coronal")
    ap.add_argument("--atlas-hemisphere", default="right_flipped")
    args = ap.parse_args()

    outputs_dir = args.outputs_dir.resolve()
    slice_dir = args.slice_dir.resolve()

    # Use the annotation warped back into moving-volume space. The
    # ants_registration/annotation_registered.nii.gz is already in moving
    # space (shape matches input_volume); laplacian_refinement/
    # annotation_refined.nii.gz is still in template half space and requires
    # a second warp-back pass that the full pipeline does later.
    ann_ants = outputs_dir / "ants_registration" / "annotation_registered.nii.gz"
    if not ann_ants.exists():
        raise FileNotFoundError(
            f"No moving-space annotation at {ann_ants}. Run the full pipeline "
            "at least through the ANTs stage first."
        )
    annotation_path = ann_ants

    real_slices = sorted(slice_dir.glob(args.slice_glob))
    if not real_slices:
        raise FileNotFoundError(f"no slices matched {args.slice_glob} in {slice_dir}")

    truth_dir = outputs_dir / "truth_export"
    truth_dir.mkdir(parents=True, exist_ok=True)
    print(f"Re-exporting truth from {annotation_path.name} → {truth_dir}")
    print(f"  slices: {len(real_slices)}  pixel: {args.pixel_size_um} um  plane: {args.slicing_plane}")

    rows = export_registered_truth_slices(
        real_slice_paths=real_slices,
        annotation_volume_path=annotation_path,
        out_dir=truth_dir,
        pixel_size_um=args.pixel_size_um,
        slicing_plane=args.slicing_plane,
        atlas_hemisphere=args.atlas_hemisphere,
    )
    print(f"Exported {len(rows)} truth overlays.")


if __name__ == "__main__":
    main()
