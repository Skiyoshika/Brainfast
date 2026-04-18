"""Build a 3-slice representative cross-section preview for a pipeline run.

Picks anterior / mid / posterior slices and renders them side-by-side so a
human reviewer can visually confirm registration quality without staring at
NCC/Dice/SSIM numbers.

Usage:
    python tools/phase_alpha_comparison_panel.py \\
        --outputs-dir project/outputs/35_C0_full_density \\
        --out panel.png \\
        --label "Phase alpha (hist_match+CLAHE)"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def _pick_three_slices(
    overlay_dir: Path,
    annotation_path: Path | None = None,
) -> list[tuple[str, Path]]:
    """Pick anterior / mid / posterior overlays.

    When *annotation_path* is provided, slice positions are chosen from the
    sub-range where the registered annotation has non-trivial coverage.
    Allen atlas only occupies ~528 AP units, so full-density 646-slice
    moving volumes have empty anterior/posterior regions at the extremes;
    picking without this guard would render blank overlays.
    """
    overlays = sorted(overlay_dir.glob("slice_*_overlay.png"))
    if not overlays:
        raise FileNotFoundError(f"No overlays in {overlay_dir}")
    n = len(overlays)
    if n < 3:
        raise ValueError(f"Need at least 3 overlays, got {n}")

    ap_start, ap_end = 0, n - 1
    if annotation_path and annotation_path.exists():
        import nibabel as nib
        import numpy as np

        data = np.asarray(nib.load(str(annotation_path)).dataobj)
        coverage = np.asarray(
            [int((data[z] != 0).sum()) for z in range(data.shape[0])]
        )
        # Only use slices with ≥20% voxel coverage.
        threshold = coverage.max() * 0.2
        valid = np.where(coverage >= threshold)[0]
        if valid.size >= 3:
            ap_start, ap_end = int(valid[0]), int(valid[-1])

    span = ap_end - ap_start
    anterior = overlays[ap_start + int(span * 0.15)]
    mid = overlays[ap_start + int(span * 0.50)]
    posterior = overlays[ap_start + int(span * 0.85)]
    return [
        (f"Anterior cortex  z={ap_start + int(span*0.15)}/{n-1}", anterior),
        (f"Mid-brain        z={ap_start + int(span*0.50)}/{n-1}", mid),
        (f"Posterior/cerebellum  z={ap_start + int(span*0.85)}/{n-1}", posterior),
    ]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--outputs-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--label", default="Registration preview")
    args = ap.parse_args()

    overlay_dir = args.outputs_dir / "truth_export"
    annotation_path = args.outputs_dir / "ants_registration" / "annotation_registered.nii.gz"
    picks = _pick_three_slices(overlay_dir, annotation_path)

    fig, axes = plt.subplots(1, 3, figsize=(24, 12))
    for ax, (title, p) in zip(axes, picks, strict=False):
        ax.imshow(mpimg.imread(p))
        ax.set_title(f"{title}\n{p.name}", fontsize=10)
        ax.axis("off")

    # Show metrics if available
    ants_metrics = args.outputs_dir / "ants_registration" / "registration_metrics.csv"
    lap_metrics = args.outputs_dir / "laplacian_refinement" / "refinement_metrics.csv"
    subtitle = args.label
    if ants_metrics.exists():
        import csv
        with ants_metrics.open() as f:
            reader = csv.DictReader(f)
            m = {r["metric"]: float(r["value"]) for r in reader}
        subtitle += f"  |  ANTs NCC={m.get('NCC', 0):.3f}  Dice={m.get('Dice', 0):.3f}  SSIM={m.get('SSIM', 0):.3f}"
    if lap_metrics.exists():
        import csv
        with lap_metrics.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        after = {r["metric"]: float(r["after"]) for r in rows}
        subtitle += f"\nAfter Laplacian: NCC={after.get('NCC', 0):.3f}  Dice={after.get('Dice', 0):.3f}  SSIM={after.get('SSIM', 0):.3f}"
    fig.suptitle(subtitle, fontsize=12, fontweight="bold")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(str(args.out), dpi=130, bbox_inches="tight")
    print(f"Panel saved: {args.out}")


if __name__ == "__main__":
    main()
