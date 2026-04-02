from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, laplace

from project.scripts.registration_3d_ants import compute_registration_metrics


def refine_registered_volume(
    fixed_path: Path,
    moving_path: Path,
    out_dir: Path,
    iterations: int = 100,
    lambda_: float = 0.18,
) -> dict[str, Path]:
    fixed_path = Path(fixed_path)
    moving_path = Path(moving_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_img = nib.load(str(fixed_path))
    moving_img = nib.load(str(moving_path))
    fixed = np.asarray(fixed_img.dataobj, dtype=np.float32)
    moving = np.asarray(moving_img.dataobj, dtype=np.float32)
    if fixed.ndim != 3 or moving.ndim != 3:
        raise ValueError("fixed and moving volumes must both be 3D")
    if fixed.shape != moving.shape:
        raise ValueError("fixed and moving volumes must have the same shape")
    refined = moving.copy()

    for _ in range(int(iterations)):
        residual = fixed - refined
        refined = refined + float(lambda_) * laplace(gaussian_filter(residual, sigma=0.8))

    disp = gaussian_filter(fixed - refined, sigma=1.2).astype(np.float32)
    field = np.stack(np.gradient(disp), axis=0).astype(np.float32)

    final_registered_path = out_dir / "final_registered.nii.gz"
    nib.save(
        nib.Nifti1Image(refined.astype(np.float32), moving_img.affine, moving_img.header),
        str(final_registered_path),
    )

    field_path = out_dir / "laplacian_deformation_field.npy"
    np.save(str(field_path), field)

    before = compute_registration_metrics(fixed, moving)
    after = compute_registration_metrics(fixed, refined)

    metrics_csv = out_dir / "refinement_metrics.csv"
    metric_order = list(before.keys())
    with metrics_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric", "before", "after", "change"])
        writer.writeheader()
        for metric in metric_order:
            before_value = float(before[metric])
            after_value = float(after[metric])
            writer.writerow(
                {
                    "metric": metric,
                    "before": before_value,
                    "after": after_value,
                    "change": after_value - before_value,
                }
            )

    return {
        "final_registered_path": final_registered_path,
        "field_path": field_path,
        "metrics_csv": metrics_csv,
    }
