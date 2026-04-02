from __future__ import annotations

import csv
import importlib
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.metrics import structural_similarity


def _norm(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    lo = float(np.percentile(arr, 1))
    hi = float(np.percentile(arr, 99))
    scale = max(hi - lo, 1e-6)
    return np.clip((arr - lo) / scale, 0.0, 1.0).astype(np.float32)


def compute_registration_metrics(fixed_arr: np.ndarray, moving_arr: np.ndarray) -> dict[str, float]:
    fixed = _norm(fixed_arr)
    moving = _norm(moving_arr)

    fixed_mask = fixed > 0.1
    moving_mask = moving > 0.1
    mask_sum = float(fixed_mask.sum() + moving_mask.sum())
    if mask_sum == 0.0:
        dice = 1.0
    else:
        dice = float(2.0 * np.logical_and(fixed_mask, moving_mask).sum() / mask_sum)

    mse = float(np.mean((fixed - moving) ** 2))

    fixed_flat = fixed.reshape(-1)
    moving_flat = moving.reshape(-1)
    variance_floor = 1e-6
    if np.std(fixed_flat) < variance_floor or np.std(moving_flat) < variance_floor:
        ncc = 0.0
    else:
        ncc = float(np.corrcoef(fixed_flat, moving_flat)[0, 1])
        if not np.isfinite(ncc):
            ncc = 0.0

    nmi = float((fixed.mean() + moving.mean()) / max(mse + 1e-6, 1e-6))

    fixed_center = fixed[fixed.shape[0] // 2] if fixed.ndim >= 3 else fixed
    moving_center = moving[moving.shape[0] // 2] if moving.ndim >= 3 else moving
    fixed_center = np.squeeze(fixed_center)
    moving_center = np.squeeze(moving_center)
    if fixed_center.ndim != 2 or moving_center.ndim != 2:
        ssim = 0.0
    elif min(fixed_center.shape) < 3 or min(moving_center.shape) < 3:
        ssim = float(1.0 if np.allclose(fixed_center, moving_center) else 0.0)
    else:
        win_size = min(7, min(fixed_center.shape), min(moving_center.shape))
        if win_size % 2 == 0:
            win_size -= 1
        win_size = max(3, win_size)
        ssim = float(
            structural_similarity(
                fixed_center,
                moving_center,
                data_range=1.0,
                win_size=win_size,
            )
        )

    return {
        "NCC": ncc,
        "NMI": nmi,
        "SSIM": ssim,
        "Dice": dice,
        "MSE": mse,
    }


def run_ants_registration(
    fixed_path: Path,
    moving_path: Path,
    out_dir: Path,
    transform: str = "SyN",
    random_seed: int = 42,
) -> dict[str, object]:
    ants = importlib.import_module("ants")

    fixed_path = Path(fixed_path)
    moving_path = Path(moving_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_img = ants.image_read(str(fixed_path))
    moving_img = ants.image_read(str(moving_path))
    reg = ants.registration(
        fixed=fixed_img,
        moving=moving_img,
        type_of_transform=str(transform),
        random_seed=int(random_seed),
    )

    registered_volume = out_dir / "ants_result.nii.gz"
    ants.image_write(reg["warpedmovout"], str(registered_volume))

    fixed_arr = np.asarray(nib.load(str(fixed_path)).dataobj, dtype=np.float32)
    registered_arr = np.asarray(nib.load(str(registered_volume)).dataobj, dtype=np.float32)
    metrics = compute_registration_metrics(fixed_arr, registered_arr)

    metrics_csv = out_dir / "registration_metrics.csv"
    metric_order = ["NCC", "NMI", "SSIM", "Dice", "MSE"]
    with metrics_csv.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["metric", "value"])
        writer.writeheader()
        for metric in metric_order:
            writer.writerow({"metric": metric, "value": metrics[metric]})

    summary_txt = out_dir / "registration_summary.txt"
    summary_txt.write_text(
        "\n".join(
            [
                f"fixed image: {fixed_path}",
                f"moving image: {moving_path}",
                f"transform: {transform}",
                f"registered output: {registered_volume}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "registered_volume": registered_volume,
        "metrics_csv": metrics_csv,
        "summary_txt": summary_txt,
        "forward_transforms": reg.get("fwdtransforms", []),
        "inverse_transforms": reg.get("invtransforms", []),
    }
