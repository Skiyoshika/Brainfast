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


def _tissue_mask(arr: np.ndarray, threshold: float = 0.1) -> np.ndarray:
    """Binary mask of tissue (non-background) voxels."""
    return arr > threshold


def _nmi_histogram(fixed: np.ndarray, moving: np.ndarray, bins: int = 64) -> float:
    """Normalized Mutual Information via joint histogram.

    NMI = (H(f) + H(m)) / H(f,m)
    Range: 1.0 (independent) to 2.0 (perfect alignment).
    """
    # Use tissue union mask to exclude background-background pairs
    mask = _tissue_mask(fixed) | _tissue_mask(moving)
    if mask.sum() < 100:
        return 1.0
    f = fixed[mask].ravel()
    m = moving[mask].ravel()
    hist2d, _, _ = np.histogram2d(f, m, bins=bins, range=[[0, 1], [0, 1]])
    # Normalize to joint probability
    pxy = hist2d / max(float(hist2d.sum()), 1.0)
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    # Entropies (avoid log(0))
    eps = 1e-10
    hx = -float(np.sum(px[px > eps] * np.log(px[px > eps])))
    hy = -float(np.sum(py[py > eps] * np.log(py[py > eps])))
    hxy = -float(np.sum(pxy[pxy > eps] * np.log(pxy[pxy > eps])))
    if hxy < eps:
        return 1.0
    return float((hx + hy) / hxy)


def compute_registration_metrics(fixed_arr: np.ndarray, moving_arr: np.ndarray) -> dict[str, float]:
    fixed = _norm(fixed_arr)
    moving = _norm(moving_arr)

    # Tissue masks for foreground-only evaluation
    fixed_mask = _tissue_mask(fixed)
    moving_mask = _tissue_mask(moving)
    tissue_union = fixed_mask | moving_mask

    # Dice on tissue foreground
    mask_sum = float(fixed_mask.sum() + moving_mask.sum())
    if mask_sum == 0.0:
        dice = 1.0
    else:
        dice = float(2.0 * np.logical_and(fixed_mask, moving_mask).sum() / mask_sum)

    # MSE on tissue union (exclude background-background pairs)
    if tissue_union.sum() == 0:
        mse = 0.0
    else:
        mse = float(np.mean((fixed[tissue_union] - moving[tissue_union]) ** 2))

    # NCC on tissue union only (avoids inflated correlation from shared zeros)
    if tissue_union.sum() < 100:
        ncc = 0.0
    else:
        f_masked = fixed[tissue_union].ravel()
        m_masked = moving[tissue_union].ravel()
        if np.std(f_masked) < 1e-6 or np.std(m_masked) < 1e-6:
            ncc = 0.0
        else:
            ncc = float(np.corrcoef(f_masked, m_masked)[0, 1])
            if not np.isfinite(ncc):
                ncc = 0.0

    # Proper NMI via joint histogram
    nmi = _nmi_histogram(fixed, moving)

    # PSNR
    psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-10))) if mse > 0 else 60.0

    # SSIM on center slice
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
        "PSNR": psnr,
    }


def run_ants_registration(
    fixed_path: Path,
    moving_path: Path,
    out_dir: Path,
    transform: str = "SyN",
    random_seed: int = 42,
) -> dict[str, object]:
    # Patch matplotlib compatibility for ANTsPy (matplotlib >=3.10 removed dedent_interpd)
    try:
        import matplotlib._docstring as _mpl_ds

        if not hasattr(_mpl_ds, "dedent_interpd"):
            _mpl_ds.dedent_interpd = lambda func: func
    except Exception:
        pass
    ants = importlib.import_module("ants")

    fixed_path = Path(fixed_path)
    moving_path = Path(moving_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fixed_img = ants.image_read(str(fixed_path))
    moving_img = ants.image_read(str(moving_path))

    # Use SyNRA (SyN + Rigid + Affine) with brain-optimised parameters.
    # For cross-modality registration (fluorescence vs Nissl), Mattes MI
    # is far more robust than CC (which assumes linear intensity relationship).
    if transform.lower() in ("syn", "synra"):
        import logging as _alog

        _alog.getLogger(__name__).info(
            "Running ANTs SyNRA with Mattes MI metric (cross-modality optimised)"
        )
        reg = ants.registration(
            fixed=fixed_img,
            moving=moving_img,
            type_of_transform="SyNRA",
            aff_metric="mattes",
            syn_metric="mattes",
            syn_sampling=32,
            reg_iterations=(200, 200, 100, 50),
            random_seed=int(random_seed),
            verbose=False,
        )
    else:
        reg = ants.registration(
            fixed=fixed_img,
            moving=moving_img,
            type_of_transform=str(transform),
            random_seed=int(random_seed),
        )

    registered_volume = out_dir / "ants_result.nii.gz"
    ants.image_write(reg["warpedmovout"], str(registered_volume))

    # Persist transform files so they survive beyond the temp directory lifetime.
    # NIfTI warp fields end in .nii.gz — Path.suffix only returns .gz, so we
    # must check for the double extension explicitly.
    # Transform persistence is best-effort: if a temp file was already cleaned
    # up we keep the original path in the metadata rather than crashing after
    # a successful registration.
    import logging
    import shutil

    _log = logging.getLogger(__name__)

    def _full_suffix(p: str) -> str:
        """Return .nii.gz or single suffix like .mat."""
        s = Path(p).name
        if s.endswith(".nii.gz"):
            return ".nii.gz"
        return Path(p).suffix

    saved_fwd, saved_inv = [], []
    for i, tf in enumerate(reg.get("fwdtransforms", [])):
        dst = out_dir / f"fwd_transform_{i}{_full_suffix(tf)}"
        try:
            shutil.copy2(tf, dst)
            saved_fwd.append(str(dst))
        except (OSError, FileNotFoundError) as exc:
            _log.warning("Could not persist forward transform %s: %s", tf, exc)
            saved_fwd.append(str(tf))
    for i, tf in enumerate(reg.get("invtransforms", [])):
        dst = out_dir / f"inv_transform_{i}{_full_suffix(tf)}"
        try:
            shutil.copy2(tf, dst)
            saved_inv.append(str(dst))
        except (OSError, FileNotFoundError) as exc:
            _log.warning("Could not persist inverse transform %s: %s", tf, exc)
            saved_inv.append(str(tf))

    fixed_arr = np.asarray(nib.load(str(fixed_path)).dataobj, dtype=np.float32)
    registered_arr = np.asarray(nib.load(str(registered_volume)).dataobj, dtype=np.float32)
    metrics = compute_registration_metrics(fixed_arr, registered_arr)

    metrics_csv = out_dir / "registration_metrics.csv"
    metric_order = ["NCC", "NMI", "SSIM", "Dice", "MSE", "PSNR"]
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
        "forward_transforms": saved_fwd or reg.get("fwdtransforms", []),
        "inverse_transforms": saved_inv or reg.get("invtransforms", []),
    }
