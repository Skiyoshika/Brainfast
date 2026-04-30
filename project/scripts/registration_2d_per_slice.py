"""2D per-slice registration against CCF template slices.

Brainfast-native port of the core algorithm behind Xu Lab's
``registration_2d`` + ``registration_batch_2d``
(regtools/registration/registration_2d.py). For workflows where the user
has independently mounted coronal sections mapped to specific CCF atlas
slice indices (the typical histology use case), this avoids the 3D whole-
brain warp entirely and registers each sample section directly to its
target CCF slice in 2D.

Key differences from Brainfast's existing 3D whole-brain pipeline:

  * **Per-slice transforms.** Each sample slice gets its own 2D ANTs
    transform (warp + affine). No cross-slice coupling. Works well when
    mounting produces independent deformations per slice.
  * **No Z warp.** The slice-to-CCF-index mapping is provided explicitly
    by the user (CSV with ``input_file, template_index`` columns). No
    midline-fissure Z alignment needed.
  * **Cross-modality safe.** Mattes MI metric by default (same rationale
    as 3D: Xu Lab's default CC fails on fluorescence-vs-Nissl in our env).

Input: either a single multi-page TIFF (one page per section) or a
directory of 2D images + a slice-mapping CSV.

Output: one per-slice sub-directory under ``output_dir`` holding the
moving slice warped into its CCF target plus ANTs transforms. A batch
summary CSV at the root.
"""

from __future__ import annotations

import csv
import gc
import shutil
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

try:
    from scripts.logging_setup import get_logger
except ImportError:  # pragma: no cover
    from logging_setup import get_logger

log = get_logger(__name__)


def _import_ants():
    # ants.plotting import is broken in some envs — stub before importing.
    sys.modules.setdefault("ants.plotting", type(sys)("ants.plotting"))
    try:
        import matplotlib._docstring as _mpl_ds

        if not hasattr(_mpl_ds, "dedent_interpd"):
            _mpl_ds.dedent_interpd = lambda f: f
    except ImportError:
        pass
    import ants  # noqa: F401

    return ants


def _extract_ccf_slice(ccf_template_path: Path | str, ccf_slice_index: int, out_path: Path) -> Path:
    """Extract a single coronal slice from the 3D CCF template as a 2D NIfTI.

    The saved 2D affine takes the in-plane direction cosines from the 3D
    affine's columns 1 and 2 (DV and ML for PIR-oriented CCF) and fills the
    third column with a unit vector orthogonal to those two (computed via
    cross product) so the 3×3 direction part stays non-singular. This keeps
    nibabel/ANTs happy when reading the saved NIfTI.
    """
    img = nib.load(str(ccf_template_path))
    data = np.asarray(img.dataobj)
    if not (0 <= int(ccf_slice_index) < data.shape[0]):
        raise IndexError(
            f"ccf_slice_index {ccf_slice_index} out of range for CCF shape {data.shape}"
        )
    slice2d = data[int(ccf_slice_index)].astype(np.float32)
    full_affine = img.affine
    col0 = full_affine[:3, 1]
    col1 = full_affine[:3, 2]
    # Orthogonal unit vector for the out-of-plane axis — spacing = 1
    perp = np.cross(col0, col1)
    norm = np.linalg.norm(perp)
    if norm > 1e-12:
        perp = perp / norm
    else:
        perp = np.array([0.0, 0.0, 1.0])
    affine2d = np.eye(4, dtype=np.float64)
    affine2d[:3, 0] = col0
    affine2d[:3, 1] = col1
    affine2d[:3, 2] = perp
    affine2d[:3, 3] = full_affine[:3, 3] + full_affine[:3, 0] * float(ccf_slice_index)
    out_img = nib.Nifti1Image(slice2d, affine2d)
    out_img.header["qform_code"] = 1
    nib.save(out_img, str(out_path))
    return out_path


def _read_moving_slice(path: Path, orient: bool = True) -> np.ndarray:
    """Read a 2D slice — TIFF (.tif/.tiff), NIfTI (.nii/.nii.gz), or PNG.

    If ``orient`` is True, applies the same transpose+flip as Xu Lab's
    ``_orient_section`` to match the CCF reference frame.
    """
    suffix = "".join(path.suffixes).lower()
    if suffix in (".tif", ".tiff"):
        from tifffile import imread

        arr = imread(str(path))
        if arr.ndim == 3:
            arr = arr[0]
    elif suffix.endswith(".nii") or suffix.endswith(".nii.gz"):
        img = nib.load(str(path))
        arr = np.asarray(img.dataobj)
        if arr.ndim == 3:
            arr = arr[..., 0] if arr.shape[2] < 5 else arr[arr.shape[0] // 2]
    elif suffix in (".png", ".jpg", ".jpeg"):
        from tifffile import imread

        arr = imread(str(path))
        if arr.ndim == 3:
            arr = arr[..., 0]
    else:
        raise ValueError(f"unsupported slice format: {path}")
    arr = np.asarray(arr, dtype=np.float32)
    if orient:
        arr = np.where(arr < 0, 0, arr)
        arr = arr.T
        arr = np.flip(arr, axis=0)
        arr = np.flip(arr, axis=1)
    return arr


def register_slice_to_ccf(
    moving_slice_path: Path | str,
    *,
    ccf_template_path: Path | str,
    ccf_slice_index: int,
    output_dir: Path | str,
    type_of_transform: str = "SyNRA",
    aff_metric: str = "mattes",
    syn_metric: str | None = "mattes",
    reg_iterations: tuple[int, ...] | None = None,
    random_seed: int = 42,
    orient_moving: bool = True,
) -> dict:
    """Register a single 2D moving slice to a single CCF coronal slice.

    Writes ANTs transforms (warp + affine + inv_warp) and the registered
    output image into ``output_dir``. Returns a dict with the transform
    paths, the registered-image path, and registration metadata.

    ``type_of_transform`` defaults to ``SyNRA`` (Rigid + Affine + SyN) with
    Mattes MI on both stages — matches Brainfast's cross-modality-safe
    defaults. Pass ``type_of_transform='Affine'`` if SyN isn't needed.

    Parameters mirror Xu Lab's ``registration_2d`` for the essentials;
    details like ``rotate_angle`` / ``flip_h`` / ``flip_v`` pre-orient
    options aren't exposed (pre-orient via ``orient_moving`` or feed an
    already-oriented slice in).
    """
    ants = _import_ants()

    moving_slice_path = Path(moving_slice_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract the target CCF slice
    ccf_slice_nifti = output_dir / f"ccf_slice_{int(ccf_slice_index):04d}.nii.gz"
    _extract_ccf_slice(ccf_template_path, ccf_slice_index, ccf_slice_nifti)

    # Load + orient moving slice
    moving_arr = _read_moving_slice(moving_slice_path, orient=orient_moving)
    # Save a NIfTI copy matching the CCF slice's affine spacing (in-plane only)
    ccf_slice_img = nib.load(str(ccf_slice_nifti))
    moving_nii_path = output_dir / "moving_slice.nii.gz"
    moving_img = nib.Nifti1Image(moving_arr.astype(np.float32), ccf_slice_img.affine)
    moving_img.header["qform_code"] = 1
    nib.save(moving_img, str(moving_nii_path))

    fixed_ants = ants.image_read(str(ccf_slice_nifti))
    moving_ants = ants.image_read(str(moving_nii_path))

    reg_kwargs = dict(
        fixed=fixed_ants,
        moving=moving_ants,
        type_of_transform=type_of_transform,
        aff_metric=aff_metric,
        random_seed=int(random_seed),
        verbose=False,
    )
    if syn_metric is not None:
        reg_kwargs["syn_metric"] = syn_metric
    if reg_iterations is not None:
        reg_kwargs["reg_iterations"] = tuple(int(v) for v in reg_iterations)

    t0 = time.time()
    reg = ants.registration(**reg_kwargs)
    elapsed = time.time() - t0

    # Save transforms + result in Brainfast layout
    ants_dir = output_dir / "ants_registration"
    ants_dir.mkdir(exist_ok=True)

    def _suffix(p: str) -> str:
        return ".nii.gz" if p.endswith(".nii.gz") else Path(p).suffix

    fwd: list[str] = []
    inv: list[str] = []
    for i, tf in enumerate(reg.get("fwdtransforms", [])):
        dst = ants_dir / f"fwd_transform_{i}{_suffix(tf)}"
        shutil.copy2(tf, dst)
        fwd.append(str(dst))
    for i, tf in enumerate(reg.get("invtransforms", [])):
        dst = ants_dir / f"inv_transform_{i}{_suffix(tf)}"
        shutil.copy2(tf, dst)
        inv.append(str(dst))

    result_path = ants_dir / "ants_result.nii.gz"
    ants.image_write(reg["warpedmovout"], str(result_path))
    del fixed_ants, moving_ants, reg
    gc.collect()

    return {
        "moving_slice": str(moving_slice_path),
        "ccf_slice_index": int(ccf_slice_index),
        "ccf_slice_nifti": str(ccf_slice_nifti),
        "moving_nifti": str(moving_nii_path),
        "registered": str(result_path),
        "forward_transforms": fwd,
        "inverse_transforms": inv,
        "elapsed_seconds": float(elapsed),
        "transform_used": type_of_transform,
    }


def register_batch_2d_to_ccf(
    slice_mapping_csv: Path | str,
    *,
    ccf_template_path: Path | str,
    moving_dir: Path | str,
    output_dir: Path | str,
    type_of_transform: str = "SyNRA",
    aff_metric: str = "mattes",
    syn_metric: str | None = "mattes",
    reg_iterations: tuple[int, ...] | None = None,
    random_seed: int = 42,
    orient_moving: bool = True,
) -> dict:
    """Iterate over a ``(input_file, template_index)`` CSV and register each
    sample slice to its target CCF slice.

    Mirrors Xu Lab's ``registration_batch_2d`` but as a pure-data Brainfast
    port. Writes per-slice sub-directories ``slice_{i:04d}_idx{ccf}/`` under
    ``output_dir`` and a summary CSV ``batch_2d_summary.csv`` at the root.

    Parameters
    ----------
    slice_mapping_csv
        CSV with columns ``input_file`` (relative to ``moving_dir``) and
        ``template_index`` (CCF coronal slice index).
    ccf_template_path
        3D CCF template NIfTI.
    moving_dir
        Directory containing sample slice files named in the mapping CSV.
    output_dir
        Destination root for per-slice outputs.

    Returns
    -------
    dict with keys ``entries`` (list of per-slice result dicts), ``summary_csv``.
    """
    mapping_csv = Path(slice_mapping_csv)
    moving_dir = Path(moving_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(mapping_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"empty mapping CSV: {mapping_csv}")

    entries: list[dict] = []
    for i, row in enumerate(rows):
        input_file = str(row["input_file"]).strip()
        try:
            template_index = int(row["template_index"])
        except (KeyError, ValueError) as e:
            entries.append(
                {"row": i, "input_file": input_file, "status": "error", "reason": str(e)}
            )
            continue
        input_path = moving_dir / input_file
        if not input_path.is_absolute():
            input_path = (moving_dir / input_file).resolve()
        if not input_path.exists():
            entries.append(
                {
                    "row": i,
                    "input_file": input_file,
                    "template_index": template_index,
                    "status": "skipped",
                    "reason": "file_not_found",
                }
            )
            continue

        slice_out = output_dir / f"slice_{i:04d}_idx{template_index:04d}"
        try:
            result = register_slice_to_ccf(
                input_path,
                ccf_template_path=ccf_template_path,
                ccf_slice_index=template_index,
                output_dir=slice_out,
                type_of_transform=type_of_transform,
                aff_metric=aff_metric,
                syn_metric=syn_metric,
                reg_iterations=reg_iterations,
                random_seed=random_seed,
                orient_moving=orient_moving,
            )
            result["status"] = "success"
            result["row"] = i
            result["input_file"] = input_file
            entries.append(result)
            log.info(
                "2D batch %d/%d: %s -> CCF slice %d OK (%.1fs)",
                i + 1,
                len(rows),
                input_file,
                template_index,
                result.get("elapsed_seconds", 0.0),
            )
        except Exception as exc:  # noqa: BLE001 — per-slice fault tolerance
            entries.append(
                {
                    "row": i,
                    "input_file": input_file,
                    "template_index": template_index,
                    "status": "error",
                    "reason": str(exc),
                }
            )
            log.warning("2D batch %d/%d %s failed: %s", i + 1, len(rows), input_file, exc)

    summary_csv = output_dir / "batch_2d_summary.csv"
    fields = [
        "row",
        "input_file",
        "template_index",
        "status",
        "reason",
        "registered",
        "elapsed_seconds",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for entry in entries:
            writer.writerow({k: entry.get(k, "") for k in fields})

    return {"entries": entries, "summary_csv": str(summary_csv)}
