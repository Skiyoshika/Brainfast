"""Migrate stale RAS-diagonal ``input_volume.nii.gz`` affines to PIR orientation.

Before commit ``73b9215`` (2026-04-18), Brainfast's volume-build code wrote a
pure-diagonal affine for input volumes:

    affine = np.diag([z_spacing, y_spacing, x_spacing, 1])
    # nib.aff2axcodes(affine) → ('R', 'A', 'S')

This mislabelled the sample's AP-slice-stack axis (axis 0, N slices) as the
physical R (lateral) direction. ANTs respects affine metadata, so it tried to
fit the sample's "lateral axis" (actually AP) against CCF's R axis (11400 µm
half-brain extent), compressing the sample's physical AP extent (~2.8 mm for
sample 35) into a tiny CCF slice range. Result: 3:1 Z compression, 34% leaf
region loss, cells bunched into a single AP band.

Commit ``73b9215`` switched volume writers to the correct PIR off-diagonal
affine (matching CCF's ``(P, I, R)`` axis codes), but any ``input_volume.nii.gz``
file produced before that commit is still stale on disk.

This utility walks a pipeline outputs directory, detects stale files (RAS
diagonal affine with Brainfast's characteristic zooms), and patches each file
in-place by rewriting only the affine (voxel data untouched). An ``.orig``
backup is kept so the patch is reversible.

Usage:
    # Dry run (list what would change)
    python -m scripts.migrate_volume_affine project/outputs

    # Apply the patch
    python -m scripts.migrate_volume_affine project/outputs --apply
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import nibabel as nib
import numpy as np

try:
    from scripts.logging_setup import get_logger
except ImportError:  # pragma: no cover
    from logging_setup import get_logger

log = get_logger(__name__)


def build_pir_affine(voxel_mm: tuple[float, float, float]) -> np.ndarray:
    """Construct the PIR off-diagonal affine matching CCF orientation.

    ``voxel_mm`` = ``(z_mm, y_mm, x_mm)`` for data axes (z = slice stack,
    y = image rows, x = image cols). Mirrors
    ``registration_3d_volume.build_volume_from_tiffs`` and
    ``volume_io._make_brainfast_affine``.
    """
    z_mm, y_mm, x_mm = (float(voxel_mm[0]), float(voxel_mm[1]), float(voxel_mm[2]))
    return np.array(
        [
            [0.0, 0.0, x_mm, 0.0],
            [-z_mm, 0.0, 0.0, 0.0],
            [0.0, -y_mm, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def is_ras_diagonal(affine: np.ndarray, tol: float = 1e-6) -> bool:
    """Detect the pre-73b9215 buggy diagonal affine.

    Returns True iff the 3×3 direction part is a pure-positive diagonal
    (i.e. axis codes would be ('R', 'A', 'S') with Brainfast's typical
    millimetre-scale spacings).
    """
    A = np.asarray(affine, dtype=np.float64)[:3, :3]
    if not np.allclose(A, np.diag(np.diagonal(A)), atol=tol):
        return False
    # All diagonal entries positive and ≤ 1 mm (Brainfast spacings are 0.01–0.1 mm)
    diag = np.diagonal(A)
    if not np.all(diag > 0):
        return False
    if not np.all(diag < 1.0):
        return False
    return True


def patch_affine_inplace(nifti_path: Path, *, apply: bool) -> dict:
    """Read ``nifti_path``, replace its affine with the PIR form, write back.

    Keeps a ``.orig`` backup next to the file before overwriting.
    Returns a status dict with ``before``, ``after``, ``patched`` fields.
    """
    img = nib.load(str(nifti_path))
    before = nib.aff2axcodes(img.affine)
    zooms = img.header.get_zooms()[:3]
    if not is_ras_diagonal(img.affine):
        return {
            "path": str(nifti_path),
            "before_codes": before,
            "after_codes": before,
            "zooms": tuple(float(z) for z in zooms),
            "patched": False,
            "reason": "already_correct_or_different_buggy_pattern",
        }

    pir = build_pir_affine(tuple(float(z) for z in zooms))
    after = nib.aff2axcodes(pir)

    if apply:
        # Backup with ``.ras_backup.nii.gz`` suffix so nibabel can still load it
        # (``.nii.gz.orig`` would confuse readers relying on extension sniffing).
        name = nifti_path.name
        if name.endswith(".nii.gz"):
            backup_name = name[: -len(".nii.gz")] + ".ras_backup.nii.gz"
        elif name.endswith(".nii"):
            backup_name = name[: -len(".nii")] + ".ras_backup.nii"
        else:
            backup_name = name + ".ras_backup"
        backup = nifti_path.with_name(backup_name)
        if not backup.exists():
            shutil.copy2(nifti_path, backup)
        data = np.asarray(img.dataobj)
        new_img = nib.Nifti1Image(data, pir, header=img.header.copy())
        new_img.set_qform(pir, code=1)
        new_img.set_sform(pir, code=1)
        new_img.header.set_zooms(tuple(float(z) for z in zooms))
        nib.save(new_img, str(nifti_path))

    return {
        "path": str(nifti_path),
        "before_codes": before,
        "after_codes": after,
        "zooms": tuple(float(z) for z in zooms),
        "patched": bool(apply),
        "reason": "buggy_ras_diagonal_patched" if apply else "buggy_ras_diagonal_detected",
    }


def find_stale_volumes(pipeline_outputs_root: Path) -> list[Path]:
    """Locate ``input_volume.nii.gz`` files under a pipeline outputs tree."""
    root = Path(pipeline_outputs_root)
    return sorted(root.rglob("input_volume.nii.gz"))


def migrate_outputs_root(pipeline_outputs_root: Path | str, *, apply: bool = False) -> list[dict]:
    """Walk a pipeline outputs tree, patching every stale input_volume affine.

    With ``apply=False`` (default), only prints what would change. Pass
    ``apply=True`` to actually rewrite files.
    """
    results: list[dict] = []
    for candidate in find_stale_volumes(Path(pipeline_outputs_root)):
        status = patch_affine_inplace(candidate, apply=apply)
        results.append(status)
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "root",
        type=Path,
        help="Pipeline outputs directory (searched recursively for input_volume.nii.gz)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Actually rewrite files. Without this flag, only reports what would change.",
    )
    args = parser.parse_args()

    results = migrate_outputs_root(args.root, apply=args.apply)
    patched = sum(1 for r in results if r["patched"])
    detected = sum(1 for r in results if r["reason"] == "buggy_ras_diagonal_detected")
    clean = sum(1 for r in results if r["reason"] == "already_correct_or_different_buggy_pattern")
    print(f"Scanned {len(results)} input_volume.nii.gz files under {args.root}")
    print(f"  Already correct or unrelated: {clean}")
    if args.apply:
        print(f"  Patched: {patched}")
    else:
        print(f"  Would patch (buggy RAS diagonal detected): {detected}")
    for r in results:
        if r["reason"].startswith("buggy_ras_diagonal"):
            arrow = "→" if args.apply else "…"
            print(
                f"  {arrow} {r['path']}: {r['before_codes']} → {r['after_codes']} (zooms={r['zooms']})"
            )


if __name__ == "__main__":  # pragma: no cover
    main()
