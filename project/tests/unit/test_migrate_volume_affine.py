"""Unit tests for migrate_volume_affine — the pre-73b9215 stale-affine fixer."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from project.scripts.migrate_volume_affine import (
    build_pir_affine,
    is_ras_diagonal,
    migrate_outputs_root,
    patch_affine_inplace,
)


def test_build_pir_affine_matches_ccf_convention():
    """The constructed affine should report axis codes ('P', 'I', 'R')."""
    aff = build_pir_affine((0.025, 0.02, 0.02))
    assert nib.aff2axcodes(aff) == ("P", "I", "R")


def test_build_pir_affine_direction_cosines():
    """Columns of the 3×3 match build_volume_from_tiffs in registration_3d_volume.py."""
    aff = build_pir_affine((0.025, 0.02, 0.02))
    np.testing.assert_allclose(aff[:3, 0], [0.0, -0.025, 0.0])   # axis 0 → -Y (P)
    np.testing.assert_allclose(aff[:3, 1], [0.0, 0.0, -0.02])    # axis 1 → -Z (I)
    np.testing.assert_allclose(aff[:3, 2], [0.02, 0.0, 0.0])     # axis 2 → +X (R)


@pytest.mark.parametrize(
    "affine,expected",
    [
        (np.diag([0.025, 0.02, 0.02, 1.0]), True),
        (np.diag([0.025, 0.025, 0.025, 1.0]), True),
        (build_pir_affine((0.025, 0.02, 0.02)), False),     # correct PIR
        (np.eye(4), False),                                  # unit voxels (not Brainfast range)
        (np.diag([0.025, -0.02, 0.02, 1.0]), False),        # negative entry → not the RAS bug
        (np.diag([2.5, 2.0, 2.0, 1.0]), False),             # spacing too big (outside Brainfast range)
    ],
)
def test_is_ras_diagonal(affine, expected):
    assert is_ras_diagonal(affine) is expected


def _write_volume(path: Path, affine: np.ndarray, shape: tuple[int, int, int] = (4, 5, 6)) -> None:
    data = np.zeros(shape, dtype=np.uint16)
    img = nib.Nifti1Image(data, affine)
    img.header.set_zooms(tuple(float(np.linalg.norm(affine[:3, i])) for i in range(3)))
    nib.save(img, str(path))


def test_patch_affine_dry_run_doesnt_modify(tmp_path):
    p = tmp_path / "input_volume.nii.gz"
    _write_volume(p, np.diag([0.025, 0.02, 0.02, 1.0]))
    mtime_before = p.stat().st_mtime_ns
    status = patch_affine_inplace(p, apply=False)
    assert status["patched"] is False
    assert status["reason"] == "buggy_ras_diagonal_detected"
    assert p.stat().st_mtime_ns == mtime_before
    # File still has RAS diagonal
    assert nib.aff2axcodes(nib.load(str(p)).affine) == ("R", "A", "S")


def test_patch_affine_apply_rewrites_file(tmp_path):
    p = tmp_path / "input_volume.nii.gz"
    _write_volume(p, np.diag([0.025, 0.02, 0.02, 1.0]))
    status = patch_affine_inplace(p, apply=True)
    assert status["patched"] is True
    assert nib.aff2axcodes(nib.load(str(p)).affine) == ("P", "I", "R")
    # Original backed up with extension-preserving name
    backup = tmp_path / "input_volume.ras_backup.nii.gz"
    assert backup.exists()
    # Backup still has the old affine (nibabel can still load it)
    assert nib.aff2axcodes(nib.load(str(backup)).affine) == ("R", "A", "S")


def test_patch_affine_preserves_voxel_data(tmp_path):
    p = tmp_path / "input_volume.nii.gz"
    data = np.arange(4 * 5 * 6, dtype=np.uint16).reshape((4, 5, 6))
    img = nib.Nifti1Image(data, np.diag([0.025, 0.02, 0.02, 1.0]))
    img.header.set_zooms((0.025, 0.02, 0.02))
    nib.save(img, str(p))
    patch_affine_inplace(p, apply=True)
    new_data = np.asarray(nib.load(str(p)).dataobj)
    np.testing.assert_array_equal(new_data, data)


def test_patch_affine_already_correct_is_noop(tmp_path):
    p = tmp_path / "input_volume.nii.gz"
    _write_volume(p, build_pir_affine((0.025, 0.02, 0.02)))
    status = patch_affine_inplace(p, apply=True)
    assert status["patched"] is False
    assert status["reason"] == "already_correct_or_different_buggy_pattern"


def test_migrate_outputs_root_finds_and_patches_all(tmp_path):
    """Walk a fake outputs dir with 2 stale files + 1 correct file; only stales get patched."""
    run_a = tmp_path / "run_A" / "volume"
    run_b = tmp_path / "run_B" / "volume"
    run_c = tmp_path / "run_C" / "volume"
    for d in (run_a, run_b, run_c):
        d.mkdir(parents=True)
    _write_volume(run_a / "input_volume.nii.gz", np.diag([0.025, 0.02, 0.02, 1.0]))
    _write_volume(run_b / "input_volume.nii.gz", np.diag([0.025, 0.025, 0.025, 1.0]))
    _write_volume(run_c / "input_volume.nii.gz", build_pir_affine((0.025, 0.02, 0.02)))

    # dry run — nothing modified
    results = migrate_outputs_root(tmp_path, apply=False)
    assert len(results) == 3
    detected = [r for r in results if r["reason"] == "buggy_ras_diagonal_detected"]
    assert len(detected) == 2

    # apply — both stales patched
    results = migrate_outputs_root(tmp_path, apply=True)
    patched = [r for r in results if r["patched"]]
    assert len(patched) == 2
    # Correct file untouched
    assert nib.aff2axcodes(nib.load(str(run_c / "input_volume.nii.gz")).affine) == ("P", "I", "R")
    # Stale files now PIR
    assert nib.aff2axcodes(nib.load(str(run_a / "input_volume.nii.gz")).affine) == ("P", "I", "R")
    assert nib.aff2axcodes(nib.load(str(run_b / "input_volume.nii.gz")).affine) == ("P", "I", "R")
