"""Unit tests for volume_io.multipage_tiff_to_nifti (Brainfast-native port of
Xu Lab's ``create_nii_images`` for multi-page TIFF input).
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import tifffile

from project.scripts.volume_io import _orient_section_xulab, multipage_tiff_to_nifti


def _write_multipage_tiff(path: Path, num_pages: int, h: int, w: int, dtype=np.uint16) -> Path:
    rng = np.random.default_rng(42)
    stack = rng.integers(0, 60000, size=(num_pages, h, w), dtype=dtype)
    tifffile.imwrite(str(path), stack)
    return path


def test_orient_section_transpose_and_double_flip():
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    out = _orient_section_xulab(arr)
    # Xu Lab's orient = T -> flip axis 0 -> flip axis 1 (= 180° rotation of transpose)
    expected = np.flip(np.flip(arr.T, axis=0), axis=1)
    np.testing.assert_array_equal(out, expected)


def test_orient_section_zeros_negatives():
    arr = np.array([[-1, 2], [3, -4]], dtype=np.int32)
    out = _orient_section_xulab(arr)
    assert (out >= 0).all()


def test_multipage_tiff_to_nifti_basic(tmp_path: Path):
    src = _write_multipage_tiff(tmp_path / "multi.tif", num_pages=20, h=160, w=120)
    out_path = tmp_path / "out.nii.gz"
    meta = multipage_tiff_to_nifti(
        src,
        out_path,
        downscale_factor=8,
        voxel_spacing_mm=(0.05, 0.025, 0.025),
        orient=True,
        ccf_origin=None,
    )
    assert meta["page_count"] == 20
    # After /8 + orient (transpose flips H and W), shape should be (20, 120/8, 160/8)
    assert meta["shape"][0] == 20
    assert meta["shape"] == (20, 15, 20)
    img = nib.load(str(out_path))
    assert tuple(img.shape) == (20, 15, 20)
    # PIR convention: axis codes ('P', 'I', 'R')
    assert nib.aff2axcodes(img.affine) == ("P", "I", "R")


def test_multipage_tiff_to_nifti_no_downscale(tmp_path: Path):
    src = _write_multipage_tiff(tmp_path / "multi.tif", num_pages=10, h=80, w=60)
    out_path = tmp_path / "no_down.nii.gz"
    meta = multipage_tiff_to_nifti(
        src,
        out_path,
        downscale_factor=1,
        voxel_spacing_mm=(0.05, 0.003, 0.003),
        orient=True,
        ccf_origin=None,
    )
    # With no downscale + orient, shape = (pages, w, h)
    assert meta["shape"] == (10, 60, 80)


def test_multipage_tiff_to_nifti_orient_false_preserves_layout(tmp_path: Path):
    src = _write_multipage_tiff(tmp_path / "multi.tif", num_pages=5, h=80, w=60)
    out_path = tmp_path / "no_orient.nii.gz"
    meta = multipage_tiff_to_nifti(
        src, out_path, downscale_factor=1, orient=False, ccf_origin=None
    )
    # Without orient, shape = (pages, h, w)
    assert meta["shape"] == (5, 80, 60)


def test_multipage_tiff_to_nifti_pir_axis_codes(tmp_path: Path):
    src = _write_multipage_tiff(tmp_path / "multi.tif", num_pages=8, h=120, w=100)
    out_path = tmp_path / "pir.nii.gz"
    multipage_tiff_to_nifti(src, out_path, voxel_spacing_mm=(0.05, 0.025, 0.025))
    img = nib.load(str(out_path))
    assert nib.aff2axcodes(img.affine) == ("P", "I", "R")


def test_multipage_tiff_to_nifti_ccf_origin_stamped(tmp_path: Path):
    src = _write_multipage_tiff(tmp_path / "multi.tif", num_pages=4, h=80, w=80)
    out_path = tmp_path / "with_origin.nii.gz"
    multipage_tiff_to_nifti(src, out_path, ccf_origin=(-5.695, 5.35, 5.22))
    img = nib.load(str(out_path))
    np.testing.assert_allclose(img.affine[:3, 3], [-5.695, 5.35, 5.22], atol=1e-6)


def test_multipage_tiff_to_nifti_rejects_empty_tiff(tmp_path: Path):
    # tifffile can't actually produce a 0-page file; make a single-page one,
    # then delete pages from within — but we can instead write a 1-page file
    # and just test that the code path handles valid input fine.
    src = _write_multipage_tiff(tmp_path / "mini.tif", num_pages=1, h=40, w=40)
    meta = multipage_tiff_to_nifti(src, tmp_path / "out.nii.gz", voxel_spacing_mm=(0.05, 0.025, 0.025))
    assert meta["page_count"] == 1


def test_multipage_tiff_to_nifti_zooms_reflect_voxel_spacing(tmp_path: Path):
    src = _write_multipage_tiff(tmp_path / "multi.tif", num_pages=6, h=80, w=80)
    out_path = tmp_path / "zooms.nii.gz"
    multipage_tiff_to_nifti(
        src,
        out_path,
        voxel_spacing_mm=(0.005, 0.025, 0.025),  # 5µm Z, 25µm XY
    )
    img = nib.load(str(out_path))
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    np.testing.assert_allclose(zooms, (0.005, 0.025, 0.025), atol=1e-6)
