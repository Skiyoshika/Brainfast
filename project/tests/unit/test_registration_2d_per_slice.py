"""Unit tests for registration_2d_per_slice.

Covers the pure-python helpers that don't need antspy (CCF slice extraction,
moving-slice orient+load). The full ANTs registration in
``register_slice_to_ccf`` is exercised via integration tests on real data
(not here — needs antspy runtime + TIFF samples).
"""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import tifffile

from project.scripts.registration_2d_per_slice import _extract_ccf_slice, _read_moving_slice


def _write_ccf_mock(path: Path, shape=(20, 30, 40)) -> Path:
    """Save a 3D NIfTI with CCF-like PIR affine (non-trivial direction cosines)."""
    data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
    affine = np.array(
        [
            [0.0, 0.0, 0.025, -5.695],
            [-0.025, 0.0, 0.0, 5.35],
            [0.0, -0.025, 0.0, 5.22],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(path))
    return path


def test_extract_ccf_slice_returns_correct_shape(tmp_path: Path):
    ccf = _write_ccf_mock(tmp_path / "ccf.nii.gz", shape=(20, 30, 40))
    out = _extract_ccf_slice(ccf, 10, tmp_path / "slice10.nii.gz")
    img = nib.load(str(out))
    # 3D NIfTI-with-2D-data: shape has a trailing 1-axis when extracted
    assert img.shape[0] == 30
    assert img.shape[1] == 40


def test_extract_ccf_slice_preserves_pixel_values(tmp_path: Path):
    ccf = _write_ccf_mock(tmp_path / "ccf.nii.gz", shape=(5, 4, 3))
    full = nib.load(str(ccf)).get_fdata()
    out = _extract_ccf_slice(ccf, 2, tmp_path / "slice2.nii.gz")
    slice_read = nib.load(str(out)).get_fdata()
    np.testing.assert_array_equal(slice_read, full[2])


def test_extract_ccf_slice_out_of_range_raises(tmp_path: Path):
    ccf = _write_ccf_mock(tmp_path / "ccf.nii.gz", shape=(5, 4, 3))
    import pytest

    with pytest.raises(IndexError, match="out of range"):
        _extract_ccf_slice(ccf, 99, tmp_path / "bad.nii.gz")


def test_extract_ccf_slice_writes_2d_affine(tmp_path: Path):
    """Extracted slice affine should have the in-plane columns of the 3D affine
    + origin shifted by the out-of-plane step × slice index.
    """
    ccf = _write_ccf_mock(tmp_path / "ccf.nii.gz", shape=(10, 20, 30))
    full_img = nib.load(str(ccf))
    out = _extract_ccf_slice(ccf, 3, tmp_path / "slice3.nii.gz")
    sliced_affine = nib.load(str(out)).affine
    # The extracted slice's columns 0 and 1 come from the full volume's
    # columns 1 and 2
    np.testing.assert_allclose(sliced_affine[:3, 0], full_img.affine[:3, 1])
    np.testing.assert_allclose(sliced_affine[:3, 1], full_img.affine[:3, 2])


def test_read_moving_slice_tif_no_orient(tmp_path: Path):
    arr = np.arange(6, dtype=np.uint16).reshape(2, 3)
    p = tmp_path / "slice.tif"
    tifffile.imwrite(str(p), arr)
    loaded = _read_moving_slice(p, orient=False)
    np.testing.assert_array_equal(loaded, arr.astype(np.float32))


def test_read_moving_slice_tif_with_orient_applies_xulab_transforms(tmp_path: Path):
    arr = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.uint16)
    p = tmp_path / "slice.tif"
    tifffile.imwrite(str(p), arr)
    loaded = _read_moving_slice(p, orient=True)
    # Xu Lab orient = transpose + flip both axes
    expected = np.flip(np.flip(arr.astype(np.float32).T, axis=0), axis=1)
    np.testing.assert_array_equal(loaded, expected)


def test_read_moving_slice_nifti(tmp_path: Path):
    arr = np.arange(12, dtype=np.uint16).reshape(3, 4)
    p = tmp_path / "slice.nii.gz"
    img = nib.Nifti1Image(arr, np.eye(4))
    nib.save(img, str(p))
    loaded = _read_moving_slice(p, orient=False)
    np.testing.assert_array_equal(loaded, arr.astype(np.float32))


def test_read_moving_slice_rejects_unknown_extension(tmp_path: Path):
    p = tmp_path / "unknown.xyz"
    p.write_bytes(b"stub")
    import pytest

    with pytest.raises(ValueError, match="unsupported"):
        _read_moving_slice(p)
