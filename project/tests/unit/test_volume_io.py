from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from tifffile import imwrite


def test_inspect_volume_source_on_stack_tiff(tmp_path: Path) -> None:
    src = tmp_path / "stack.tif"
    arr = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
    imwrite(str(src), arr)

    from scripts.volume_io import inspect_volume_source

    info = inspect_volume_source(src)
    assert info.source_type == "stack_tiff"
    assert info.shape == (3, 4, 5)
    assert info.num_slices == 3


def test_volume_source_to_nifti_preserves_stack_with_padding(tmp_path: Path) -> None:
    src = tmp_path / "stack.tif"
    arr = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5)
    imwrite(str(src), arr)
    out = tmp_path / "brain.nii.gz"

    from scripts.volume_io import volume_source_to_nifti

    _, shape = volume_source_to_nifti(
        src,
        out,
        pixel_um_xy=5.0,
        z_spacing_um=25.0,
        target_um=None,
        pad_z=1,
        pad_y=2,
        pad_x=3,
        normalize=False,
    )

    img = nib.load(str(out))
    data = np.asarray(img.dataobj)
    assert shape == (5, 8, 11)
    assert data.shape == (5, 8, 11)
    np.testing.assert_array_equal(data[1:4, 2:6, 3:8], arr.astype(np.float32))
    np.testing.assert_allclose(img.header.get_zooms()[:3], (0.025, 0.005, 0.005))
    assert nib.aff2axcodes(img.affine) == ("P", "I", "R")
    np.testing.assert_allclose(
        img.affine,
        np.array(
            [
                [0.0, 0.0, 0.005, 0.0],
                [-0.025, 0.0, 0.0, 0.0],
                [0.0, -0.005, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        ),
    )


def test_volume_source_to_nifti_accepts_slice_dir(tmp_path: Path) -> None:
    src_dir = tmp_path / "slices"
    src_dir.mkdir()
    for z in range(3):
        imwrite(str(src_dir / f"z{z:04d}.tif"), np.full((6, 8), z + 1, dtype=np.uint16))
    out = tmp_path / "brain_dir.nii.gz"

    from scripts.volume_io import volume_source_to_nifti

    _, shape = volume_source_to_nifti(
        src_dir,
        out,
        pixel_um_xy=5.0,
        z_spacing_um=25.0,
        target_um=10.0,
        normalize=False,
    )

    img = nib.load(str(out))
    data = np.asarray(img.dataobj)
    assert shape == (3, 3, 4)
    assert data.shape == (3, 3, 4)
    assert float(data[0, 0, 0]) == 1.0
    assert float(data[2, 0, 0]) == 3.0
    np.testing.assert_allclose(img.header.get_zooms()[:3], (0.025, 0.01, 0.01))
    assert nib.aff2axcodes(img.affine) == ("P", "I", "R")
