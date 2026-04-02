import nibabel as nib
import numpy as np
import pytest
from tifffile import imwrite

from project.scripts.registration_3d_volume import (
    build_volume_from_tiffs,
    prepare_half_template_inputs,
)


def test_build_volume_from_tiffs_writes_target_resolution_volume(tmp_path):
    slice_dir = tmp_path / "slices"
    slice_dir.mkdir()

    imwrite(slice_dir / "z001.tif", np.arange(16, dtype=np.uint8).reshape(4, 4))
    imwrite(slice_dir / "z002.tif", np.arange(16, 32, dtype=np.uint8).reshape(4, 4))

    output_path = tmp_path / "brain_25um.nii.gz"
    meta = build_volume_from_tiffs(
        slice_dir,
        output_path,
        pixel_um_xy=12.5,
        z_spacing_um=25.0,
        target_um=25.0,
        glob_pattern="z*.tif",
    )

    assert output_path.exists()

    img = nib.load(str(output_path))
    assert img.shape == (2, 2, 2)
    assert img.header.get_zooms()[:3] == pytest.approx((0.025, 0.025, 0.025))
    assert meta["shape"] == [2, 2, 2]
    assert meta["downsample_factor"] == 2


def test_prepare_half_template_inputs_crops_ap_range_and_left_half(tmp_path):
    template_path = tmp_path / "template.nii.gz"
    annotation_path = tmp_path / "annotation.nii.gz"

    template_data = np.arange(6 * 6 * 6, dtype=np.float32).reshape(6, 6, 6)
    annotation_data = np.arange(6 * 6 * 6, dtype=np.int32).reshape(6, 6, 6)
    affine = np.eye(4)

    nib.save(nib.Nifti1Image(template_data, affine), str(template_path))
    nib.save(nib.Nifti1Image(annotation_data, affine), str(annotation_path))

    out_dir = tmp_path / "prepared"
    meta = prepare_half_template_inputs(
        template_path,
        annotation_path,
        hemisphere="left",
        ap_start=1,
        ap_end=5,
        out_dir=out_dir,
    )

    tmpl_half = nib.load(str(out_dir / "template_half.nii.gz"))
    ann_half = nib.load(str(out_dir / "annotation_half.nii.gz"))

    assert tmpl_half.shape == (4, 6, 3)
    assert ann_half.shape == (4, 6, 3)
    assert meta["hemisphere"] == "left"
    assert meta["ap_range"] == [1, 5]
