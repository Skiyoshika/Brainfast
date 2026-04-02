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
    assert img.get_data_dtype() == np.uint16
    assert meta["shape"] == [2, 2, 2]
    assert meta["downsample_factor"] == 2
    assert meta["slice_count"] == 2
    assert meta["volume_path"] == output_path


def test_build_volume_from_tiffs_raises_for_empty_glob(tmp_path):
    with pytest.raises(FileNotFoundError):
        build_volume_from_tiffs(
            tmp_path / "slices",
            tmp_path / "brain_25um.nii.gz",
            pixel_um_xy=12.5,
            z_spacing_um=25.0,
        )


def test_build_volume_from_tiffs_collapses_3d_tiffs_to_first_plane(tmp_path):
    slice_dir = tmp_path / "slices"
    slice_dir.mkdir()

    stacked = np.stack(
        [
            np.full((2, 2), 5, dtype=np.uint8),
            np.full((2, 2), 250, dtype=np.uint8),
        ],
        axis=0,
    )
    imwrite(slice_dir / "z001.tif", stacked)

    output_path = tmp_path / "brain_25um.nii.gz"
    build_volume_from_tiffs(
        slice_dir,
        output_path,
        pixel_um_xy=12.5,
        z_spacing_um=25.0,
        target_um=25.0,
    )

    data = np.asarray(nib.load(str(output_path)).dataobj)
    assert data.dtype == np.uint16
    assert np.all(data == 0)


def test_build_volume_from_tiffs_normalizes_and_clips_to_uint16(tmp_path):
    slice_dir = tmp_path / "slices"
    slice_dir.mkdir()

    imwrite(slice_dir / "z001.tif", np.zeros((2, 2), dtype=np.uint8))
    imwrite(slice_dir / "z002.tif", np.full((2, 2), 255, dtype=np.uint8))

    output_path = tmp_path / "brain_25um.nii.gz"
    build_volume_from_tiffs(
        slice_dir,
        output_path,
        pixel_um_xy=12.5,
        z_spacing_um=25.0,
        target_um=25.0,
    )

    data = np.asarray(nib.load(str(output_path)).dataobj)
    assert data.dtype == np.uint16
    assert data.min() == 0
    assert data.max() == 65535


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
    assert tmpl_half.get_data_dtype() == np.float32
    assert ann_half.get_data_dtype() == np.int32
    assert meta["hemisphere"] == "left"
    assert meta["ap_range"] == [1, 5]
    assert meta["shape"] == [4, 6, 3]


@pytest.mark.parametrize(
    ("hemisphere", "expected_first", "expected_last"),
    [
        ("right", [1.0, 0.0, 3.0], [1.0, 0.0, 5.0]),
        ("right_flipped", [1.0, 0.0, 5.0], [1.0, 0.0, 3.0]),
    ],
)
def test_prepare_half_template_inputs_updates_affine_for_right_hemispheres(
    tmp_path, hemisphere, expected_first, expected_last
):
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
        hemisphere=hemisphere,
        ap_start=1,
        ap_end=5,
        out_dir=out_dir,
    )

    tmpl_half = nib.load(str(out_dir / "template_half.nii.gz"))
    ann_half = nib.load(str(out_dir / "annotation_half.nii.gz"))

    assert tmpl_half.shape == (4, 6, 3)
    assert ann_half.shape == (4, 6, 3)
    assert nib.affines.apply_affine(tmpl_half.affine, [0, 0, 0]).tolist() == expected_first
    assert nib.affines.apply_affine(tmpl_half.affine, [0, 0, 2]).tolist() == expected_last
    assert nib.affines.apply_affine(ann_half.affine, [0, 0, 0]).tolist() == expected_first
    assert nib.affines.apply_affine(ann_half.affine, [0, 0, 2]).tolist() == expected_last
    assert meta["hemisphere"] == hemisphere
    assert meta["ap_range"] == [1, 5]
    assert meta["shape"] == [4, 6, 3]
