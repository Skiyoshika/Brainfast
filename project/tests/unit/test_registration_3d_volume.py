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


def test_build_volume_from_tiffs_uncapped_allows_downsample_above_four(tmp_path):
    """Miki-level registration needs native xy resolution. When raw_factor > 4
    (e.g. 5 µm pixel → 25 µm target = factor 5), the old hardcoded
    min(4, ...) cap forced over-downsampling and destroyed the moving volume's
    intensity detail, blocking ANTs MI/CC convergence (NCC -0.16 vs Miki 0.66).

    With xy_downsample_cap=None (new default), downsample_factor should match
    round(raw_factor) — no artificial ceiling."""
    slice_dir = tmp_path / "slices"
    slice_dir.mkdir()

    # 10×10 slices → after downsample=5 → 2×2 per slice
    imwrite(slice_dir / "z001.tif", np.full((10, 10), 100, dtype=np.uint16))
    imwrite(slice_dir / "z002.tif", np.full((10, 10), 200, dtype=np.uint16))

    meta = build_volume_from_tiffs(
        slice_dir,
        tmp_path / "vol_uncapped.nii.gz",
        pixel_um_xy=5.0,
        z_spacing_um=25.0,
        target_um=25.0,
        glob_pattern="z*.tif",
    )

    # raw_factor = 25/5 = 5. Old capped default would return 4. New default
    # (cap=None) should return 5 to match Miki-level voxel density.
    assert meta["downsample_factor"] == 5, (
        f"uncapped default regressed: got {meta['downsample_factor']}, expected 5"
    )


def test_build_volume_from_tiffs_respects_explicit_cap(tmp_path):
    """Operators on memory-constrained machines can still ask for a cap."""
    slice_dir = tmp_path / "slices"
    slice_dir.mkdir()

    imwrite(slice_dir / "z001.tif", np.full((10, 10), 100, dtype=np.uint16))

    meta = build_volume_from_tiffs(
        slice_dir,
        tmp_path / "vol_capped.nii.gz",
        pixel_um_xy=5.0,
        z_spacing_um=25.0,
        target_um=25.0,
        xy_downsample_cap=2,
    )

    # raw_factor=5, cap=2 → downsample=2
    assert meta["downsample_factor"] == 2


def test_build_volume_from_tiffs_emits_allen_compatible_affine(tmp_path):
    """The emitted NIfTI must carry an Allen-CCF-compatible off-diagonal affine
    so ANTs sees the moving volume in the same physical coordinate system as
    the template, rather than a raw-diagonal affine that forces SyN to absorb
    a global rigid mismatch.

    Reference structure (from RegTools `create_nifti_image` and Allen
    average_template_25.nii.gz):
        row 0: (0, 0, +dz, *)         — source axis-2 → physical axis-0
        row 1: (-dx, 0, 0, *)         — source axis-0 → physical axis-1 (negated)
        row 2: (0, -dy, 0, *)         — source axis-1 → physical axis-2 (negated)
        row 3: (0, 0,  0,  1)
    where (dx, dy, dz) = (z_spacing_mm, xy_spacing_mm, xy_spacing_mm).
    The previous diag-only affine (Bug: NCC -0.24 vs Miki 0.66) is the
    regression we're guarding against.
    """
    slice_dir = tmp_path / "slices"
    slice_dir.mkdir()

    imwrite(slice_dir / "z001.tif", np.full((10, 10), 100, dtype=np.uint16))
    imwrite(slice_dir / "z002.tif", np.full((10, 10), 200, dtype=np.uint16))

    out = tmp_path / "vol.nii.gz"
    build_volume_from_tiffs(
        slice_dir,
        out,
        pixel_um_xy=5.0,
        z_spacing_um=24.765,
        target_um=25.0,  # raw_factor = 5, no cap → downsample=5 → xy spacing = 25 µm
    )

    img = nib.load(str(out))
    A = img.affine
    dz_mm = 0.024765
    dy_mm = 0.025
    dx_mm = 0.025

    # Off-diagonal placements must be non-zero, with the expected signs.
    assert A[0, 2] == pytest.approx(dx_mm, rel=1e-3), (
        f"affine[0,2] should be +xy_spacing; got {A[0, 2]}"
    )
    assert A[1, 0] == pytest.approx(-dz_mm, rel=1e-3), (
        f"affine[1,0] should be -z_spacing; got {A[1, 0]}"
    )
    assert A[2, 1] == pytest.approx(-dy_mm, rel=1e-3), (
        f"affine[2,1] should be -xy_spacing; got {A[2, 1]}"
    )
    # Diagonal (except last row) must be zero — i.e. NOT a plain diag affine.
    assert A[0, 0] == 0.0
    assert A[1, 1] == 0.0
    assert A[2, 2] == 0.0
    # Homogeneous row.
    assert (A[3] == np.array([0, 0, 0, 1])).all()


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


def test_prepare_half_template_inputs_left_flipped_mirrors_left_half(tmp_path):
    """Symmetric to right_flipped: takes CCF left half then LR-mirrors so the
    lateral edge lands on the user's image-right side (standard mount for
    left-hemisphere coronal sections)."""
    template_path = tmp_path / "template.nii.gz"
    annotation_path = tmp_path / "annotation.nii.gz"

    template_data = np.arange(4 * 4 * 6, dtype=np.float32).reshape(4, 4, 6)
    annotation_data = np.arange(4 * 4 * 6, dtype=np.int32).reshape(4, 4, 6)
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(template_data, affine), str(template_path))
    nib.save(nib.Nifti1Image(annotation_data, affine), str(annotation_path))

    out_dir = tmp_path / "prepared_lf"
    meta = prepare_half_template_inputs(
        template_path,
        annotation_path,
        hemisphere="left_flipped",
        ap_start=0,
        ap_end=4,
        out_dir=out_dir,
    )
    tmpl_half = nib.load(str(out_dir / "template_half.nii.gz"))
    arr = np.asarray(tmpl_half.dataobj)

    # Shape: ap_range × full y × half x
    assert tmpl_half.shape == (4, 4, 3)
    assert meta["hemisphere"] == "left_flipped"

    # Verify the LR mirror actually happened: arr should equal
    # template_data[:, :, :3][:, :, ::-1]
    expected = template_data[:, :, :3][:, :, ::-1]
    np.testing.assert_array_equal(arr, expected)


def test_prepare_half_template_inputs_whole_returns_full_template(tmp_path):
    """Whole brain mode skips both the LR crop and the LR flip."""
    template_path = tmp_path / "template.nii.gz"
    annotation_path = tmp_path / "annotation.nii.gz"

    template_data = np.arange(4 * 4 * 6, dtype=np.float32).reshape(4, 4, 6)
    annotation_data = np.arange(4 * 4 * 6, dtype=np.int32).reshape(4, 4, 6)
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(template_data, affine), str(template_path))
    nib.save(nib.Nifti1Image(annotation_data, affine), str(annotation_path))

    out_dir = tmp_path / "prepared_whole"
    meta = prepare_half_template_inputs(
        template_path,
        annotation_path,
        hemisphere="whole",
        ap_start=0,
        ap_end=4,
        out_dir=out_dir,
    )
    tmpl_half = nib.load(str(out_dir / "template_half.nii.gz"))

    # Width is preserved (no half-crop)
    assert tmpl_half.shape == (4, 4, 6)
    expected = template_data
    np.testing.assert_array_equal(np.asarray(tmpl_half.dataobj), expected)
    assert meta["hemisphere"] == "whole"


def test_prepare_half_template_inputs_both_alias_matches_whole(tmp_path):
    """`both` is an alias of `whole` for back-compat with older config files."""
    template_path = tmp_path / "template.nii.gz"
    annotation_path = tmp_path / "annotation.nii.gz"

    template_data = np.arange(4 * 4 * 6, dtype=np.float32).reshape(4, 4, 6)
    annotation_data = np.arange(4 * 4 * 6, dtype=np.int32).reshape(4, 4, 6)
    affine = np.eye(4)
    nib.save(nib.Nifti1Image(template_data, affine), str(template_path))
    nib.save(nib.Nifti1Image(annotation_data, affine), str(annotation_path))

    out_dir = tmp_path / "prepared_both"
    meta = prepare_half_template_inputs(
        template_path,
        annotation_path,
        hemisphere="both",
        ap_start=0,
        ap_end=4,
        out_dir=out_dir,
    )
    tmpl_half = nib.load(str(out_dir / "template_half.nii.gz"))
    assert tmpl_half.shape == (4, 4, 6)
    assert meta["hemisphere"] == "both"


@pytest.mark.parametrize("hemisphere", ["right", "right_flipped"])
def test_prepare_half_template_inputs_uses_zero_origin_affine_for_ants(tmp_path, hemisphere):
    """Template and annotation use zero-origin diagonal affines so ANTs sees
    spatial overlap with the input volume (also at origin)."""
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
    # Zero-origin: voxel [0,0,0] maps to physical (0,0,0)
    assert nib.affines.apply_affine(tmpl_half.affine, [0, 0, 0]).tolist() == [0.0, 0.0, 0.0]
    # Template and annotation share the same affine
    np.testing.assert_array_equal(tmpl_half.affine, ann_half.affine)
    assert meta["hemisphere"] == hemisphere
    assert meta["ap_range"] == [1, 5]
    assert meta["shape"] == [4, 6, 3]
