from __future__ import annotations

import numpy as np
import nibabel as nib
from tifffile import imwrite

import project.scripts.truth_export_3d as truth_export_3d


def test_export_registered_truth_slices_writes_per_slice_labels_and_overlays(tmp_path, monkeypatch):
    annotation_volume_path = tmp_path / "annotation_registered.nii.gz"
    out_dir = tmp_path / "registered_slices"

    volume = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        dtype=np.int32,
    )
    nib.save(nib.Nifti1Image(volume, np.eye(4)), str(annotation_volume_path))

    raw0 = tmp_path / "raw_0000.tif"
    raw1 = tmp_path / "raw_0001.tif"
    imwrite(str(raw0), np.array([[10, 20], [30, 40]], dtype=np.uint16))
    imwrite(str(raw1), np.array([[50, 60], [70, 80]], dtype=np.uint16))

    def fake_render_overlay(**kwargs):
        out_png = kwargs["out_png"]
        out_png.write_bytes(b"png")
        return out_png, {"warp": {"method": "3d_truth_export"}}

    monkeypatch.setattr(truth_export_3d, "render_overlay", fake_render_overlay)

    rows = truth_export_3d.export_registered_truth_slices(
        real_slice_paths=[raw0, raw1],
        annotation_volume_path=annotation_volume_path,
        out_dir=out_dir,
        pixel_size_um=25.0,
        slicing_plane="coronal",
    )

    assert len(rows) == 2
    assert (out_dir / "slice_0000_registered_label.tif").exists()
    assert (out_dir / "slice_0001_overlay.png").exists()
    assert rows[0]["registration_method"] == "3d_truth_export"
