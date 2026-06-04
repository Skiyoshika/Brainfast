from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
from tifffile import imread, imwrite

import project.scripts.truth_export_3d as truth_export_3d


@pytest.mark.parametrize(
    ("slicing_plane", "index", "expected"),
    [
        ("coronal", 1, np.array([[5, 6], [7, 8]], dtype=np.int32)),
        ("horizontal", 1, np.array([[3, 4], [7, 8]], dtype=np.int32)),
        ("sagittal", 1, np.array([[2, 4], [6, 8]], dtype=np.int32)),
        ("unexpected", 1, np.array([[5, 6], [7, 8]], dtype=np.int32)),
    ],
)
def test_select_volume_slice_handles_planes_and_defaults_to_coronal(slicing_plane, index, expected):
    volume = np.array(
        [
            [[1, 2], [3, 4]],
            [[5, 6], [7, 8]],
        ],
        dtype=np.int32,
    )

    selected = truth_export_3d._select_volume_slice(volume, index, slicing_plane)

    assert np.array_equal(selected, expected)


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

    recorded: list[dict[str, object]] = []

    def fake_render_overlay(**kwargs):
        recorded.append(kwargs)
        out_png = kwargs["out_png"]
        out_png.write_bytes(b"png")
        return out_png, {"warp": {"method": "3d_truth_export"}}

    monkeypatch.setattr(truth_export_3d, "render_overlay", fake_render_overlay)
    progress_events: list[tuple[int, int]] = []

    rows = truth_export_3d.export_registered_truth_slices(
        real_slice_paths=[raw0, raw1],
        annotation_volume_path=annotation_volume_path,
        out_dir=out_dir,
        pixel_size_um=25.0,
        slicing_plane="coronal",
        progress_cb=lambda done, total: progress_events.append((done, total)),
    )

    assert len(rows) == 2
    label0 = imread(str(out_dir / "slice_0000_registered_label.tif"))
    assert np.array_equal(label0, np.array([[1, 2], [3, 4]], dtype=np.int32))
    assert (out_dir / "slice_0001_overlay.png").exists()
    assert len(recorded) == 2
    assert recorded[0]["prewarped_label"] is True
    assert recorded[0]["warped_label_out"] == out_dir / "slice_0000_registered_label.tif"
    assert recorded[0]["alpha"] == 0.72
    assert recorded[0]["mode"] == "fill"
    assert recorded[0]["major_top_k"] == 28
    assert recorded[0]["fit_mode"] == "cover"
    assert recorded[0]["edge_smooth_iter"] == 0
    assert recorded[0]["warp_params"] == {}
    assert recorded[0]["return_meta"] is True
    assert rows[0]["registration_method"] == "3d_truth_export"
    assert rows[0] == {
        "slice_id": 0,
        "real_slice_path": str(raw0),
        "registered_label_path": str(out_dir / "slice_0000_registered_label.tif"),
        "overlay_path": str(out_dir / "slice_0000_overlay.png"),
        "registration_method": "3d_truth_export",
    }
    assert progress_events[-1] == (2, 2)


def test_export_registered_truth_slices_samples_overlays_but_writes_all_labels(
    tmp_path,
    monkeypatch,
):
    annotation_volume_path = tmp_path / "annotation_registered.nii.gz"
    out_dir = tmp_path / "registered_slices"
    volume = np.arange(12, dtype=np.int32).reshape(3, 2, 2)
    nib.save(nib.Nifti1Image(volume, np.eye(4)), str(annotation_volume_path))

    raw_paths = []
    for idx in range(3):
        raw = tmp_path / f"raw_{idx:04d}.tif"
        imwrite(str(raw), np.full((2, 2), idx, dtype=np.uint16))
        raw_paths.append(raw)

    rendered: list[Path] = []
    label_only: list[Path] = []

    def fake_render_overlay(**kwargs):
        rendered.append(kwargs["out_png"])
        kwargs["out_png"].write_bytes(b"png")
        imwrite(str(kwargs["warped_label_out"]), imread(str(kwargs["label_slice_path"])))
        return kwargs["out_png"], {"warp": {"method": "3d_truth_export"}}

    def fake_write_label_only(**kwargs):
        label_only.append(kwargs["label_path"])
        imwrite(str(kwargs["label_path"]), kwargs["label_slice"])
        return {"warp": {"method": "3d_truth_export_label_only"}}

    monkeypatch.setattr(truth_export_3d, "render_overlay", fake_render_overlay)
    monkeypatch.setattr(truth_export_3d, "_write_prewarped_label_only", fake_write_label_only)

    rows = truth_export_3d.export_registered_truth_slices(
        real_slice_paths=raw_paths,
        annotation_volume_path=annotation_volume_path,
        out_dir=out_dir,
        pixel_size_um=25.0,
        slicing_plane="coronal",
        overlay_stride=2,
    )

    assert rendered == [
        out_dir / "slice_0000_overlay.png",
        out_dir / "slice_0002_overlay.png",
    ]
    assert label_only == [out_dir / "slice_0001_registered_label.tif"]
    assert rows[0]["overlay_path"].endswith("slice_0000_overlay.png")
    assert rows[1]["overlay_path"] == ""
    assert rows[2]["overlay_path"].endswith("slice_0002_overlay.png")
    assert all(Path(row["registered_label_path"]).exists() for row in rows)


def test_export_registered_truth_slices_rejects_mismatched_slice_count(tmp_path):
    annotation_volume_path = tmp_path / "annotation_registered.nii.gz"
    volume = np.arange(8, dtype=np.int32).reshape(2, 2, 2)
    nib.save(nib.Nifti1Image(volume, np.eye(4)), str(annotation_volume_path))

    raw0 = tmp_path / "raw_0000.tif"
    imwrite(str(raw0), np.array([[10, 20], [30, 40]], dtype=np.uint16))

    with pytest.raises(ValueError, match="slice count mismatch"):
        truth_export_3d.export_registered_truth_slices(
            real_slice_paths=[raw0],
            annotation_volume_path=annotation_volume_path,
            out_dir=tmp_path / "registered_slices",
            pixel_size_um=25.0,
            slicing_plane="coronal",
        )
