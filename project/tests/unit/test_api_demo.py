from __future__ import annotations

import csv
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd
from tifffile import imwrite

import project.frontend.server_context as ctx
from project.frontend.server import app


def _write_metrics_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["NCC", "0.7123"])
        writer.writerow(["SSIM", "0.3842"])
        writer.writerow(["Dice", "0.9188"])
        writer.writerow(["MSE", "0.0410"])
        writer.writerow(["PSNR", "13.9021"])


def _make_nifti(path: Path, data: np.ndarray) -> None:
    img = nib.Nifti1Image(data, np.eye(4, dtype=np.float32))
    nib.save(img, str(path))


def _make_registration_run(root: Path, name: str) -> Path:
    run_dir = root / name
    run_dir.mkdir()

    brain = np.zeros((16, 32, 24), dtype=np.float32)
    annotation = np.zeros((16, 32, 24), dtype=np.int16)
    for z in range(3, 13):
        brain[z, 6:28, 4:20] = np.linspace(0.2, 1.0, 22 * 16, dtype=np.float32).reshape(22, 16)
        annotation[z, 6:28, 4:20] = 1
        annotation[z, 11:23, 9:18] = 2

    brain_path = run_dir / "registered_brain.nii.gz"
    annotation_path = run_dir / "annotation_fixed_half.nii.gz"
    _make_nifti(brain_path, brain)
    _make_nifti(annotation_path, annotation)
    _write_metrics_csv(run_dir / "registration_metrics.csv")
    (run_dir / "report.html").write_text("<html>demo</html>", encoding="utf-8")

    metadata = {
        "input_source": str(root / f"{name}.tif"),
        "registered_brain": str(brain_path.resolve()),
        "annotation_fixed_half": str(annotation_path.resolve()),
        "backend": "ants",
        "laplacian_enabled": True,
        "hemisphere": "left",
        "target_um": 25.0,
        "staining_stats": {
            "atlas_coverage": 0.9821,
            "staining_rate": 0.2142,
        },
    }
    (run_dir / "registration_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    return run_dir


def test_demo_panel_prefers_latest_registration_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ctx, "OUTPUT_DIR", tmp_path)
    run_dir = _make_registration_run(tmp_path, "demo_run")

    with app.test_client() as client:
        resp = client.get("/api/outputs/demo-panel")
        assert resp.status_code == 200
        assert (run_dir / "qc_panel.jpg").exists()

        best = client.get("/api/outputs/demo-best-slice")
        assert best.status_code == 200
        assert (run_dir / "qc_best_slice.jpg").exists()

        annotated = client.get("/api/outputs/demo-annotated-slice")
        assert annotated.status_code == 200
        assert (run_dir / "qc_annotated_slice.jpg").exists()


def test_reg_stats_returns_registration_run_metrics(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ctx, "OUTPUT_DIR", tmp_path)
    _make_registration_run(tmp_path, "demo_run")

    with app.test_client() as client:
        resp = client.get("/api/outputs/reg-stats")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["mode"] == "registration_run"
        assert data["pipeline"] == "ANTS + Laplacian"
        assert data["ncc"] == 0.7123
        assert data["staining_rate"] == 0.2142


def test_detection_samples_returns_rendered_marked_slices(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ctx, "OUTPUT_DIR", tmp_path)

    source_paths: list[Path] = []
    rows = []
    for sid in range(3):
        raw = np.zeros((96, 120), dtype=np.uint16)
        raw[18:78, 22:102] = 400 + sid * 100
        raw[30 + sid : 34 + sid, 40:44] = 2000
        source_path = tmp_path / f"slice_{sid:04d}.tif"
        imwrite(str(source_path), raw)
        source_paths.append(source_path)
        rows.append(
            {
                "cell_id": sid + 1,
                "slice_id": sid,
                "x": 40 + sid * 8,
                "y": 34 + sid * 6,
                "score": 42.0 + sid,
                "detector": "cellpose_cyto2",
                "area_px": 18.0,
                "source_slice_path": str(source_path),
                "count_sampling_mode": "single",
                "count_slice_interval_n": 1,
            }
        )
    pd.DataFrame(rows).to_csv(tmp_path / "cells_dedup.csv", index=False)

    with app.test_client() as client:
        resp = client.get("/api/outputs/detection-samples")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["count"] == 3
        assert data["samples"][0]["detectors"] == "cellpose_cyto2"

        sample_url = data["samples"][0]["url"]
        image_resp = client.get(sample_url)
        assert image_resp.status_code == 200
        assert (tmp_path / "detection_samples" / "cellcount_sample_01.png").exists()


def test_cell_chart_uses_outputs_tables(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ctx, "OUTPUT_DIR", tmp_path)

    pd.DataFrame(
        [
            {"region_id": 997, "depth": 0, "count": 4, "region_name": "root", "acronym": "root"},
            {"region_id": 315, "depth": 5, "count": 2, "region_name": "Isocortex", "acronym": "Isocortex"},
            {"region_id": 477, "depth": 4, "count": 2, "region_name": "Striatum", "acronym": "STR"},
        ]
    ).to_csv(tmp_path / "cell_counts_hierarchy.csv", index=False)
    pd.DataFrame(
        [
            {
                "mapping_status": "ok",
                "region_id": 672,
                "region_name": "Caudoputamen",
                "parent_name": "Striatum dorsal region",
                "structure_id_path": "/997/8/567/623/477/485/672/",
                "source_slice_path": str(tmp_path / "z0100.tif"),
            },
            {
                "mapping_status": "ok",
                "region_id": 322,
                "region_name": "Primary somatosensory area",
                "parent_name": "Primary somatosensory area",
                "structure_id_path": "/997/8/567/688/695/315/453/322/",
                "source_slice_path": str(tmp_path / "z0105.tif"),
            },
            {
                "mapping_status": "outside_registered_slice",
                "region_id": 0,
                "region_name": "OUTSIDE_ATLAS",
                "parent_name": "",
                "structure_id_path": "",
                "source_slice_path": str(tmp_path / "z0110.tif"),
            },
        ]
    ).to_csv(tmp_path / "cells_mapped.csv", index=False)

    with app.test_client() as client:
        resp = client.get("/api/outputs/cell-chart")
        assert resp.status_code == 200
        assert (tmp_path / "cell_count_chart.png").exists()


def test_cell_summary_returns_product_summary_json(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ctx, "OUTPUT_DIR", tmp_path)

    pd.DataFrame(
        [
            {"region_id": 997, "depth": 0, "count": 4, "region_name": "root", "acronym": "root"},
            {"region_id": 315, "depth": 5, "count": 2, "region_name": "Isocortex", "acronym": "Isocortex"},
            {"region_id": 477, "depth": 4, "count": 2, "region_name": "Striatum", "acronym": "STR"},
        ]
    ).to_csv(tmp_path / "cell_counts_hierarchy.csv", index=False)
    pd.DataFrame(
        [
            {
                "mapping_status": "ok",
                "region_id": 672,
                "region_name": "Caudoputamen",
                "parent_name": "Striatum dorsal region",
                "structure_id_path": "/997/8/567/623/477/485/672/",
                "source_slice_path": str(tmp_path / "tmp_channel" / "ch_0_0000.tif"),
                "count_sampling_mode": "single",
            },
            {
                "mapping_status": "outside_registered_slice",
                "region_id": 0,
                "region_name": "OUTSIDE_ATLAS",
                "parent_name": "",
                "structure_id_path": "",
                "source_slice_path": str(tmp_path / "tmp_channel" / "ch_0_0001.tif"),
                "count_sampling_mode": "single",
            },
        ]
    ).to_csv(tmp_path / "cells_mapped.csv", index=False)
    (tmp_path / "detection_summary.json").write_text(
        '{"sampling_mode":"single","dedup_detector_counts":{"cellpose_cyto2":2}}',
        encoding="utf-8",
    )

    with app.test_client() as client:
        resp = client.get("/api/outputs/cell-summary")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["summary"]["scope_kind"] == "working_set"
        assert data["summary"]["counting_mode"] == "single"
        assert data["summary"]["detectors"] == "cellpose_cyto2"
        assert data["summary"]["outside_count"] == 1
