from __future__ import annotations

import csv
import json
from pathlib import Path

from scripts.build_registration_report import build_outputs_index, build_run_report


def test_build_run_report_and_index(tmp_path: Path) -> None:
    outputs_root = tmp_path / "outputs"
    run_dir = outputs_root / "demo_run"
    run_dir.mkdir(parents=True)

    for name in [
        "overview.png",
        "registered_brain.nii.gz",
        "registration_metadata.json",
        "registration_metrics.csv",
    ]:
        (run_dir / name).write_bytes(b"placeholder")

    metadata = {
        "input_source": str(tmp_path / "Sample" / "demo_stack.tif"),
        "registered_brain": str((run_dir / "registered_brain.nii.gz").resolve()),
        "annotation_fixed_half": "",
        "registered_brain_pre_laplacian": "",
        "backend": "ants",
        "laplacian_enabled": True,
        "hemisphere": "left",
        "target_um": 25.0,
        "metrics_before_laplacian": {
            "NCC": 0.50,
            "SSIM": 0.30,
            "Dice": 0.70,
            "MSE": 0.10,
            "PSNR": 12.0,
        },
    }
    (run_dir / "registration_metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

    with (run_dir / "registration_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["NCC", "0.6500"])
        writer.writerow(["SSIM", "0.3600"])
        writer.writerow(["Dice", "0.7600"])
        writer.writerow(["MSE", "0.0800"])
        writer.writerow(["PSNR", "13.5000"])

    report_path = build_run_report(run_dir)
    assert report_path.exists()
    assert (run_dir / "OPEN_ME_FIRST.txt").exists()
    report_html = report_path.read_text(encoding="utf-8")
    assert "Brainfast Report" in report_html
    assert "Looks improved" in report_html

    index_path = build_outputs_index(outputs_root)
    assert index_path.exists()
    assert (outputs_root / "OPEN_ME_FIRST.txt").exists()
    assert "demo_stack.tif" in index_path.read_text(encoding="utf-8")
