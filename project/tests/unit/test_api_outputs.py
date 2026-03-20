from __future__ import annotations

import csv
import json
from pathlib import Path

import project.frontend.server_context as ctx
from project.frontend.server import app


def _write_metrics_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["NCC", "0.70"])
        writer.writerow(["SSIM", "0.33"])
        writer.writerow(["Dice", "0.91"])
        writer.writerow(["MSE", "0.04"])
        writer.writerow(["PSNR", "13.5"])


def _make_registration_run(root: Path, name: str) -> Path:
    run_dir = root / name
    run_dir.mkdir()
    (run_dir / "overview.png").write_bytes(b"png")
    (run_dir / "overview_before.png").write_bytes(b"png")
    (run_dir / "report.html").write_text("<html></html>", encoding="utf-8")
    (run_dir / "registration_summary.txt").write_text("summary", encoding="utf-8")
    (run_dir / "staining_stats.json").write_text(
        json.dumps(
            {
                "atlas_coverage": 0.82,
                "staining_rate": 0.41,
                "positive_fraction_of_atlas": 0.33,
            }
        ),
        encoding="utf-8",
    )
    meta = {
        "input_source": str(root / f"{name}.tif"),
        "backend": "ants",
        "laplacian_enabled": True,
        "hemisphere": "left",
        "target_um": 25.0,
        "metrics_before_laplacian": {
            "NCC": 0.65,
            "SSIM": 0.30,
            "Dice": 0.89,
            "MSE": 0.05,
            "PSNR": 12.7,
        },
    }
    (run_dir / "registration_metadata.json").write_text(json.dumps(meta), encoding="utf-8")
    _write_metrics_csv(run_dir / "registration_metrics.csv")
    return run_dir


def test_registration_runs_api_lists_registration_reports(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ctx, "OUTPUT_DIR", tmp_path)

    run_dir = _make_registration_run(tmp_path, "demo_run")

    with app.test_client() as client:
        resp = client.get("/api/outputs/registration-runs")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["count"] == 1
        run = data["runs"][0]
        assert run["name"] == "demo_run"
        assert run["pipeline_label"] == "ANTS + Laplacian"
        assert run["verdict_title"] == "Looks improved"
        assert run["staining_stats"]["staining_rate"] == 0.41
        assert run["artifacts"]["overview"].endswith("/demo_run/overview.png")

        summary = client.get("/api/outputs/registration-run/demo_run/registration_summary.txt")
        assert summary.status_code == 200
        assert summary.get_data(as_text=True) == "summary"


def test_registration_run_pin_reorders_list(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ctx, "OUTPUT_DIR", tmp_path)

    older = _make_registration_run(tmp_path, "older_run")
    newer = _make_registration_run(tmp_path, "newer_run")
    older.touch()
    newer.touch()

    with app.test_client() as client:
        resp = client.get("/api/outputs/registration-runs")
        assert [run["name"] for run in resp.get_json()["runs"]] == ["newer_run", "older_run"]

        pin_resp = client.post("/api/outputs/registration-run/older_run/pin")
        assert pin_resp.status_code == 200

        resp = client.get("/api/outputs/registration-runs")
        runs = resp.get_json()["runs"]
        assert [run["name"] for run in runs] == ["older_run", "newer_run"]
        assert runs[0]["pinned"] is True


def test_registration_run_delete_bad_archives_run(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ctx, "OUTPUT_DIR", tmp_path)
    _make_registration_run(tmp_path, "bad_run")

    with app.test_client() as client:
        resp = client.post("/api/outputs/registration-run/bad_run/delete-bad")
        assert resp.status_code == 200
        payload = resp.get_json()
        assert payload["ok"] is True
        assert not (tmp_path / "bad_run").exists()

        archive_root = tmp_path / "archive" / "registration_runs"
        archived = list(archive_root.iterdir())
        assert len(archived) == 1
        assert archived[0].name.endswith("bad_run")

        runs_resp = client.get("/api/outputs/registration-runs")
        assert runs_resp.get_json()["count"] == 0


def test_leaf_channel_returns_empty_when_channel_file_missing(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ctx, "OUTPUT_DIR", tmp_path)

    with app.test_client() as client:
        resp = client.get("/api/outputs/leaf/red")
        assert resp.status_code == 204
        assert resp.get_data(as_text=True) == ""
