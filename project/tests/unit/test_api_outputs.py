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

    _make_registration_run(tmp_path, "demo_run")

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


def _make_job_with_channel_slices(
    tmp_path: Path, job_id: str, channels: dict[str, int], n_slices: int = 3
) -> Path:
    """Build a minimal outputs/jobs/<job>/ dir with tmp_channel/ch_N_*.tif
    files + a runtime_config that declares the channel_map.
    """
    import json as _json

    import numpy as np
    from tifffile import imwrite

    job_dir = tmp_path / "jobs" / job_id
    (job_dir / "tmp_channel").mkdir(parents=True)
    for _ch_name, ch_idx in channels.items():
        for z in range(n_slices):
            arr = (np.arange(64 * 64, dtype=np.uint16).reshape(64, 64) * (ch_idx + 1)) % 65535
            imwrite(str(job_dir / "tmp_channel" / f"ch_{ch_idx}_{z:04d}.tif"), arr)
    # Also create a runtime_config so the endpoint can resolve channel_map
    (job_dir / "runtime_configs").mkdir()
    (job_dir / "runtime_configs" / "run_config_20260420_000000.json").write_text(
        _json.dumps({"input": {"channel_map": channels}}),
        encoding="utf-8",
    )
    # Artifact marker so _outputs_root picks up this job
    (job_dir / "ants_registration").mkdir()
    return job_dir


def test_raw_channel_slice_returns_png_for_existing_channel(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ctx, "OUTPUT_DIR", tmp_path)
    _make_job_with_channel_slices(
        tmp_path, "job_dual", channels={"red": 0, "farred": 2}, n_slices=3
    )
    with app.test_client() as client:
        resp = client.get("/api/outputs/raw-channel-slice?job=job_dual&z=1&channel=farred")
        assert resp.status_code == 200
        assert resp.headers.get("Content-Type", "").startswith("image/png")
        # PNG magic number
        data = resp.get_data()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_raw_channel_slice_returns_404_for_missing_channel(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ctx, "OUTPUT_DIR", tmp_path)
    _make_job_with_channel_slices(
        tmp_path, "job_red_only", channels={"red": 0}, n_slices=3
    )
    with app.test_client() as client:
        resp = client.get("/api/outputs/raw-channel-slice?job=job_red_only&z=0&channel=farred")
        assert resp.status_code == 404


def test_raw_channel_slice_accepts_tint_and_still_returns_png(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ctx, "OUTPUT_DIR", tmp_path)
    _make_job_with_channel_slices(
        tmp_path, "job_tint", channels={"red": 0, "farred": 2}, n_slices=2
    )
    with app.test_client() as client:
        resp = client.get(
            "/api/outputs/raw-channel-slice?job=job_tint&z=0&channel=farred&tint=00ffff"
        )
        assert resp.status_code == 200
        data = resp.get_data()
        assert data[:8] == b"\x89PNG\r\n\x1a\n"


def test_channel_info_reports_present_channels(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ctx, "OUTPUT_DIR", tmp_path)
    _make_job_with_channel_slices(
        tmp_path, "job_info", channels={"red": 0, "farred": 2}, n_slices=5
    )
    with app.test_client() as client:
        resp = client.get("/api/outputs/channel-info?job=job_info")
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert set(data["channels"]) == {"red", "farred"}
        assert data["slice_count"] == 5
        assert data["channel_map"]["red"] == 0
        assert data["channel_map"]["farred"] == 2
