"""REST API tests for Phase β 3D liquify endpoints."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import project.frontend.server_context as ctx  # noqa: E402
from project.frontend.server import app  # noqa: E402


@pytest.fixture
def client(tmp_path, monkeypatch):
    # Re-root the output dir so each test starts clean and does not touch
    # real pipeline artifacts on disk.
    output_root = tmp_path / "outputs"
    output_root.mkdir()
    monkeypatch.setattr(ctx, "OUTPUT_DIR", output_root)
    # run_state may contain cross-test leakage; reset
    ctx.run_state["outputDir"] = ""
    ctx.run_state["runName"] = ""
    with app.test_client() as c:
        yield c


def test_state_empty_for_fresh_job(client):
    resp = client.get("/api/liquify-3d/state?job=test1")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert data["pair_count"] == 0
    assert data["pairs"] == []
    assert data["refined_annotation_exists"] is False


def test_add_pair_then_list(client):
    add_resp = client.post(
        "/api/liquify-3d/add-pair",
        data=json.dumps(
            {"jobId": "test2", "z": 150, "real": [100.0, 80.0], "atlas": [102.0, 81.0]}
        ),
        content_type="application/json",
    )
    assert add_resp.status_code == 200
    add_data = add_resp.get_json()
    assert add_data["ok"] is True
    assert add_data["pair_count"] == 1
    assert add_data["pair"]["z"] == 150

    state_resp = client.get("/api/liquify-3d/state?job=test2")
    state = state_resp.get_json()
    assert state["pair_count"] == 1
    assert state["pairs"][0]["real_y"] == 100.0
    assert state["pairs"][0]["atlas_x"] == 81.0


def test_add_pair_rejects_invalid_payload(client):
    resp = client.post(
        "/api/liquify-3d/add-pair",
        data=json.dumps({"jobId": "test3", "z": "not_int"}),
        content_type="application/json",
    )
    assert resp.status_code == 400
    assert resp.get_json()["ok"] is False


def test_remove_pair(client):
    for i, z in enumerate([100, 200, 300]):
        client.post(
            "/api/liquify-3d/add-pair",
            data=json.dumps(
                {"jobId": "test4", "z": z, "real": [i, i], "atlas": [i + 1, i + 1]}
            ),
            content_type="application/json",
        )
    del_resp = client.delete("/api/liquify-3d/pair/1?job=test4")
    assert del_resp.status_code == 200
    assert del_resp.get_json()["pair_count"] == 2

    state = client.get("/api/liquify-3d/state?job=test4").get_json()
    remaining_z = [p["z"] for p in state["pairs"]]
    assert remaining_z == [100, 300]


def test_clear_empties_store(client):
    client.post(
        "/api/liquify-3d/add-pair",
        data=json.dumps(
            {"jobId": "test5", "z": 1, "real": [2, 3], "atlas": [4, 5]}
        ),
        content_type="application/json",
    )
    clear_resp = client.post(
        "/api/liquify-3d/clear",
        data=json.dumps({"jobId": "test5"}),
        content_type="application/json",
    )
    assert clear_resp.status_code == 200
    assert clear_resp.get_json()["pair_count"] == 0
    state = client.get("/api/liquify-3d/state?job=test5").get_json()
    assert state["pair_count"] == 0


def test_apply_without_source_annotation_returns_404(client):
    client.post(
        "/api/liquify-3d/add-pair",
        data=json.dumps(
            {"jobId": "test6", "z": 50, "real": [10, 10], "atlas": [11, 11]}
        ),
        content_type="application/json",
    )
    resp = client.post(
        "/api/liquify-3d/apply",
        data=json.dumps({"jobId": "test6"}),
        content_type="application/json",
    )
    assert resp.status_code == 404
    assert "annotation" in resp.get_json()["error"]


def _seed_job_pair(client, job_id, z=100, real=(50.0, 80.0), atlas=(55.0, 82.0)):
    return client.post(
        "/api/liquify-3d/add-pair",
        data=json.dumps(
            {"jobId": job_id, "z": z, "real": list(real), "atlas": list(atlas)}
        ),
        content_type="application/json",
    )


# ---------------------------------------------------------------------------
# Phase γ class-prior endpoints
# ---------------------------------------------------------------------------


def test_class_prior_status_empty_class(client, tmp_path, monkeypatch):
    monkeypatch.setattr(ctx, "PROJECT_ROOT", tmp_path)
    resp = client.get("/api/liquify-3d/class-prior/status?class=ChATe27")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert data["sample_count"] == 0
    assert data["ready_for_warm_start"] is False


def test_class_prior_status_missing_class_param_returns_400(client):
    resp = client.get("/api/liquify-3d/class-prior/status")
    assert resp.status_code == 400


def test_class_prior_save_merges_current_job_pairs(client, tmp_path, monkeypatch):
    monkeypatch.setattr(ctx, "PROJECT_ROOT", tmp_path)
    job_id = "sampleA"
    _seed_job_pair(client, job_id, z=100, real=(55, 82), atlas=(50, 80))
    save_resp = client.post(
        "/api/liquify-3d/class-prior/save",
        data=json.dumps({"jobId": job_id, "class": "ChATe27"}),
        content_type="application/json",
    )
    assert save_resp.status_code == 200, save_resp.get_json()
    sd = save_resp.get_json()
    assert sd["sample_count"] == 1
    assert sd["merged_pair_count"] == 1
    assert sd["ready_for_warm_start"] is False

    status = client.get("/api/liquify-3d/class-prior/status?class=ChATe27").get_json()
    assert status["sample_count"] == 1
    assert status["entry_count"] == 1


def test_class_prior_save_rejects_empty_job(client, tmp_path, monkeypatch):
    monkeypatch.setattr(ctx, "PROJECT_ROOT", tmp_path)
    resp = client.post(
        "/api/liquify-3d/class-prior/save",
        data=json.dumps({"jobId": "emptyJob", "class": "ChATe27"}),
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_class_prior_apply_warm_start_below_threshold_returns_404(client, tmp_path, monkeypatch):
    monkeypatch.setattr(ctx, "PROJECT_ROOT", tmp_path)
    # Only 1 sample contributes — below MIN_SAMPLES_FOR_APPLY=3
    _seed_job_pair(client, "sA", z=100, real=(55, 82), atlas=(50, 80))
    client.post(
        "/api/liquify-3d/class-prior/save",
        data=json.dumps({"jobId": "sA", "class": "ChATe27"}),
        content_type="application/json",
    )
    resp = client.post(
        "/api/liquify-3d/class-prior/apply-warm-start",
        data=json.dumps({"jobId": "newJob", "class": "ChATe27"}),
        content_type="application/json",
    )
    assert resp.status_code == 404


def test_class_prior_apply_warm_start_after_three_samples(client, tmp_path, monkeypatch):
    monkeypatch.setattr(ctx, "PROJECT_ROOT", tmp_path)
    for i, jid in enumerate(["s1", "s2", "s3"]):
        _seed_job_pair(client, jid, z=100, real=(55 + i, 82 + i), atlas=(50, 80))
        client.post(
            "/api/liquify-3d/class-prior/save",
            data=json.dumps({"jobId": jid, "class": "ChATe27"}),
            content_type="application/json",
        )

    resp = client.post(
        "/api/liquify-3d/class-prior/apply-warm-start",
        data=json.dumps({"jobId": "newJob", "class": "ChATe27"}),
        content_type="application/json",
    )
    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    assert data["pair_count"] == 1

    # The new job should now have the prior pre-populated
    state = client.get("/api/liquify-3d/state?job=newJob").get_json()
    assert state["pair_count"] == 1
    assert state["pairs"][0]["z"] == 100


def test_class_prior_apply_warm_start_refuses_existing_pairs_without_force(
    client, tmp_path, monkeypatch
):
    monkeypatch.setattr(ctx, "PROJECT_ROOT", tmp_path)
    for i, jid in enumerate(["s1", "s2", "s3"]):
        _seed_job_pair(client, jid, z=100, real=(55 + i, 82 + i), atlas=(50, 80))
        client.post(
            "/api/liquify-3d/class-prior/save",
            data=json.dumps({"jobId": jid, "class": "ChATe27"}),
            content_type="application/json",
        )
    # Seed newJob with a manual pair the user already laid down
    _seed_job_pair(client, "newJob", z=200, real=(10, 10), atlas=(9, 9))

    resp = client.post(
        "/api/liquify-3d/class-prior/apply-warm-start",
        data=json.dumps({"jobId": "newJob", "class": "ChATe27"}),
        content_type="application/json",
    )
    assert resp.status_code == 409

    force_resp = client.post(
        "/api/liquify-3d/class-prior/apply-warm-start",
        data=json.dumps({"jobId": "newJob", "class": "ChATe27", "force": True}),
        content_type="application/json",
    )
    assert force_resp.status_code == 200


def test_state_includes_guidance_when_source_annotation_missing(client):
    resp = client.get("/api/liquify-3d/state?job=no-source")
    data = resp.get_json()
    assert data["source_annotation_available"] is False
    assert data["guidance"] is not None
    assert "annotation_registered" in data["guidance"]


def test_progress_endpoint_empty_for_fresh_job(client):
    resp = client.get("/api/liquify-3d/progress?job=fresh")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    # No progress yet → no stage/percent keys
    assert "stage" not in data or data.get("percent", None) is None


def test_progress_endpoint_returns_latest_snapshot(client, tmp_path):
    from project.scripts.liquify_progress import write_liquify_progress

    job_id = "prog-test"
    job_dir = ctx._job_output_dir(job_id)
    write_liquify_progress(
        job_dir=job_dir,
        stage="solve",
        stage_index=2,
        stage_count=4,
        percent=42,
        message="Solving axis 1",
    )
    resp = client.get(f"/api/liquify-3d/progress?job={job_id}")
    data = resp.get_json()
    assert data["percent"] == 42
    assert data["stage"] == "solve"
    assert data["message"] == "Solving axis 1"


def test_class_prior_coverage_returns_per_z_bin_counts(client, tmp_path, monkeypatch):
    """Coverage heatmap data: array of {z, count} for every contributing
    landmark, so the frontend can render where corrections are dense vs sparse."""
    monkeypatch.setattr(ctx, "PROJECT_ROOT", tmp_path)
    cls = "ChATe27"
    # Seed three samples — z=100 has 2 contributions, z=200 has 1.
    for jid, z in [("s1", 100), ("s2", 100), ("s3", 200)]:
        _seed_job_pair(client, jid, z=z, real=(55, 82), atlas=(50, 80))
        client.post(
            "/api/liquify-3d/class-prior/save",
            data=json.dumps({"jobId": jid, "class": cls}),
            content_type="application/json",
        )

    resp = client.get(f"/api/liquify-3d/class-prior/coverage?class={cls}")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    bins = {row["z"]: row["count"] for row in data["bins"]}
    assert bins[100] == 2  # both s1 and s2 merged into one entry, n=2
    # Wait: actually the merge collapses pairs at the same atlas voxel into
    # ONE entry with n=2. So we should see 1 entry at z=100 with count 2.
    assert len([b for b in data["bins"] if b["z"] == 100]) == 1
    assert bins[200] == 1


def test_class_prior_coverage_unknown_class_returns_empty(client, tmp_path, monkeypatch):
    monkeypatch.setattr(ctx, "PROJECT_ROOT", tmp_path)
    resp = client.get("/api/liquify-3d/class-prior/coverage?class=Nonexistent")
    assert resp.status_code == 200
    data = resp.get_json()
    assert data["ok"] is True
    assert data["bins"] == []


def test_class_registry_list_returns_known_classes(client, tmp_path, monkeypatch):
    """Endpoint must surface classes from BOTH the registry config and on-disk priors."""
    import json as _json
    monkeypatch.setattr(ctx, "PROJECT_ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "sample_class_registry.json").write_text(
        _json.dumps({"patterns": [{"match": r"^pv", "class": "PVe3"}]}),
        encoding="utf-8",
    )
    pri = tmp_path / "train_data_set" / "class_priors" / "ChATe27"
    pri.mkdir(parents=True)
    (pri / "landmark_prior.csv").write_text(
        "z,atlas_y,atlas_x,sum_dy,sum_dx,sum_sq_dy,sum_sq_dx,n\n",
        encoding="utf-8",
    )

    resp = client.get("/api/liquify-3d/class-registry/list")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "PVe3" in data["classes"]
    assert "ChATe27" in data["classes"]


def test_class_registry_detect_matches_pattern(client, tmp_path, monkeypatch):
    import json as _json
    monkeypatch.setattr(ctx, "PROJECT_ROOT", tmp_path)
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "sample_class_registry.json").write_text(
        _json.dumps({"patterns": [{"match": r"^3[59]_", "class": "ChATe27"}]}),
        encoding="utf-8",
    )

    resp = client.get("/api/liquify-3d/class-registry/detect?sampleId=35_C0")
    assert resp.status_code == 200
    assert resp.get_json()["class"] == "ChATe27"

    resp_miss = client.get("/api/liquify-3d/class-registry/detect?sampleId=42_C0")
    assert resp_miss.get_json()["class"] is None


def test_qc_done_marks_job_and_writes_marker_file(client, tmp_path, monkeypatch):
    monkeypatch.setattr(ctx, "PROJECT_ROOT", tmp_path)
    job_id = "qc-1"
    resp = client.post(
        "/api/liquify-3d/qc-done",
        data=json.dumps(
            {
                "jobId": job_id,
                "metrics": {"NCC": 0.42, "Dice": 0.81},
                "note": "first ChATe27 ok",
            }
        ),
        content_type="application/json",
    )
    assert resp.status_code == 200, resp.get_json()
    body = resp.get_json()
    assert body["ok"] is True
    assert body["jobId"] == job_id
    marker = ctx._job_file(job_id, "qc_done.json")
    assert marker.exists()
    payload = json.loads(marker.read_text(encoding="utf-8"))
    assert payload["metrics"]["NCC"] == 0.42
    assert payload["note"] == "first ChATe27 ok"
    assert payload["timestamp"]


def test_qc_done_status_endpoint_returns_marker_when_present(client, tmp_path, monkeypatch):
    monkeypatch.setattr(ctx, "PROJECT_ROOT", tmp_path)
    job_id = "qc-2"
    client.post(
        "/api/liquify-3d/qc-done",
        data=json.dumps({"jobId": job_id, "metrics": {"NCC": 0.5}}),
        content_type="application/json",
    )
    resp = client.get(f"/api/liquify-3d/qc-status?job={job_id}")
    assert resp.status_code == 200
    s = resp.get_json()
    assert s["done"] is True
    assert s["metrics"]["NCC"] == 0.5


def test_qc_status_returns_done_false_for_fresh_job(client):
    resp = client.get("/api/liquify-3d/qc-status?job=never-touched")
    assert resp.status_code == 200
    s = resp.get_json()
    assert s["done"] is False


def test_qc_done_appends_to_class_prior_sample_log_when_class_provided(
    client, tmp_path, monkeypatch
):
    """When the QC-done call carries a className, append the metrics into
    the corresponding class prior's sample_log.jsonl so γ history captures
    the registration quality each contributing sample reached."""
    monkeypatch.setattr(ctx, "PROJECT_ROOT", tmp_path)
    job_id = "qc-3"
    cls = "ChATe27"
    # Seed the prior with the job's pairs first so the sample_log exists
    _seed_job_pair(client, job_id, z=100, real=(55, 82), atlas=(50, 80))
    client.post(
        "/api/liquify-3d/class-prior/save",
        data=json.dumps({"jobId": job_id, "class": cls}),
        content_type="application/json",
    )
    client.post(
        "/api/liquify-3d/qc-done",
        data=json.dumps(
            {"jobId": job_id, "className": cls, "metrics": {"NCC": 0.55, "Dice": 0.78}}
        ),
        content_type="application/json",
    )
    log_path = tmp_path / "train_data_set" / "class_priors" / cls / "sample_log.jsonl"
    assert log_path.exists()
    lines = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    # Two records: one from class-prior/save (no metrics) + one from qc-done
    qc_records = [r for r in lines if (r.get("metrics") or {}).get("NCC") == 0.55]
    assert len(qc_records) == 1, f"expected 1 qc-done record in sample log, got: {lines}"


def test_finalize_endpoint_missing_refined_annotation_returns_404(client):
    resp = client.post(
        "/api/liquify-3d/finalize",
        data=json.dumps({"jobId": "no-refined"}),
        content_type="application/json",
    )
    assert resp.status_code == 404
    assert "annotation_refined_liquify3d" in resp.get_json()["error"]


def test_finalize_endpoint_happy_path(client, tmp_path):
    """End-to-end finalize through the REST layer with a fully seeded job dir.

    This is the acceptance test for close-the-loop: after hitting /finalize
    the job directory must contain ``cell_counts_hierarchy_liquify3d.csv``
    with at least one row.
    """
    import pandas as pd
    from tifffile import imwrite

    job_id = "close-loop-ok"
    job_dir = ctx._job_output_dir(job_id)

    # 1. Seed the refined annotation at the expected path
    shape = (3, 5, 5)
    ann = np.full(shape, 2, dtype=np.int32)
    nib.save(
        nib.Nifti1Image(ann, np.eye(4)),
        str(job_dir / "annotation_refined_liquify3d.nii.gz"),
    )

    # 2. Seed tmp_merged with matching-count slices so _resolve_real_slice_paths
    #    can glob them
    merged_dir = job_dir / "tmp_merged"
    merged_dir.mkdir()
    for z in range(shape[0]):
        imwrite(
            str(merged_dir / f"merged_{z:04d}.tif"),
            np.full((5, 5), 100, dtype=np.uint16),
        )

    # 3. Seed cells_dedup.csv with 2 cells in slice 1
    pd.DataFrame(
        {
            "cell_id": [0, 1],
            "slice_id": [1, 1],
            "x": [2.0, 3.0],
            "y": [2.0, 3.0],
        }
    ).to_csv(job_dir / "cells_dedup.csv", index=False)

    resp = client.post(
        "/api/liquify-3d/finalize",
        data=json.dumps({"jobId": job_id, "pixelSizeUm": 5.0}),
        content_type="application/json",
    )
    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    assert data["ok"] is True
    assert data["mapped_count"] == 2

    hierarchy_csv = Path(data["cell_counts_hierarchy_csv"])
    assert hierarchy_csv.exists()
    # cell_counts_liquify3d.csv is present alongside the original
    assert (job_dir / "cell_counts_hierarchy_liquify3d.csv").exists()


def test_apply_with_source_annotation_writes_refined_nifti(client, tmp_path):
    job_id = "test7"
    # Place a small fake registered annotation where the endpoint expects it.
    job_dir = ctx._job_output_dir(job_id)
    ants_dir = job_dir / "ants_registration"
    ants_dir.mkdir(parents=True, exist_ok=True)
    ann = np.zeros((4, 6, 6), dtype=np.int32)
    ann[:, :, 3] = 7
    nib.save(
        nib.Nifti1Image(ann, np.eye(4)), str(ants_dir / "annotation_registered.nii.gz")
    )

    client.post(
        "/api/liquify-3d/add-pair",
        data=json.dumps(
            {"jobId": job_id, "z": 2, "real": [3.0, 4.0], "atlas": [3.0, 3.0]}
        ),
        content_type="application/json",
    )

    resp = client.post(
        "/api/liquify-3d/apply",
        data=json.dumps({"jobId": job_id, "rtol": 1e-2, "maxiter": 200}),
        content_type="application/json",
    )
    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    assert data["ok"] is True
    assert data["pair_count"] == 1
    assert Path(data["output_path"]).exists()

    # state endpoint should now report refined_annotation_exists True
    state = client.get(f"/api/liquify-3d/state?job={job_id}").get_json()
    assert state["refined_annotation_exists"] is True
