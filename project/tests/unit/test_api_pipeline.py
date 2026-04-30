from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import project.frontend.server_context as ctx  # noqa: E402
from project.frontend.server import app  # noqa: E402
from project.scripts.paths import RunPaths  # noqa: E402


def _minimal_cfg(input_dir: Path) -> dict:
    return {
        "project": {"name": "test_project"},
        "input": {
            "slice_dir": str(input_dir),
            "slice_glob": "z*.tif",
            "pixel_size_um_xy": 5.0,
            "slice_spacing_um": 25.0,
            "channel_map": {"red": 0},
            "active_channel": "red",
        },
        "registration": {"atlas_z_refine_range": 0},
        "detection": {"primary_model": "fallback"},
        "dedup": {"neighbor_slices": 1, "r_xy_um": 8.0},
        "outputs": {
            "leaf_csv": "outputs/leaf.csv",
            "hierarchy_csv": "outputs/hierarchy.csv",
            "qc_dir": "outputs/qc",
        },
    }


def test_preflight_returns_structured_warning(tmp_path: Path, monkeypatch) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    atlas_path = tmp_path / "annotation_25.nii.gz"
    atlas_path.write_text("atlas", encoding="utf-8")
    struct_path = tmp_path / "allen.csv"
    struct_path.write_text("id,name\n1,root\n", encoding="utf-8")
    config_path = tmp_path / "run_config.json"
    config_path.write_text(json.dumps(_minimal_cfg(input_dir)), encoding="utf-8")
    # OUTPUT_DIR must contain the config_path so _resolve_config_path() containment check passes
    monkeypatch.setattr(ctx, "OUTPUT_DIR", tmp_path)

    payload = {
        "configPath": str(config_path),
        "inputDir": str(input_dir),
        "atlasPath": str(atlas_path),
        "structPath": str(struct_path),
        "channels": ["red"],
        "params": {
            "pixelSizeUm": "5.0",
            "alignMode": "affine",
            "atlasPath": str(atlas_path),
            "structPath": str(struct_path),
        },
    }

    with app.test_client() as client:
        resp = client.post("/api/pipeline/preflight", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert any(
            issue["field"] == "registration.atlas_z_refine_range"
            and issue["severity"] == "warning"
            for issue in data["issues"]
        )


def test_preflight_returns_structured_error_for_invalid_runtime_config(
    tmp_path: Path, monkeypatch
) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    atlas_path = tmp_path / "annotation_25.nii.gz"
    atlas_path.write_text("atlas", encoding="utf-8")
    struct_path = tmp_path / "allen.csv"
    struct_path.write_text("id,name\n1,root\n", encoding="utf-8")
    cfg = _minimal_cfg(input_dir)
    cfg["input"]["pixel_size_um_xy"] = 0
    config_path = tmp_path / "bad_run_config.json"
    config_path.write_text(json.dumps(cfg), encoding="utf-8")
    # OUTPUT_DIR must contain the config_path so containment check passes
    monkeypatch.setattr(ctx, "OUTPUT_DIR", tmp_path)

    payload = {
        "configPath": str(config_path),
        "inputDir": str(input_dir),
        "atlasPath": str(atlas_path),
        "structPath": str(struct_path),
        "channels": ["red"],
        "params": {
            "pixelSizeUm": "",
            "alignMode": "affine",
            "atlasPath": str(atlas_path),
            "structPath": str(struct_path),
        },
    }

    with app.test_client() as client:
        resp = client.post("/api/pipeline/preflight", json=payload)
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is False
        assert any(
            issue["field"] == "input.pixel_size_um_xy" and issue["severity"] == "error"
            for issue in data["issues"]
        )


def test_error_log_and_status_expose_structured_progress(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(ctx, "OUTPUT_DIR", tmp_path)
    job_id = "job_test123"
    monkeypatch.setattr(
        ctx,
        "_job_states",
        {
            ctx.DEFAULT_JOB_ID: ctx.run_state,
            job_id: {
            "running": True,
            "done": False,
            "error": None,
            "logs": ["[PROGRESS:step=3/6:phase=registration:slices=4/12] Processing"],
            "errors": [
                {
                    "timestamp": "2026-03-19T10:00:00",
                    "message": "Registration score dropped below threshold.",
                    "step": "registration",
                    "recoverable": False,
                    "source": "backend",
                }
            ],
            "channels": ["red"],
            "proc": None,
            "current_channel": "red",
            "history": [],
            "config_path": None,
            "startEpoch": 1234567890,
            "job_id": job_id,
            "outputs_dir": str(tmp_path / "jobs" / job_id),
            "progress": {
                "phase": "registration",
                "stepCurrent": 3,
                "stepTotal": 6,
                "slicesDone": 4,
                "slicesTotal": 12,
                "message": "Processing slice 4 / 12",
            },
            },
        },
    )

    with app.test_client() as client:
        error_resp = client.get(f"/api/error-log?job={job_id}")
        assert error_resp.status_code == 200
        error_data = error_resp.get_json()
        assert error_data["count"] == 1
        assert error_data["errors"][0]["step"] == "registration"

        status_resp = client.get(f"/api/status?job={job_id}")
        assert status_resp.status_code == 200
        status_data = status_resp.get_json()
        assert status_data["jobId"] == job_id
        assert status_data["progress"]["phase"] == "registration"
        assert status_data["progress"]["stepCurrent"] == 3
        assert status_data["slicesDone"] == 4
        assert status_data["slicesTotal"] == 12


def test_status_exposes_eta_when_pipeline_progress_file_present(tmp_path: Path, monkeypatch) -> None:
    """When pipeline_progress.json exists for a job, /api/status should
    include both the on-disk stage info and a non-null ``eta`` block.
    """
    monkeypatch.setattr(ctx, "OUTPUT_DIR", tmp_path)
    job_id = "job_eta_check"
    job_dir = tmp_path / "jobs" / job_id
    job_dir.mkdir(parents=True)

    # Simulate a pipeline 30 minutes in, mid-ANTs at 45%.
    import time as _time

    from project.scripts.pipeline_progress import write_stage_progress

    now = _time.time()
    stage_started = now - 1800  # ANTs running for 30 min
    run_started = now - 1900    # plus 100s of stages 1-2
    progress_path = job_dir / "pipeline_progress.json"
    # Write directly (preserves controlled timestamps)
    progress_path.write_text(
        '{"stageName":"ANTS Registration","stageIndex":3,"stageCount":6,'
        '"percent":45,"message":"Running ANTs",'
        f'"ts":{now},"stageStartedTs":{stage_started},"runStartedTs":{run_started},'
        '"artifacts":{}}',
        encoding="utf-8",
    )

    monkeypatch.setattr(
        ctx,
        "_job_states",
        {
            ctx.DEFAULT_JOB_ID: ctx.run_state,
            job_id: {
                "running": True, "done": False, "error": None,
                "logs": [], "errors": [], "channels": ["red"], "proc": None,
                "current_channel": "red", "history": [], "config_path": None,
                "startEpoch": int(run_started), "job_id": job_id,
                "outputs_dir": str(job_dir),
                "progress": {"phase": "registration", "stepCurrent": 3, "stepTotal": 6,
                             "slicesDone": 0, "slicesTotal": 111, "message": ""},
            },
        },
    )

    with app.test_client() as client:
        status_resp = client.get(f"/api/status?job={job_id}")
        assert status_resp.status_code == 200
        data = status_resp.get_json()

        # On-disk progress fields are now exposed
        assert data["progress"]["stageName"] == "ANTS Registration"
        assert data["progress"]["stageIndex"] == 3
        assert data["progress"]["stagePercent"] == 45

        # ETA block is populated
        assert data["eta"] is not None
        assert data["eta"]["etr_total_s"] > 0
        assert data["eta"]["method"] in {"baseline_only", "linear_in_stage", "linear_corrected"}
        # ANTs at 45% with 1800s elapsed → roughly 2200s remaining in stage,
        # plus Lap 30s + Truth 7×111=777s + Quant 16.5×111=1832s = 4839s total
        # (with self-correction baseline-through-current ≈ (60+60+2000×0.45) = 1020s,
        # actual elapsed 1900s → correction ≈ 1.86×, so remaining ≈ 4839 × 1.86 ≈ 9000s)
        assert 1000 < data["eta"]["etr_total_s"] < 20000


def test_job_output_dirs_do_not_overlap() -> None:
    left = ctx._job_output_dir("job_alpha")
    right = ctx._job_output_dir("job_beta")
    assert left != right
    assert left.name == "job_alpha"
    assert right.name == "job_beta"


def test_runpaths_accepts_custom_outputs_dir(tmp_path: Path) -> None:
    cfg = _minimal_cfg(tmp_path / "input")
    custom_root = tmp_path / "jobs" / "job_alpha"
    paths = RunPaths.from_project_root(tmp_path, cfg, outputs_dir=custom_root)
    assert paths.outputs == custom_root
    assert paths.cells_detected == custom_root / "cells_detected.csv"
    assert paths.registered_slices == custom_root / "registered_slices"


# ---------------------------------------------------------------------------
# Task 1 — Shared runtime-state path contract
# ---------------------------------------------------------------------------


def test_runpaths_exposes_shared_state_root_under_outputs(tmp_path: Path) -> None:
    """Calibration, class priors, and Cellpose training samples must all live
    under a shared state root (``outputs/state/``) rather than dirtying the
    tracked source tree (``train_data_set/``, ``cellpose_training/``).
    """
    cfg = _minimal_cfg(tmp_path / "input")
    job_out = tmp_path / "jobs" / "job_alpha"
    paths = RunPaths.from_project_root(tmp_path, cfg, outputs_dir=job_out)

    # Canonical shared state layout — not under job_out (must survive jobs)
    expected_root = tmp_path / "outputs" / "state"
    assert paths.state_root == expected_root
    assert paths.calibration_samples_dir == expected_root / "calibration" / "samples"
    assert (
        paths.calibration_tuned_json
        == expected_root / "calibration" / "trainset_tuned_params.json"
    )
    assert paths.class_priors_dir == expected_root / "class_priors"
    assert paths.cellpose_training_dir == expected_root / "cellpose_training"


def test_runpaths_state_paths_never_point_inside_source_tree(tmp_path: Path) -> None:
    """Regression guard: these paths used to write under
    ``PROJECT_ROOT / 'train_data_set'`` and ``PROJECT_ROOT / 'cellpose_training'``,
    which polluted the git-tracked source tree. They must now live under
    ``outputs/state/``.
    """
    cfg = _minimal_cfg(tmp_path / "input")
    paths = RunPaths.from_project_root(tmp_path, cfg)

    forbidden = {tmp_path / "train_data_set", tmp_path / "cellpose_training"}
    for p in (
        paths.state_root,
        paths.calibration_samples_dir,
        paths.calibration_tuned_json,
        paths.class_priors_dir,
        paths.cellpose_training_dir,
    ):
        for fb in forbidden:
            assert fb not in p.parents, f"{p} unexpectedly lives inside {fb}"


def test_runpaths_state_root_overridable_via_env(tmp_path: Path, monkeypatch) -> None:
    """A deployment might want state on a separate disk. Setting
    BRAINFAST_STATE_DIR should redirect every state subpath accordingly.
    """
    alt_root = tmp_path / "alt_state_disk"
    monkeypatch.setenv("BRAINFAST_STATE_DIR", str(alt_root))
    cfg = _minimal_cfg(tmp_path / "input")
    paths = RunPaths.from_project_root(tmp_path, cfg)

    assert paths.state_root == alt_root
    assert paths.calibration_samples_dir == alt_root / "calibration" / "samples"
    assert paths.class_priors_dir == alt_root / "class_priors"
    assert paths.cellpose_training_dir == alt_root / "cellpose_training"


def test_info_reads_version_json() -> None:
    # Ensure PROJECT_ROOT points to the real project dir so version.json is found.
    # Read the expected version dynamically — the contract here is "/api/info
    # echoes whatever version.json says", not "version.json is some specific
    # string"; legitimate version bumps shouldn't require touching this test.
    import json

    import project.frontend.server_context as ctx

    saved = ctx.PROJECT_ROOT
    ctx.PROJECT_ROOT = Path(__file__).resolve().parents[2]
    try:
        version_payload = json.loads(
            (ctx.PROJECT_ROOT / "version.json").read_text(encoding="utf-8")
        )
        with app.test_client() as client:
            resp = client.get("/api/info")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["version"] == version_payload["version"]
    finally:
        ctx.PROJECT_ROOT = saved
