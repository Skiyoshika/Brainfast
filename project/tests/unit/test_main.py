from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from project.scripts import main


def test_refresh_demo_visuals_passes_active_output_and_raw_dir(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    refresh_script = project_root / "scripts" / "refresh_demo.py"
    refresh_script.parent.mkdir(parents=True)
    refresh_script.write_text("# test stub\n", encoding="utf-8")

    output_dir = tmp_path / "outputs" / "run_001"
    output_dir.mkdir(parents=True)
    raw_dir = tmp_path / "data" / "brain_001"
    raw_dir.mkdir(parents=True)

    recorded: dict[str, object] = {}

    def fake_run(cmd, cwd=None, timeout=None):
        recorded["cmd"] = list(cmd)
        recorded["cwd"] = cwd
        recorded["timeout"] = timeout
        return object()

    monkeypatch.setattr(main.subprocess, "run", fake_run)

    main._refresh_demo_visuals(project_root, output_dir, raw_dir)

    assert recorded["cmd"] == [
        sys.executable,
        str(refresh_script),
        "--outputs-dir",
        str(output_dir),
        "--raw-dir",
        str(raw_dir),
    ]
    assert recorded["cwd"] == str(project_root)
    assert recorded["timeout"] == 180


def test_refresh_demo_visuals_skips_when_script_missing(tmp_path, monkeypatch):
    project_root = tmp_path / "project"
    project_root.mkdir()
    output_dir = tmp_path / "outputs" / "run_001"
    output_dir.mkdir(parents=True)

    called = False

    def fake_run(*args, **kwargs):
        nonlocal called
        called = True
        raise AssertionError("subprocess.run should not be called when refresh_demo.py is missing")

    monkeypatch.setattr(main.subprocess, "run", fake_run)

    main._refresh_demo_visuals(project_root, output_dir, None)

    assert called is False


def test_run_real_input_routes_whole_brain_mode_to_3d_orchestrator(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "z0000.tif").write_bytes(b"")

    route_hit: dict[str, object] = {}

    monkeypatch.setattr(
        main, "_collect_slice_files", lambda *_args, **_kwargs: [input_dir / "z0000.tif"]
    )
    monkeypatch.setattr(
        main,
        "_extract_channel_to_tmp",
        lambda *_args, **_kwargs: [tmp_path / "tmp_channel" / "z0000.tif"],
    )
    monkeypatch.setattr(
        main,
        "merge_every_n_slices",
        lambda *_args, **_kwargs: [tmp_path / "tmp_merged" / "z0000.tif"],
    )

    def fake_quantify_against_exported_truth(**kwargs):
        route_hit["quantify_kwargs"] = kwargs
        return {"truth_source": "3d_registered_volume", "cells_mapped_csv": "mapped.csv"}

    monkeypatch.setattr(
        main,
        "_quantify_against_exported_truth",
        fake_quantify_against_exported_truth,
        raising=False,
    )

    def fake_run_whole_brain_3d(**kwargs):
        route_hit["kwargs"] = kwargs
        route_hit["quantify_result"] = kwargs["cfg"]["quantify_fn"](
            truth_rows=[{"slice_id": 0}],
            cfg=kwargs["cfg"],
            outputs_dir=kwargs["outputs_dir"],
        )
        return {"truth_source": "3d_registered_volume"}

    monkeypatch.setattr(main, "run_whole_brain_3d", fake_run_whole_brain_3d, raising=False)

    result = main.run_real_input(
        cfg={
            "input": {
                "slice_glob": "*.tif",
                "slice_interval_n": 1,
            },
            "registration": {
                "scope": "whole",
                "whole_brain_backend": "miki_3d",
            },
        },
        input_dir=input_dir,
        outputs_dir=tmp_path / "outputs",
    )

    assert result["truth_source"] == "3d_registered_volume"
    assert route_hit["kwargs"]["outputs_dir"].name == "outputs"
    assert route_hit["quantify_result"]["truth_source"] == "3d_registered_volume"
    assert route_hit["quantify_kwargs"]["outputs_dir"].name == "outputs"


def test_run_real_input_preserves_truth_export_mode_settings(tmp_path, monkeypatch):
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "z0000.tif").write_bytes(b"")

    route_hit: dict[str, object] = {}

    monkeypatch.setattr(
        main, "_collect_slice_files", lambda *_args, **_kwargs: [input_dir / "z0000.tif"]
    )
    monkeypatch.setattr(
        main,
        "_extract_channel_to_tmp",
        lambda *_args, **_kwargs: [tmp_path / "tmp_channel" / "z0000.tif"],
    )
    monkeypatch.setattr(
        main,
        "merge_every_n_slices",
        lambda *_args, **_kwargs: [tmp_path / "tmp_merged" / "z0000.tif"],
    )
    monkeypatch.setattr(
        main,
        "_load_tuned_overlay_params",
        lambda *_args, **_kwargs: ({"from_tuned": True}, "contain", 2),
    )
    monkeypatch.setattr(
        main,
        "_snapshot_tuned_params_into_job",
        lambda *_args, **_kwargs: None,
    )

    def fake_run_whole_brain_3d(**kwargs):
        route_hit["truth_export"] = dict(kwargs["cfg"]["truth_export"])
        return {"truth_source": "3d_registered_volume"}

    monkeypatch.setattr(main, "run_whole_brain_3d", fake_run_whole_brain_3d, raising=False)

    main.run_real_input(
        cfg={
            "input": {
                "slice_glob": "*.tif",
                "slice_interval_n": 1,
            },
            "registration": {
                "scope": "whole",
                "whole_brain_backend": "miki_3d",
            },
            "truth_export": {
                "overlay_stride": 20,
                "write_overlays": True,
                "profile": "fast_qc",
            },
        },
        input_dir=input_dir,
        outputs_dir=tmp_path / "outputs",
    )

    truth = route_hit["truth_export"]
    assert truth["overlay_stride"] == 20
    assert truth["write_overlays"] is True
    assert truth["profile"] == "fast_qc"
    assert truth["warp_params"] == {"from_tuned": True}
    assert truth["fit_mode"] == "contain"
    assert truth["edge_smooth_iter"] == 2


def test_quantify_passes_axis_alignment_matrix_to_cell_to_ccf(tmp_path, monkeypatch):
    import nibabel as nib
    import pandas as pd
    import scripts.atlas_mapper as atlas_mapper
    import scripts.cell_to_ccf as cell_to_ccf
    from tifffile import imwrite

    matrix = np.eye(4, dtype=np.float32)
    matrix[0, 1] = 0.25
    axis_path = tmp_path / "axisAlignA.npy"
    np.save(axis_path, matrix)

    sample_volume = tmp_path / "sample.nii.gz"
    ccf_template = tmp_path / "template.nii.gz"
    ccf_annotation = tmp_path / "annotation.nii.gz"
    for path in (sample_volume, ccf_template, ccf_annotation):
        nib.save(nib.Nifti1Image(np.zeros((2, 5, 5), dtype=np.float32), np.eye(4)), str(path))

    real_slice = tmp_path / "real.tif"
    registered_label = tmp_path / "registered_label.tif"
    imwrite(str(real_slice), np.zeros((5, 5), dtype=np.uint8))
    imwrite(str(registered_label), np.zeros((5, 5), dtype=np.uint16))

    monkeypatch.setattr(main, "_resolve_structure_source", lambda _root: tmp_path / "structures.csv")
    monkeypatch.setattr(
        main,
        "detect_cells",
        lambda *_args, **_kwargs: pd.DataFrame({"x": [2.0], "y": [1.0], "score": [0.9]}),
    )
    monkeypatch.setattr(
        main,
        "apply_dedup_kdtree",
        lambda mapped, **_kwargs: (mapped.copy(), {"kept": len(mapped)}),
    )
    monkeypatch.setattr(
        main,
        "aggregate_by_region",
        lambda _df: (
            pd.DataFrame({"region_id": [1], "count": [1]}),
            pd.DataFrame({"region_id": [1], "count": [1]}),
        ),
    )
    monkeypatch.setattr(main, "write_outputs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(atlas_mapper, "_attach_structure_metadata", lambda df, _csv: df)

    captured: dict[str, object] = {}

    def fake_map_cells_via_ccf_transform(
        cells,
        *,
        sample_volume_path,
        ccf_annotation_path,
        inverse_transforms,
        pixel_size_um,
        ccf_template_path,
        axis_align_matrix,
    ):
        captured["axis_align_matrix"] = axis_align_matrix
        return pd.DataFrame(
            {
                "cell_id": cells["cell_id"].to_numpy(),
                "slice_id": cells["slice_id"].to_numpy(),
                "x": cells["x"].to_numpy(),
                "y": cells["y"].to_numpy(),
                "region_id": [1],
                "mapping_status": ["mapped"],
            }
        )

    monkeypatch.setattr(cell_to_ccf, "map_cells_via_ccf_transform", fake_map_cells_via_ccf_transform)

    main._quantify_against_exported_truth(
        truth_rows=[
            {
                "slice_id": 0,
                "real_slice_path": str(real_slice),
                "registered_label_path": str(registered_label),
                "overlay_path": "",
            }
        ],
        cfg={
            "input": {"pixel_size_um_xy": 1.0, "slice_spacing_um": 1.0},
            "registration": {
                "use_cell_to_ccf_mapping": True,
                "template_path": str(ccf_template),
                "annotation_path": str(ccf_annotation),
            },
            "dedup": {},
        },
        outputs_dir=tmp_path / "out",
        ants_meta={"inverse_transforms": [str(tmp_path / "inv.mat")], "fixed_image": str(ccf_template)},
        volume_meta={"volume_path": str(sample_volume), "axis_align_matrix_path": str(axis_path)},
    )

    np.testing.assert_array_equal(captured["axis_align_matrix"], matrix)


def test_load_tuned_overlay_params_falls_back_to_shared_state(tmp_path, monkeypatch):
    """Task 2 — when the job's outputs_dir has no ``trainset_tuned_params.json``
    but the shared-state calibration JSON exists under
    ``outputs/state/calibration/trainset_tuned_params.json``, the loader must
    return the shared values (so a UI-triggered learn survives across jobs).
    """
    import json

    from project.scripts import paths as paths_mod
    from project.scripts.main import _load_tuned_overlay_params

    # Point the shared-state root at tmp for this test
    monkeypatch.setenv("BRAINFAST_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(paths_mod, "resolve_state_root", lambda _pr: tmp_path / "state")

    # Project root + empty per-job outputs — no local tuned JSON
    project_root = tmp_path / "proj"
    project_root.mkdir()
    job_outputs = tmp_path / "jobs" / "job_a"
    job_outputs.mkdir(parents=True)

    # Seed the SHARED tuned JSON with a distinctive payload
    shared = tmp_path / "state" / "calibration" / "trainset_tuned_params.json"
    shared.parent.mkdir(parents=True, exist_ok=True)
    shared.write_text(
        json.dumps(
            {
                "warpParams": {"shared_marker": 7},
                "fitMode": "contain",
                "edgeSmoothIter": 2,
            }
        ),
        encoding="utf-8",
    )

    warp, fit, edge = _load_tuned_overlay_params(job_outputs, project_root=project_root)
    assert warp.get("shared_marker") == 7
    assert fit == "contain"
    assert edge == 2


def test_load_tuned_overlay_params_prefers_job_local_over_shared(tmp_path, monkeypatch):
    """Task 2 — when a job-local tuned JSON exists it wins over the shared
    one so a per-job snapshot pins behavior across later global learns.
    """
    import json as _json

    from project.scripts import paths as paths_mod
    from project.scripts.main import _load_tuned_overlay_params

    monkeypatch.setenv("BRAINFAST_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setattr(paths_mod, "resolve_state_root", lambda _pr: tmp_path / "state")

    project_root = tmp_path / "proj"
    project_root.mkdir()
    job_outputs = tmp_path / "jobs" / "job_b"
    job_outputs.mkdir(parents=True)

    # Shared — should be IGNORED because job-local is present
    shared = tmp_path / "state" / "calibration" / "trainset_tuned_params.json"
    shared.parent.mkdir(parents=True, exist_ok=True)
    shared.write_text(
        _json.dumps({"warpParams": {"src": "shared"}, "fitMode": "contain", "edgeSmoothIter": 5}),
        encoding="utf-8",
    )
    (job_outputs / "trainset_tuned_params.json").write_text(
        _json.dumps({"warpParams": {"src": "local"}, "fitMode": "cover", "edgeSmoothIter": 1}),
        encoding="utf-8",
    )

    warp, fit, edge = _load_tuned_overlay_params(job_outputs, project_root=project_root)
    assert warp.get("src") == "local"
    assert fit == "cover"
    assert edge == 1


def test_extract_channel_preserves_other_channels_in_tmp_dir(tmp_path):
    """Calling _extract_channel_to_tmp for a second channel must not wipe the
    first channel's files — otherwise dual-channel UI overlay breaks.
    """
    import numpy as np
    from tifffile import imwrite

    from project.scripts.main import _extract_channel_to_tmp

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    for i in range(2):
        arr = np.zeros((4, 4), dtype=np.uint16)
        imwrite(str(src_dir / f"z{i:04d}_C0.tif"), arr)
        imwrite(str(src_dir / f"z{i:04d}_C1.tif"), arr + 100)

    tmp_dir = tmp_path / "tmp_channel"
    src_c0 = sorted(src_dir.glob("*_C0.tif"))
    src_c1 = sorted(src_dir.glob("*_C1.tif"))

    _extract_channel_to_tmp(src_c0, tmp_dir, ch_idx=0)
    assert sorted(p.name for p in tmp_dir.glob("ch_0_*.tif")) == [
        "ch_0_0000.tif",
        "ch_0_0001.tif",
    ]

    _extract_channel_to_tmp(src_c1, tmp_dir, ch_idx=1)
    names = sorted(p.name for p in tmp_dir.glob("*.tif"))
    assert "ch_0_0000.tif" in names
    assert "ch_0_0001.tif" in names
    assert "ch_1_0000.tif" in names
    assert "ch_1_0001.tif" in names


def test_extract_channel_replaces_only_same_channel_on_rerun(tmp_path):
    import numpy as np
    from tifffile import imwrite

    from project.scripts.main import _extract_channel_to_tmp

    src_dir = tmp_path / "src"
    src_dir.mkdir()
    for i in range(3):
        arr = np.zeros((4, 4), dtype=np.uint16)
        imwrite(str(src_dir / f"z{i:04d}_C0.tif"), arr)
        imwrite(str(src_dir / f"z{i:04d}_C1.tif"), arr + 100)

    tmp_dir = tmp_path / "tmp_channel"
    _extract_channel_to_tmp(sorted(src_dir.glob("*_C0.tif")), tmp_dir, ch_idx=0)
    _extract_channel_to_tmp(sorted(src_dir.glob("*_C1.tif")), tmp_dir, ch_idx=1)
    _extract_channel_to_tmp(sorted(src_dir.glob("*_C0.tif"))[:2], tmp_dir, ch_idx=0)
    names = sorted(p.name for p in tmp_dir.glob("*.tif"))
    assert sorted(n for n in names if n.startswith("ch_0_")) == [
        "ch_0_0000.tif",
        "ch_0_0001.tif",
    ]
    assert sorted(n for n in names if n.startswith("ch_1_")) == [
        "ch_1_0000.tif",
        "ch_1_0001.tif",
        "ch_1_0002.tif",
    ]


def test_run_config_defaults_anchor_whole_brain_3d_contract():
    _project = Path(__file__).resolve().parents[2]
    template = (_project / "configs" / "run_config.template.json").read_text(
        encoding="utf-8"
    )
    sample = (_project / "configs" / "run_config_35.json").read_text(encoding="utf-8")

    for text in (template, sample):
        assert '"whole_brain_backend": "miki_3d"' in text
        assert '"truth_source": "3d_registered_volume"' in text


def test_check_env_validates_ants_is_optional():
    from project.scripts import check_env

    # ANTs is optional (needed only for whole-brain 3D registration)
    assert "ants" in check_env.OPTIONAL_MODULES
    assert "ants" not in check_env.REQUIRED_MODULES


def test_check_env_resolves_relative_input_dir_from_config_parent(tmp_path):
    from project.scripts import check_env

    project_root = tmp_path / "project"
    project_root.mkdir()
    config_path = project_root / "configs" / "run_config.json"
    config_path.parent.mkdir(parents=True)
    relative_input = Path("data/raw")
    resolved_input = config_path.parent / relative_input
    resolved_input.mkdir(parents=True)

    result = check_env._resolve_input_dir(
        str(relative_input),
        project_root=project_root,
        config_path=config_path,
    )

    assert result == resolved_input


def test_check_env_main_uses_dynamic_structure_source_and_warns_on_nrrd_fallback(
    tmp_path, monkeypatch
):
    from project.scripts import check_env

    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    structure_source = tmp_path / "custom_structure.csv"
    structure_source.write_text("id,name\n1,root\n", encoding="utf-8")

    printed: list[tuple[bool, str, str, str]] = []

    monkeypatch.setattr(
        check_env, "default_structure_source", lambda _project_root: structure_source
    )
    monkeypatch.setattr(
        check_env,
        "atlas_asset_status",
        lambda _project_root: {"annotationReady": False, "annotationNrrdReady": True},
    )
    monkeypatch.setattr(check_env, "load_config", lambda _cfg_path: {})
    monkeypatch.setattr(
        check_env, "validate_runtime_config", lambda _cfg, require_input_dir=False: []
    )
    monkeypatch.setattr(
        check_env,
        "_print_status",
        lambda ok, kind, label, detail="": printed.append((ok, kind, label, detail)),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["check_env.py", "--config", str(config_path)],
    )

    exit_code = check_env.main()

    assert exit_code == 0
    assert any(
        ok and kind == "FAIL" and label == "asset" and detail == str(structure_source)
        for ok, kind, label, detail in printed
    )
    assert any(
        not ok
        and kind == "WARN"
        and label == "asset"
        and "annotation_25.nrrd exists but annotation_25.nii.gz is still missing" in detail
        for ok, kind, label, detail in printed
    )
