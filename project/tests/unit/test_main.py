from __future__ import annotations

import sys
from pathlib import Path

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


def test_run_config_defaults_anchor_whole_brain_3d_contract():
    template = Path(r"D:\Brainfast\project\configs\run_config.template.json").read_text(
        encoding="utf-8"
    )
    sample = Path(r"D:\Brainfast\project\configs\run_config_35.json").read_text(encoding="utf-8")

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
