from __future__ import annotations

from pathlib import Path

from project.scripts.run_batch_manifest import BatchJob, build_run_command, load_manifest


def test_load_manifest_uses_default_config_and_sample_id_as_output_name(tmp_path):
    manifest = tmp_path / "batch.csv"
    manifest.write_text(
        "sample_id,input_dir,config,output_name,output_dir\nbrain_01,data/brain_01,,,\n",
        encoding="utf-8",
    )
    default_config = tmp_path / "run_config.json"
    default_config.write_text("{}", encoding="utf-8")

    jobs = load_manifest(manifest, default_config)

    assert jobs == [
        BatchJob(
            sample_id="brain_01",
            input_dir=(tmp_path / "data" / "brain_01").resolve(),
            config=default_config,
            output_name="brain_01",
            output_dir=None,
        )
    ]


def test_load_manifest_resolves_row_specific_paths_relative_to_manifest(tmp_path):
    manifest = tmp_path / "batch.csv"
    manifest.write_text(
        "sample_id,input_dir,config,output_name,output_dir\n"
        "brain_02,data/brain_02,configs/run_02.json,run_02,outputs/run_02\n",
        encoding="utf-8",
    )
    default_config = tmp_path / "unused.json"
    default_config.write_text("{}", encoding="utf-8")

    jobs = load_manifest(manifest, default_config)

    assert jobs[0].input_dir == (tmp_path / "data" / "brain_02").resolve()
    assert jobs[0].config == (tmp_path / "configs" / "run_02.json").resolve()
    assert jobs[0].output_name == "run_02"
    assert jobs[0].output_dir == (tmp_path / "outputs" / "run_02").resolve()


def test_build_run_command_includes_explicit_output_dir():
    project_root = Path(r"D:\Brainfast\project")
    job = BatchJob(
        sample_id="brain_03",
        input_dir=Path(r"D:\data\brain_03"),
        config=Path(r"D:\Brainfast\project\configs\run_config_35.json"),
        output_name="brain_03_run",
        output_dir=Path(r"D:\Brainfast\project\outputs\brain_03_run"),
    )

    cmd = build_run_command(job, project_root, "python")

    assert cmd == [
        "python",
        str(project_root / "scripts" / "main.py"),
        "--config",
        str(job.config),
        "--run-real-input",
        str(job.input_dir),
        "--output-name",
        "brain_03_run",
        "--output-dir",
        str(job.output_dir),
    ]
