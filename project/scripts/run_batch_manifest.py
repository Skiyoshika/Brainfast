from __future__ import annotations

import argparse
import csv
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BatchJob:
    sample_id: str
    input_dir: Path
    config: Path
    output_name: str
    output_dir: Path | None = None


def _resolve_manifest_path(raw: str, manifest_path: Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def load_manifest(manifest_path: Path, default_config: Path) -> list[BatchJob]:
    jobs: list[BatchJob] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = {"sample_id", "input_dir"} - set(reader.fieldnames or [])
        if missing:
            raise ValueError("Manifest is missing required columns: " + ", ".join(sorted(missing)))

        for row_num, row in enumerate(reader, start=2):
            sample_id = str(row.get("sample_id", "")).strip()
            input_dir_raw = str(row.get("input_dir", "")).strip()
            if not sample_id or not input_dir_raw:
                raise ValueError(
                    f"Manifest row {row_num} must provide both sample_id and input_dir"
                )

            input_dir = _resolve_manifest_path(input_dir_raw, manifest_path)
            config_raw = str(row.get("config", "")).strip()
            config = (
                _resolve_manifest_path(config_raw, manifest_path) if config_raw else default_config
            )

            output_name = str(row.get("output_name", "")).strip() or sample_id
            output_dir_raw = str(row.get("output_dir", "")).strip()
            output_dir = (
                _resolve_manifest_path(output_dir_raw, manifest_path) if output_dir_raw else None
            )

            jobs.append(
                BatchJob(
                    sample_id=sample_id,
                    input_dir=input_dir,
                    config=config,
                    output_name=output_name,
                    output_dir=output_dir,
                )
            )
    return jobs


def build_run_command(job: BatchJob, project_root: Path, python_executable: str) -> list[str]:
    cmd = [
        python_executable,
        str(project_root / "scripts" / "main.py"),
        "--config",
        str(job.config),
        "--run-real-input",
        str(job.input_dir),
        "--output-name",
        job.output_name,
    ]
    if job.output_dir is not None:
        cmd.extend(["--output-dir", str(job.output_dir)])
    return cmd


def _validate_jobs(jobs: list[BatchJob]) -> None:
    for job in jobs:
        if not job.input_dir.exists():
            raise FileNotFoundError(
                f"Input directory not found for {job.sample_id}: {job.input_dir}"
            )
        if not job.config.exists():
            raise FileNotFoundError(f"Config not found for {job.sample_id}: {job.config}")


def run_jobs(
    jobs: list[BatchJob],
    project_root: Path,
    python_executable: str,
    keep_going: bool = False,
    dry_run: bool = False,
) -> int:
    _validate_jobs(jobs)

    failures = 0
    for idx, job in enumerate(jobs, start=1):
        cmd = build_run_command(job, project_root, python_executable)
        print(f"[{idx}/{len(jobs)}] {job.sample_id}")
        print("  input :", job.input_dir)
        print("  config:", job.config)
        print("  cmd   :", " ".join(f'"{part}"' if " " in part else part for part in cmd))
        if dry_run:
            continue

        started = time.time()
        result = subprocess.run(cmd, cwd=str(project_root))
        elapsed = time.time() - started
        print(f"  exit  : {result.returncode} ({elapsed:.1f}s)")
        if result.returncode != 0:
            failures += 1
            if not keep_going:
                return result.returncode

    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run Brainfast across a CSV manifest of sample_id/input_dir rows."
    )
    parser.add_argument("--manifest", required=True, help="CSV manifest file")
    parser.add_argument(
        "--default-config",
        required=True,
        help="Fallback config path used when a manifest row leaves config blank",
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for invoking scripts/main.py",
    )
    parser.add_argument(
        "--keep-going",
        action="store_true",
        help="Continue running remaining jobs even if one job fails",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved commands without executing them",
    )
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    project_root = script_path.parents[1]
    manifest_path = Path(args.manifest).expanduser().resolve()
    default_config = Path(args.default_config).expanduser().resolve()

    jobs = load_manifest(manifest_path, default_config)
    if not jobs:
        raise ValueError("Manifest did not contain any jobs")

    return run_jobs(
        jobs,
        project_root=project_root,
        python_executable=args.python,
        keep_going=args.keep_going,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    raise SystemExit(main())
