from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path


def _progress_path(outputs_dir: Path) -> Path:
    return Path(outputs_dir) / "pipeline_progress.json"


def write_stage_progress(
    outputs_dir: Path,
    stage_name: str,
    stage_index: int,
    stage_count: int,
    percent: int,
    message: str,
    artifacts: dict | None = None,
) -> Path:
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stageName": str(stage_name),
        "stageIndex": int(stage_index),
        "stageCount": int(stage_count),
        "percent": int(percent),
        "message": str(message),
        "artifacts": dict(artifacts or {}),
    }
    progress_path = _progress_path(outputs_dir)
    fd, tmp_name = tempfile.mkstemp(
        prefix="pipeline_progress.",
        suffix=".tmp",
        dir=str(outputs_dir),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2))
        last_error: OSError | None = None
        for attempt in range(8):
            try:
                tmp_path.replace(progress_path)
                last_error = None
                break
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.01 * (attempt + 1))
        if last_error is not None:
            raise last_error
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return progress_path


def read_stage_progress(outputs_dir: Path) -> dict:
    progress_path = _progress_path(Path(outputs_dir))
    if not progress_path.exists():
        return {}
    return json.loads(progress_path.read_text(encoding="utf-8"))
