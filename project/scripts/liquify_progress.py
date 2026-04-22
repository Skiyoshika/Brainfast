"""Per-job progress marker for long-running liquify operations.

The frontend polls ``GET /api/liquify-3d/progress`` while an ``apply`` or
``finalize`` call is in flight. Backend worker threads call
``write_liquify_progress`` at milestone points so the poller surfaces
current stage + percent instead of a frozen spinner.

Deliberately **separate** from ``pipeline_progress.json`` so running liquify
does not overwrite the main pipeline's progress display. Same atomic-write
pattern (tempfile + ``os.replace``) to avoid torn reads.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path

_FILENAME = "liquify_progress.json"


def _progress_path(job_dir: Path) -> Path:
    return Path(job_dir) / _FILENAME


def write_liquify_progress(
    job_dir: Path | str,
    *,
    stage: str,
    stage_index: int,
    stage_count: int,
    percent: int,
    message: str,
) -> Path:
    """Write a progress snapshot atomically.

    Raises
    ------
    ValueError
        If ``percent`` is outside ``[0, 100]``.
    """
    if not 0 <= int(percent) <= 100:
        raise ValueError(f"percent must be in [0, 100], got {percent}")

    job_dir = Path(job_dir)
    job_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "stage": str(stage),
        "stage_index": int(stage_index),
        "stage_count": int(stage_count),
        "percent": int(percent),
        "message": str(message),
        "ts": time.time(),
    }
    progress_path = _progress_path(job_dir)
    fd, tmp_name = tempfile.mkstemp(
        prefix="liquify_progress.",
        suffix=".tmp",
        dir=str(job_dir),
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


def read_liquify_progress(job_dir: Path | str) -> dict:
    progress_path = _progress_path(Path(job_dir))
    if not progress_path.exists():
        return {}
    try:
        return json.loads(progress_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        # Treat a momentarily-malformed file as "no progress" rather than
        # bubbling up — the next writer pass will fix it.
        return {}


def clear_liquify_progress(job_dir: Path | str) -> None:
    p = _progress_path(Path(job_dir))
    if p.exists():
        p.unlink()
