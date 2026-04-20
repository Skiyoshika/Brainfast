from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from project.scripts import pipeline_progress
from project.scripts.pipeline_progress import read_stage_progress, write_stage_progress


def test_write_and_read_stage_progress_round_trip(tmp_path):
    write_stage_progress(
        outputs_dir=tmp_path,
        stage_name="ANTS Registration",
        stage_index=3,
        stage_count=6,
        percent=42,
        message="Running SyN stage",
        artifacts={"metrics_csv": "registration_metrics.csv"},
    )

    progress = read_stage_progress(tmp_path)

    assert progress["stageName"] == "ANTS Registration"
    assert progress["stageIndex"] == 3
    assert progress["percent"] == 42
    assert progress["artifacts"]["metrics_csv"] == "registration_metrics.csv"


def test_write_stage_progress_is_safe_for_multiple_concurrent_writers(tmp_path, monkeypatch):
    writer_count = 6
    barrier = threading.Barrier(writer_count)
    path_type = type(tmp_path)
    original_replace = path_type.replace
    temp_sources: set[str] = set()
    first_replace_seen: set[str] = set()
    seen_lock = threading.Lock()

    def tracked_replace(self, target, *args, **kwargs):
        if self.suffix == ".tmp" and self.name.startswith("pipeline_progress."):
            temp_path = str(self)
            temp_sources.add(temp_path)
            should_wait = False
            with seen_lock:
                if temp_path not in first_replace_seen:
                    first_replace_seen.add(temp_path)
                    should_wait = True
            if should_wait:
                barrier.wait(timeout=5)
        return original_replace(self, target, *args, **kwargs)

    monkeypatch.setattr(path_type, "replace", tracked_replace)

    def _writer(idx: int) -> None:
        write_stage_progress(
            outputs_dir=tmp_path,
            stage_name=f"Stage {idx}",
            stage_index=idx,
            stage_count=writer_count,
            percent=idx * 10,
            message=f"writer {idx}",
            artifacts={"writer": idx},
        )

    with ThreadPoolExecutor(max_workers=writer_count) as executor:
        futures = [executor.submit(_writer, idx) for idx in range(writer_count)]
        for future in futures:
            future.result()

    progress = read_stage_progress(tmp_path)

    # New ETA-supporting fields are part of every snapshot now.
    assert set(progress) == {
        "stageName",
        "stageIndex",
        "stageCount",
        "percent",
        "message",
        "artifacts",
        "ts",
        "stageStartedTs",
        "runStartedTs",
    }
    assert len(temp_sources) == writer_count
    assert isinstance(progress["artifacts"], dict)
    # Timestamps must be monotonically valid floats (within last 60 sec)
    import time as _time

    now = _time.time()
    assert isinstance(progress["ts"], (int, float))
    assert isinstance(progress["stageStartedTs"], (int, float))
    assert isinstance(progress["runStartedTs"], (int, float))
    assert now - progress["ts"] < 60
