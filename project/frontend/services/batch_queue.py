"""Batch processing queue for Brainfast multi-sample workflows.

A single background worker thread pops items from a FIFO queue and runs
them one at a time using the existing ``server_context._runner`` function.
Sample status is persisted to the SQLite database via ``project_manager``.

Usage::

    from project.frontend.services.batch_queue import enqueue_sample, get_queue_status

    job_id = enqueue_sample(
        sample_id="uuid-...",
        config_path="/path/to/config.json",
        input_dir="/path/to/slices",
        channels=["red"],
    )
    status = get_queue_status()
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime, timezone
from typing import Any

from project.frontend.services import project_manager as pm

# ── Queue state ───────────────────────────────────────────────────────────────

_queue_lock = threading.Lock()
_queue: list[dict[str, Any]] = []      # pending items (FIFO)
_active_item: dict[str, Any] | None = None   # currently running item
_worker_started = False


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ── Public API ────────────────────────────────────────────────────────────────


def enqueue_sample(
    sample_id: str,
    *,
    config_path: str,
    input_dir: str,
    channels: list[str],
    run_params: dict | None = None,
) -> str:
    """Add a sample to the batch queue.  Returns a new job_id for this run."""
    job_id = f"batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"
    item = {
        "sample_id": sample_id,
        "job_id": job_id,
        "config_path": config_path,
        "input_dir": input_dir,
        "channels": channels,
        "run_params": run_params or {},
        "queued_at": _now_iso(),
    }
    with _queue_lock:
        _queue.append(item)
    pm.update_sample_status(sample_id, "queued", job_id=job_id)
    pm.append_run_event(sample_id, "queued", f"job_id={job_id}")
    _ensure_worker()
    return job_id


def cancel_queued(sample_id: str) -> bool:
    """Remove a PENDING item from the queue.  Returns True if removed."""
    with _queue_lock:
        before = len(_queue)
        _queue[:] = [item for item in _queue if item["sample_id"] != sample_id]
        removed = len(_queue) < before
    if removed:
        pm.update_sample_status(sample_id, "pending")
        pm.append_run_event(sample_id, "cancelled_from_queue")
    return removed


def get_queue_status() -> dict[str, Any]:
    """Return current queue state as a serialisable dict."""
    with _queue_lock:
        pending = list(_queue)
        active = _active_item

    items = []
    if active:
        items.append({**active, "status": "running"})
    for item in pending:
        items.append({**item, "status": "queued"})

    return {
        "active_job_id": active["job_id"] if active else None,
        "queue_length": len(pending),
        "items": items,
    }


# ── Worker thread ─────────────────────────────────────────────────────────────


def _ensure_worker() -> None:
    global _worker_started
    with _queue_lock:
        if _worker_started:
            return
        _worker_started = True
    t = threading.Thread(target=_worker_loop, daemon=True, name="brainfast-batch-worker")
    t.start()


def _worker_loop() -> None:
    global _active_item
    while True:
        with _queue_lock:
            if not _queue:
                # No more work; reset flag so a new thread can start next time
                global _worker_started  # noqa: PLW0603
                _worker_started = False
                return
            item = _queue.pop(0)
            _active_item = item

        sample_id = item["sample_id"]
        job_id = item["job_id"]
        pm.update_sample_status(sample_id, "running", job_id=job_id)
        pm.append_run_event(sample_id, "started", f"job_id={job_id}")

        try:
            # Import here to avoid circular imports at module load time
            import project.frontend.server_context as ctx

            t = threading.Thread(
                target=ctx._runner,
                args=(item["config_path"], item["input_dir"], item["channels"]),
                kwargs={"run_params": item.get("run_params"), "job_id": job_id},
                daemon=True,
                name=f"runner-{job_id}",
            )
            t.start()
            t.join()  # Block until this sample finishes

            job_state = ctx.get_job_state(job_id)
            if job_state.get("error"):
                pm.update_sample_status(
                    sample_id, "error",
                    job_id=job_id,
                    error=str(job_state["error"]),
                    outputs_dir=str(job_state.get("outputs_dir", "")),
                )
                pm.append_run_event(sample_id, "error", str(job_state["error"]))
            else:
                pm.update_sample_status(
                    sample_id, "done",
                    job_id=job_id,
                    outputs_dir=str(job_state.get("outputs_dir", "")),
                )
                pm.append_run_event(sample_id, "done")

        except Exception as exc:
            pm.update_sample_status(sample_id, "error", job_id=job_id, error=str(exc))
            pm.append_run_event(sample_id, "error", str(exc))
        finally:
            with _queue_lock:
                _active_item = None
