"""Project and sample CRUD for Brainfast multi-sample workflows.

All persistence goes through ``services.database`` (SQLite).  This module
provides high-level helpers used by ``blueprints/api_projects.py``.

Sample status lifecycle::

    pending  →  queued  →  running  →  done
                                    ↘  error
"""

from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any

from project.frontend.services.database import get_conn, init_db, now_iso


def _ensure_db() -> None:
    init_db()


# ── Projects ─────────────────────────────────────────────────────────────────


def list_projects() -> list[dict[str, Any]]:
    _ensure_db()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, name, description, created_at FROM projects ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_project(project_id: str) -> dict[str, Any] | None:
    _ensure_db()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT id, name, description, created_at FROM projects WHERE id = ?",
            (project_id,),
        ).fetchone()
    return dict(row) if row else None


def create_project(name: str, description: str = "") -> dict[str, Any]:
    _ensure_db()
    pid = str(uuid.uuid4())
    ts = now_iso()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO projects (id, name, description, created_at) VALUES (?,?,?,?)",
            (pid, str(name), str(description), ts),
        )
    return {"id": pid, "name": name, "description": description, "created_at": ts}


def delete_project(project_id: str) -> bool:
    """Delete a project and all its samples. Returns True if a row was deleted."""
    _ensure_db()
    with get_conn() as conn:
        conn.execute("DELETE FROM run_events WHERE sample_id IN "
                     "(SELECT id FROM samples WHERE project_id = ?)", (project_id,))
        conn.execute("DELETE FROM samples WHERE project_id = ?", (project_id,))
        cur = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
    return cur.rowcount > 0


# ── Samples ───────────────────────────────────────────────────────────────────


def list_samples(project_id: str) -> list[dict[str, Any]]:
    _ensure_db()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM samples WHERE project_id = ? ORDER BY created_at ASC",
            (project_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_sample(sample_id: str) -> dict[str, Any] | None:
    _ensure_db()
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM samples WHERE id = ?", (sample_id,)).fetchone()
    return dict(row) if row else None


def add_sample(
    project_id: str,
    *,
    name: str,
    config_path: str = "",
    input_dir: str = "",
    outputs_dir: str = "",
) -> dict[str, Any]:
    _ensure_db()
    sid = str(uuid.uuid4())
    ts = now_iso()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO samples "
            "(id, project_id, name, config_path, input_dir, outputs_dir, "
            " status, job_id, error, created_at, finished_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (sid, project_id, name, config_path, input_dir, outputs_dir,
             "pending", "", "", ts, ""),
        )
    return get_sample(sid)  # type: ignore[return-value]


def update_sample_status(
    sample_id: str,
    status: str,
    *,
    job_id: str = "",
    error: str = "",
    outputs_dir: str = "",
) -> bool:
    _ensure_db()
    finished = now_iso() if status in ("done", "error") else ""
    with get_conn() as conn:
        if outputs_dir:
            sql = "UPDATE samples SET status=?, job_id=?, error=?, finished_at=?, outputs_dir=? WHERE id=?"
            params = (status, job_id, error, finished, outputs_dir, sample_id)
        else:
            sql = "UPDATE samples SET status=?, job_id=?, error=?, finished_at=? WHERE id=?"
            params = (status, job_id, error, finished, sample_id)
        cur = conn.execute(sql, params)
    return cur.rowcount > 0


def delete_sample(sample_id: str) -> bool:
    _ensure_db()
    with get_conn() as conn:
        conn.execute("DELETE FROM run_events WHERE sample_id = ?", (sample_id,))
        cur = conn.execute("DELETE FROM samples WHERE id = ?", (sample_id,))
    return cur.rowcount > 0


# ── Run events ────────────────────────────────────────────────────────────────


def append_run_event(sample_id: str, event: str, data: str = "") -> None:
    _ensure_db()
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO run_events (sample_id, event, timestamp, data) VALUES (?,?,?,?)",
            (sample_id, event, now_iso(), data),
        )


def get_run_events(sample_id: str, limit: int = 200) -> list[dict[str, Any]]:
    _ensure_db()
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT id, event, timestamp, data FROM run_events "
            "WHERE sample_id = ? ORDER BY id DESC LIMIT ?",
            (sample_id, limit),
        ).fetchall()
    return [dict(r) for r in reversed(rows)]
