"""Thin SQLite wrapper for Brainfast project/sample persistence.

The database lives at:
  - Persistent mode: ~/.brainfast/brainfast.db
  - Portable mode:   project/outputs/brainfast.db  (if ~/.brainfast is not writable)

Schema (created on first use)::

    projects (id TEXT PRIMARY KEY, name TEXT, description TEXT, created_at TEXT)
    samples  (id TEXT PRIMARY KEY, project_id TEXT, name TEXT,
              config_path TEXT, input_dir TEXT, outputs_dir TEXT,
              status TEXT, job_id TEXT, error TEXT,
              created_at TEXT, finished_at TEXT)
    run_events (id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id TEXT, event TEXT, timestamp TEXT, data TEXT)

Usage::

    from project.frontend.services.database import get_conn, init_db
    init_db()
    with get_conn() as conn:
        conn.execute("INSERT INTO projects VALUES (?,?,?,?)", ...)
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

_LOCK = threading.Lock()
_DB_PATH: Path | None = None


def _resolve_db_path() -> Path:
    """Return the path to the SQLite database, creating directories as needed."""
    candidate = Path.home() / ".brainfast" / "brainfast.db"
    try:
        candidate.parent.mkdir(parents=True, exist_ok=True)
        # Quick write-access check
        candidate.parent.joinpath(".write_test").touch()
        candidate.parent.joinpath(".write_test").unlink()
        return candidate
    except OSError:
        # Fallback to portable location alongside project outputs
        fallback = Path(__file__).resolve().parents[2] / "outputs" / "brainfast.db"
        fallback.parent.mkdir(parents=True, exist_ok=True)
        return fallback


def _db_path() -> Path:
    global _DB_PATH
    if _DB_PATH is None:
        with _LOCK:
            if _DB_PATH is None:
                _DB_PATH = _resolve_db_path()
    return _DB_PATH


_SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    description TEXT DEFAULT '',
    created_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS samples (
    id          TEXT PRIMARY KEY,
    project_id  TEXT NOT NULL,
    name        TEXT NOT NULL,
    config_path TEXT DEFAULT '',
    input_dir   TEXT DEFAULT '',
    outputs_dir TEXT DEFAULT '',
    status      TEXT DEFAULT 'pending',
    job_id      TEXT DEFAULT '',
    error       TEXT DEFAULT '',
    created_at  TEXT NOT NULL,
    finished_at TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS run_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    sample_id   TEXT NOT NULL,
    event       TEXT NOT NULL,
    timestamp   TEXT NOT NULL,
    data        TEXT DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_samples_project ON samples(project_id);
CREATE INDEX IF NOT EXISTS idx_events_sample   ON run_events(sample_id);
"""


def init_db() -> None:
    """Create tables if they don't exist. Safe to call multiple times."""
    with _LOCK:
        conn = sqlite3.connect(str(_db_path()))
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()


@contextmanager
def get_conn():
    """Context manager yielding a sqlite3.Connection with row_factory set.

    Commits on success, rolls back on exception, closes always::

        with get_conn() as conn:
            conn.execute("SELECT ...", ())
    """
    conn = sqlite3.connect(str(_db_path()))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")  # Allow concurrent reads
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def now_iso() -> str:
    """Return current UTC time as ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")
