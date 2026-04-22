"""End-to-end smoke test: boot the real production server (Waitress) in a subprocess
and hit a handful of read-only endpoints to prove the wiring works end-to-end.

This complements Flask-test-client unit tests by verifying that:
1. `python -m project.frontend.server` actually starts and listens
2. Waitress serves the same routes Flask's test client does
3. The atlas-status banner endpoint is reachable from a real HTTP client
4. The static index.html is served
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = PROJECT_ROOT.parent

pytestmark = pytest.mark.integration


def _find_free_port() -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _wait_for_server(url: str, timeout: float = 20.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = requests.get(url, timeout=2.0)
            if r.status_code < 500:
                return True
        except requests.RequestException:
            pass
        time.sleep(0.5)
    return False


@pytest.fixture(scope="module")
def running_server():
    port = _find_free_port()
    env = dict(os.environ)
    env["BRAINFAST_PORT"] = str(port)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    # Force Waitress path (production) — no BRAINFAST_DEV
    env.pop("BRAINFAST_DEV", None)

    proc = subprocess.Popen(
        [sys.executable, "-m", "project.frontend.server"],
        env=env,
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    base = f"http://127.0.0.1:{port}"
    try:
        if not _wait_for_server(f"{base}/api/info"):
            out, err = proc.communicate(timeout=5)
            raise RuntimeError(
                f"server did not start:\nSTDOUT:\n{out.decode(errors='replace')}\n"
                f"STDERR:\n{err.decode(errors='replace')}"
            )
        yield base
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)


def test_waitress_serves_api_info(running_server: str) -> None:
    r = requests.get(f"{running_server}/api/info", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert "version" in data


def test_waitress_serves_atlas_status(running_server: str) -> None:
    r = requests.get(f"{running_server}/api/atlas/status", timeout=5)
    assert r.status_code == 200
    data = r.json()
    assert data["ok"] is True
    for key in ("annotationReady", "structureReady", "allRequiredReady", "downloadHint"):
        assert key in data


def test_waitress_serves_static_index(running_server: str) -> None:
    r = requests.get(f"{running_server}/", timeout=5)
    assert r.status_code == 200
    body = r.text
    assert "Brainfast" in body
    assert 'id="atlasMissingBanner"' in body  # Phase 2.2 banner markup ships in index.html


def test_waitress_reports_server_header(running_server: str) -> None:
    """Confirms Waitress is the actual WSGI server — Flask dev sends 'Werkzeug'."""
    r = requests.get(f"{running_server}/api/info", timeout=5)
    server_hdr = r.headers.get("Server", "")
    assert "waitress" in server_hdr.lower(), f"expected waitress, got: {server_hdr!r}"
