"""Tests for the server entry point, including WSGI runner selection.

In production, `server.main()` must serve via Waitress (a real WSGI server),
not Flask's dev server. The dev server is only used when BRAINFAST_DEV=1.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


def _import_server_main():
    from project.frontend import server

    return server


def test_main_uses_waitress_by_default(monkeypatch):
    monkeypatch.delenv("BRAINFAST_DEV", raising=False)
    monkeypatch.setenv("BRAINFAST_PORT", "18787")

    server = _import_server_main()

    with (
        patch.object(server, "ensure_port_available", lambda *a, **k: None),
        patch("waitress.serve") as waitress_serve,
        patch.object(server.app, "run") as flask_run,
    ):
        server.main()

    waitress_serve.assert_called_once()
    flask_run.assert_not_called()

    kwargs = waitress_serve.call_args.kwargs
    assert kwargs.get("host") == "127.0.0.1"
    assert kwargs.get("port") == 18787


def test_main_uses_flask_dev_server_when_BRAINFAST_DEV_set(monkeypatch):
    monkeypatch.setenv("BRAINFAST_DEV", "1")
    monkeypatch.setenv("BRAINFAST_PORT", "18788")

    server = _import_server_main()

    with (
        patch.object(server, "ensure_port_available", lambda *a, **k: None),
        patch.object(server.app, "run") as flask_run,
        patch("waitress.serve") as waitress_serve,
    ):
        server.main()

    flask_run.assert_called_once()
    waitress_serve.assert_not_called()


def test_main_falls_back_to_flask_when_waitress_missing(monkeypatch):
    """Graceful degradation: if waitress isn't installed, use app.run with a warning.

    Keeps existing installs working during the rollout before `pip install -e .`
    is rerun. Setting sys.modules['waitress'] = None makes `import waitress`
    raise ImportError without patching builtins.__import__ — the latter
    interacts badly with coverage's sys.settrace hook on some CI runners.
    """
    import sys as _sys

    monkeypatch.delenv("BRAINFAST_DEV", raising=False)
    monkeypatch.setenv("BRAINFAST_PORT", "18789")

    server = _import_server_main()

    # Force `import waitress` inside server.main() to raise ImportError
    monkeypatch.setitem(_sys.modules, "waitress", None)

    with (
        patch.object(server, "ensure_port_available", lambda *a, **k: None),
        patch.object(server.app, "run") as flask_run,
    ):
        server.main()

    flask_run.assert_called_once()


def test_ensure_port_available_raises_on_taken_port():
    """Sanity check: the port-in-use guard still raises RuntimeError so a user
    sees a clear message instead of a cryptic OSError."""
    import socket

    server = _import_server_main()

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    try:
        with pytest.raises(RuntimeError, match="already in use"):
            server.ensure_port_available(port)
    finally:
        s.close()
