from __future__ import annotations

import json
from pathlib import Path

import pytest

from project.frontend.app_metadata import (
    DEFAULT_MAX_CONTENT_LENGTH,
    read_version_info,
    resolve_max_content_length,
)


def test_version_info_matches_version_json():
    repo_root = Path(__file__).resolve().parents[3]
    version_path = repo_root / "project" / "version.json"
    payload = json.loads(version_path.read_text(encoding="utf-8"))
    info = read_version_info(repo_root / "project")
    assert info["version"] == str(payload["version"])
    assert info["build_date"] == str(payload["build_date"])
    assert info["commit"] == str(payload["commit"])
    assert info["repository"] == str(payload["repository"])
    assert info["releases_api"] == str(payload["releases_api"])
    assert info["releases_page"] == str(payload["releases_page"])


def test_repository_metadata_files_exist():
    repo_root = Path(__file__).resolve().parents[3]
    assert (repo_root / "REPRODUCE.md").exists()
    assert (repo_root / "CITATION.cff").exists()
    assert (repo_root / "project" / "version.json").exists()


def test_resolve_max_content_length_default(monkeypatch):
    monkeypatch.delenv("BRAINFAST_MAX_CONTENT_LENGTH", raising=False)
    assert resolve_max_content_length() == DEFAULT_MAX_CONTENT_LENGTH


def test_resolve_max_content_length_invalid_value_falls_back(monkeypatch):
    monkeypatch.setenv("BRAINFAST_MAX_CONTENT_LENGTH", "invalid")
    assert resolve_max_content_length() == DEFAULT_MAX_CONTENT_LENGTH


def test_server_file_configures_request_size_limit():
    repo_root = Path(__file__).resolve().parents[3]
    server_text = (repo_root / "project" / "frontend" / "server.py").read_text(encoding="utf-8")
    assert 'app.config["MAX_CONTENT_LENGTH"] = resolve_max_content_length()' in server_text
    assert "RequestEntityTooLarge" in server_text


def test_server_bootstraps_sys_path_before_project_imports():
    repo_root = Path(__file__).resolve().parents[3]
    server_text = (repo_root / "project" / "frontend" / "server.py").read_text(encoding="utf-8")
    sys_path_idx = server_text.index("_repo_root = str(PROJECT_ROOT.parent)")
    import_idx = server_text.index(
        "from project.frontend.app_metadata import resolve_max_content_length"
    )
    assert sys_path_idx < import_idx


def test_read_version_info_falls_back_to_internal_bundle_path(tmp_path):
    internal = tmp_path / "_internal"
    internal.mkdir()
    payload = {
        "version": "0.4.0",
        "build_date": "2026-04-01",
        "commit": "abc123",
        "repository": "https://example.com/repo",
        "releases_api": "https://example.com/api/releases/latest",
        "releases_page": "https://example.com/releases/latest",
    }
    (internal / "version.json").write_text(json.dumps(payload), encoding="utf-8")

    assert read_version_info(tmp_path) == payload


def test_ensure_port_available_raises_runtime_error_when_bind_fails(monkeypatch):
    from project.frontend import server

    class FakeSocket:
        def bind(self, _addr):
            raise OSError(10048, "Only one usage of each socket address")

        def close(self):
            return None

    monkeypatch.setattr(server.socket, "socket", lambda *args, **kwargs: FakeSocket())

    with pytest.raises(RuntimeError, match="8787"):
        server.ensure_port_available(8787)


def test_server_main_checks_port_before_running_app(monkeypatch):
    """Order invariant: port-guard runs before the WSGI handoff.

    Uses BRAINFAST_DEV=1 to force the Flask dev path (app.run). The default
    production path is Waitress — covered separately in test_server.py. The
    point of THIS test is just: port check happens before we hand off.
    """
    from project.frontend import server

    calls = []

    monkeypatch.setattr(
        server,
        "ensure_port_available",
        lambda port, host="127.0.0.1": calls.append(("check", host, port)),
    )
    monkeypatch.setattr(
        server.app,
        "run",
        lambda **kwargs: calls.append(("run", kwargs["host"], kwargs["port"])),
    )
    monkeypatch.setenv("BRAINFAST_PORT", "9001")
    monkeypatch.setenv("BRAINFAST_DEV", "1")

    server.main()

    assert calls == [
        ("check", "127.0.0.1", 9001),
        ("run", "127.0.0.1", 9001),
    ]
