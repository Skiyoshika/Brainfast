from __future__ import annotations

import json
from pathlib import Path

from project.frontend.update_checker import (
    check_for_update,
    fetch_latest_release,
    is_newer_version,
    normalize_version_tag,
    version_sort_key,
)


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def read(self) -> bytes:
        return json.dumps(self._payload).encode("utf-8")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def test_normalize_version_tag_strips_leading_v():
    assert normalize_version_tag("v1.2.3") == "1.2.3"
    assert normalize_version_tag("1.2.3") == "1.2.3"


def test_version_sort_key_prefers_release_over_prerelease():
    assert version_sort_key("1.0.0") > version_sort_key("1.0.0-rc1")


def test_is_newer_version_handles_semver_ordering():
    assert is_newer_version("0.4.0", "0.3.0-desktop") is True
    assert is_newer_version("0.3.0", "0.3.0") is False


def test_fetch_latest_release_parses_github_payload(monkeypatch):
    payload = {
        "tag_name": "v0.4.0",
        "name": "Brainfast 0.4.0",
        "published_at": "2026-04-01T00:00:00Z",
        "html_url": "https://github.com/Skiyoshika/Brainfast/releases/tag/v0.4.0",
    }

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: _FakeResponse(payload),
    )

    data = fetch_latest_release(Path("D:/Brainfast/project"))
    assert data["version"] == "0.4.0"
    assert data["tag"] == "v0.4.0"
    assert data["html_url"].endswith("/v0.4.0")


def test_check_for_update_reports_update(monkeypatch):
    payload = {
        "tag_name": "v9.9.9",
        "name": "Brainfast 9.9.9",
        "published_at": "2026-04-01T00:00:00Z",
        "html_url": "https://github.com/Skiyoshika/Brainfast/releases/tag/v9.9.9",
    }

    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: _FakeResponse(payload),
    )

    result = check_for_update(Path("D:/Brainfast/project"))
    assert result["has_update"] is True
    assert result["latest_version"] == "9.9.9"
