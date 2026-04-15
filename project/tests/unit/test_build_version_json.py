from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from project.scripts.build_version_json import main  # noqa: E402

EXPECTED_KEYS = {"version", "build_date", "commit", "repository", "releases_api", "releases_page"}


def test_build_version_json_produces_6_field_schema(tmp_path):
    """build_version_json.main() should produce the full 6-field version JSON."""
    out = tmp_path / "version.json"
    env_patch = {
        "GITHUB_REF_NAME": "v1.0.0",
        "GITHUB_SHA": "abcdef1234567890",
        "BRAINFAST_VERSION_JSON": str(out),
    }
    with mock.patch.dict("os.environ", env_patch):
        rc = main()

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert set(payload.keys()) == EXPECTED_KEYS
    assert payload["version"] == "1.0.0"
    assert payload["commit"].startswith("abcdef1") or len(payload["commit"]) == 7


def test_build_version_json_default_version_when_no_ref(tmp_path):
    """Without GITHUB_REF_NAME the version should fall back to 0.0.0-dev."""
    out = tmp_path / "version.json"
    env_patch = {
        "GITHUB_REF_NAME": "",
        "GITHUB_SHA": "deadbeef12345678",
        "BRAINFAST_VERSION_JSON": str(out),
    }
    with mock.patch.dict("os.environ", env_patch):
        rc = main()

    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["version"] == "0.0.0-dev"
    assert set(payload.keys()) == EXPECTED_KEYS
