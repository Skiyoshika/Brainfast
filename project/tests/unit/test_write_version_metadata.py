from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.write_version_metadata import normalize_version  # noqa: E402


def test_normalize_version_removes_leading_v():
    assert normalize_version("v1.2.3") == "1.2.3"
    assert normalize_version("1.2.3") == "1.2.3"


def test_version_json_fixture_shape_matches_writer_contract():
    repo_root = Path(__file__).resolve().parents[3]
    payload = json.loads((repo_root / "project" / "version.json").read_text(encoding="utf-8"))
    assert {
        "version",
        "build_date",
        "commit",
        "repository",
        "releases_api",
        "releases_page",
    } <= set(payload)
