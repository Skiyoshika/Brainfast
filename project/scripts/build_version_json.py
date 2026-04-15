#!/usr/bin/env python3
"""build_version_json.py — Write project/version.json from GITHUB_REF_NAME.

Delegates to write_version_metadata.main() so both writers produce the
identical 6-field JSON schema (version, build_date, commit, repository,
releases_api, releases_page).

Usage (called by release.yml):
    python project/scripts/build_version_json.py

Environment variables read:
    GITHUB_REF_NAME          — e.g. "v0.5.1" (set automatically by GitHub Actions)
    GITHUB_SHA               — full commit SHA (set automatically by GitHub Actions)
    BRAINFAST_VERSION_JSON   — optional override for output path
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent.parent
DEFAULT_VERSION_JSON = REPO_ROOT / "project" / "version.json"


def _git_short_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True, cwd=REPO_ROOT
        ).strip()
    except Exception:
        return os.environ.get("GITHUB_SHA", "unknown")[:7]


def main() -> int:
    from project.scripts.write_version_metadata import main as write_version_main

    ref_name = os.environ.get("GITHUB_REF_NAME", "")
    version = ref_name.lstrip("v") if ref_name else "0.0.0-dev"

    commit = _git_short_sha()
    build_date = datetime.now(tz=timezone.utc).strftime("%Y-%m-%d")

    output_path = os.environ.get("BRAINFAST_VERSION_JSON", str(DEFAULT_VERSION_JSON))

    return write_version_main(
        [
            "--version",
            version,
            "--commit",
            commit,
            "--build-date",
            build_date,
            "--output",
            output_path,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
