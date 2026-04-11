from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_MAX_CONTENT_LENGTH = 128 * 1024 * 1024
DEFAULT_REPOSITORY_URL = "https://github.com/Skiyoshika/Brainfast"
DEFAULT_RELEASES_API_URL = "https://api.github.com/repos/Skiyoshika/Brainfast/releases/latest"
DEFAULT_RELEASES_PAGE_URL = f"{DEFAULT_REPOSITORY_URL}/releases/latest"


def resolve_max_content_length() -> int:
    raw = os.environ.get("BRAINFAST_MAX_CONTENT_LENGTH", "").strip()
    if not raw:
        return DEFAULT_MAX_CONTENT_LENGTH
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_CONTENT_LENGTH
    return value if value > 0 else DEFAULT_MAX_CONTENT_LENGTH


def read_version_info(project_root: Path) -> dict[str, str]:
    fallback = {
        "version": "0.3.0-desktop",
        "build_date": "",
        "commit": "",
        "repository": DEFAULT_REPOSITORY_URL,
        "releases_api": DEFAULT_RELEASES_API_URL,
        "releases_page": DEFAULT_RELEASES_PAGE_URL,
    }
    data = None
    for version_path in _version_candidates(project_root):
        if not version_path.exists():
            continue
        try:
            data = json.loads(version_path.read_text(encoding="utf-8"))
            break
        except Exception:
            continue
    if data is None:
        return fallback
    return {
        "version": str(data.get("version", fallback["version"])),
        "build_date": str(data.get("build_date", "")),
        "commit": str(data.get("commit", "")),
        "repository": str(data.get("repository", fallback["repository"])),
        "releases_api": str(data.get("releases_api", fallback["releases_api"])),
        "releases_page": str(data.get("releases_page", fallback["releases_page"])),
    }


def _version_candidates(project_root: Path) -> list[Path]:
    root = Path(project_root)
    return [
        root / "version.json",
        root / "_internal" / "version.json",
        root / "project" / "version.json",
        root / "_internal" / "project" / "version.json",
    ]
