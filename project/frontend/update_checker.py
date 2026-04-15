from __future__ import annotations

import json
import os
import re
import urllib.request
from pathlib import Path

from project.frontend.app_metadata import read_version_info


def update_checks_enabled() -> bool:
    raw = os.environ.get("BRAINFAST_UPDATE_CHECK", "").strip().lower()
    return raw not in {"0", "false", "off", "no"}


def normalize_version_tag(value: str) -> str:
    text = str(value or "").strip()
    if text.lower().startswith("v"):
        return text[1:]
    return text


def version_sort_key(value: str) -> tuple[tuple[int, ...], int, str]:
    text = normalize_version_tag(value)
    if not text:
        return ((0,), 1, "")
    base, sep, suffix = text.partition("-")
    numbers = tuple(int(part) for part in re.findall(r"\d+", base)) or (0,)
    is_prerelease = 0 if sep else 1
    return (numbers, is_prerelease, suffix.lower())


def is_newer_version(candidate: str, current: str) -> bool:
    return version_sort_key(candidate) > version_sort_key(current)


def latest_release_url(project_root: Path) -> str:
    info = read_version_info(project_root)
    raw = os.environ.get("BRAINFAST_RELEASES_PAGE_URL", "").strip()
    if raw:
        return raw
    return info["releases_page"]


def latest_release_api_url(project_root: Path) -> str:
    info = read_version_info(project_root)
    raw = os.environ.get("BRAINFAST_RELEASES_API_URL", "").strip()
    if raw:
        return raw
    return info["releases_api"]


def fetch_latest_release(project_root: Path, *, timeout: float = 4.0) -> dict[str, str]:
    url = latest_release_api_url(project_root)
    req = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "BrainfastDesktopUpdateChecker/1.0",
        },
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("release API returned a non-object payload")
    tag_name = str(payload.get("tag_name", "")).strip()
    if not tag_name:
        raise RuntimeError("release API response did not include tag_name")
    return {
        "version": normalize_version_tag(tag_name),
        "tag": tag_name,
        "name": str(payload.get("name", "")).strip(),
        "published_at": str(payload.get("published_at", "")).strip(),
        "html_url": str(payload.get("html_url") or latest_release_url(project_root)).strip(),
    }


def check_for_update(project_root: Path, *, timeout: float = 4.0) -> dict[str, str | bool]:
    info = read_version_info(project_root)
    current_version = info["version"]
    latest = fetch_latest_release(project_root, timeout=timeout)
    has_update = is_newer_version(latest["version"], current_version)
    return {
        "enabled": update_checks_enabled(),
        "current_version": current_version,
        "latest_version": latest["version"],
        "latest_tag": latest["tag"],
        "latest_name": latest["name"],
        "latest_url": latest["html_url"],
        "published_at": latest["published_at"],
        "has_update": has_update,
    }
