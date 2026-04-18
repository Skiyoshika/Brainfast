"""Sample-class auto-detect registry.

Loads ``configs/sample_class_registry.json`` (a list of regex → class
mappings) so a freshly-launched job whose ID matches a pattern is
automatically associated with the right class prior, removing the need to
type the class name every time.

Registry schema::

    {
      "patterns": [
        {"match": "^3[59]_", "class": "ChATe27"},
        {"match": "^PV",     "class": "PVe3"}
      ]
    }

Patterns are evaluated top-to-bottom; first match wins. Invalid entries
(missing fields or bad regex) are silently skipped so a typo in the
registry does not crash the wizard or the liquify endpoints.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


def detect_class_for_sample(
    sample_id: str | None,
    *,
    registry_path: Path,
) -> str | None:
    """Return the class name whose pattern matches *sample_id*, or ``None``."""
    if not sample_id:
        return None
    if not Path(registry_path).exists():
        return None
    try:
        registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    patterns = registry.get("patterns") or []
    for entry in patterns:
        if not isinstance(entry, dict):
            continue
        pattern = entry.get("match")
        cls = entry.get("class")
        if not pattern or not cls:
            continue
        try:
            if re.search(str(pattern), str(sample_id)):
                return str(cls)
        except re.error:
            # Invalid regex — skip rather than fail
            continue
    return None


def list_known_classes(
    *,
    registry_path: Path,
    priors_root: Path,
) -> list[str]:
    """Return a deduplicated, sorted list of class names known to the system.

    Pulls from two sources so the frontend dropdown surfaces both:
      * Classes declared in the registry (even if no samples corrected yet)
      * Classes that have an on-disk prior (even if not in the registry)
    """
    classes: set[str] = set()

    if Path(registry_path).exists():
        try:
            registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
            for entry in registry.get("patterns") or []:
                if isinstance(entry, dict) and entry.get("class"):
                    classes.add(str(entry["class"]))
        except (OSError, json.JSONDecodeError):
            pass

    if Path(priors_root).exists():
        for child in Path(priors_root).iterdir():
            if child.is_dir() and (child / "landmark_prior.csv").exists():
                classes.add(child.name)

    return sorted(classes)
