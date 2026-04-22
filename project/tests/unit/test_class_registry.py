"""Unit tests for the sample-class auto-detect registry.

The registry maps regex patterns over sample/job IDs to class names so
that a freshly-launched run automatically gets associated with the right
class prior, removing the need for the user to type "ChATe27" every time.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def test_detect_returns_none_when_no_registry(tmp_path):
    from project.scripts.class_registry import detect_class_for_sample

    assert detect_class_for_sample("35_C0", registry_path=tmp_path / "missing.json") is None


def test_detect_matches_first_pattern(tmp_path):
    from project.scripts.class_registry import detect_class_for_sample

    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "patterns": [
                    {"match": r"^3[59]_", "class": "ChATe27"},
                    {"match": r"^4[01]_", "class": "ChATe27"},
                    {"match": r"^PV", "class": "PVe3"},
                ]
            }
        ),
        encoding="utf-8",
    )

    assert detect_class_for_sample("35_C0", registry_path=registry) == "ChATe27"
    assert detect_class_for_sample("39_anything", registry_path=registry) == "ChATe27"
    assert detect_class_for_sample("41_C1", registry_path=registry) == "ChATe27"
    assert detect_class_for_sample("PVe4_full", registry_path=registry) == "PVe3"
    assert detect_class_for_sample("unknown42", registry_path=registry) is None


def test_detect_skips_invalid_pattern_entries(tmp_path):
    from project.scripts.class_registry import detect_class_for_sample

    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {
                "patterns": [
                    {"match": "[invalid(regex"},  # missing class + bad regex
                    {"class": "OnlyClass"},        # missing match
                    {"match": r"^X", "class": "X-class"},  # valid
                ]
            }
        ),
        encoding="utf-8",
    )
    assert detect_class_for_sample("X_sample", registry_path=registry) == "X-class"
    # Invalid entries don't crash; we just keep going.
    assert detect_class_for_sample("not_x", registry_path=registry) is None


def test_list_known_classes_includes_registry_and_priors(tmp_path):
    """``list_known_classes`` powers the frontend dropdown — it should
    surface BOTH classes registered in registry.json AND classes that have
    on-disk priors, even if the registry hasn't been updated."""
    from project.scripts.class_registry import list_known_classes

    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps(
            {"patterns": [{"match": r"^pv", "class": "PVe3"}]}
        ),
        encoding="utf-8",
    )
    priors_root = tmp_path / "class_priors"
    (priors_root / "ChATe27").mkdir(parents=True)
    (priors_root / "ChATe27" / "landmark_prior.csv").write_text("z,atlas_y,atlas_x,sum_dy,sum_dx,sum_sq_dy,sum_sq_dx,n\n", encoding="utf-8")
    (priors_root / "Untracked").mkdir()
    (priors_root / "Untracked" / "landmark_prior.csv").write_text("z,atlas_y,atlas_x,sum_dy,sum_dx,sum_sq_dy,sum_sq_dx,n\n", encoding="utf-8")

    classes = list_known_classes(registry_path=registry, priors_root=priors_root)
    # Both registry and priors classes should appear, deduplicated, sorted
    assert "PVe3" in classes
    assert "ChATe27" in classes
    assert "Untracked" in classes
    assert classes == sorted(set(classes))


def test_list_known_classes_handles_missing_inputs(tmp_path):
    from project.scripts.class_registry import list_known_classes

    # Both inputs missing → empty list, no crash
    assert list_known_classes(
        registry_path=tmp_path / "no.json",
        priors_root=tmp_path / "no_priors",
    ) == []


def test_detect_handles_empty_sample_id(tmp_path):
    from project.scripts.class_registry import detect_class_for_sample

    registry = tmp_path / "registry.json"
    registry.write_text(
        json.dumps({"patterns": [{"match": r"^x", "class": "x"}]}),
        encoding="utf-8",
    )
    assert detect_class_for_sample("", registry_path=registry) is None
    assert detect_class_for_sample(None, registry_path=registry) is None  # type: ignore[arg-type]
