"""Unit tests for config_validation.py — no atlas file required."""
from __future__ import annotations

import json
import pytest

from project.scripts.config_validation import (
    _get,
    _make_issue,
    collect_runtime_config_issues,
    config_value,
    is_placeholder_value,
    load_config,
    validate_runtime_config,
)


# ---------------------------------------------------------------------------
# is_placeholder_value
# ---------------------------------------------------------------------------


def test_placeholder_none():
    assert is_placeholder_value(None) is True


def test_placeholder_empty_string():
    assert is_placeholder_value("") is True


def test_placeholder_todo():
    assert is_placeholder_value("todo") is True
    assert is_placeholder_value("  TODO  ") is True


def test_placeholder_tbd():
    assert is_placeholder_value("TBD") is True


def test_placeholder_real_value():
    assert is_placeholder_value("my_project") is False
    assert is_placeholder_value(5.0) is False
    assert is_placeholder_value(0) is False


# ---------------------------------------------------------------------------
# _get (dotted key access)
# ---------------------------------------------------------------------------


def test_get_nested():
    cfg = {"a": {"b": {"c": 42}}}
    assert _get(cfg, "a.b.c") == 42


def test_get_missing_returns_default():
    cfg = {"a": {}}
    assert _get(cfg, "a.b.c") is None
    assert _get(cfg, "a.b.c", "fallback") == "fallback"


def test_get_top_level():
    cfg = {"name": "test"}
    assert _get(cfg, "name") == "test"


# ---------------------------------------------------------------------------
# _make_issue
# ---------------------------------------------------------------------------


def test_make_issue_structure():
    issue = _make_issue("some.field", "error", "something is wrong")
    assert issue["field"] == "some.field"
    assert issue["severity"] == "error"
    assert "wrong" in issue["message"]


# ---------------------------------------------------------------------------
# config_value
# ---------------------------------------------------------------------------


def test_config_value_resolves_nested():
    cfg = {"input": {"pixel_size_um_xy": 5.0}}
    assert config_value(cfg, "input.pixel_size_um_xy") == 5.0


# ---------------------------------------------------------------------------
# collect_runtime_config_issues — valid config produces only warning
# ---------------------------------------------------------------------------

VALID_CFG = {
    "project": {"name": "TestProject"},
    "input": {
        "slice_glob": "*.tif",
        "pixel_size_um_xy": 5.0,
        "slice_spacing_um": 25.0,
        "channel_map": {"red": 0},
        "active_channel": "red",
    },
    "detection": {"primary_model": "log"},
    "dedup": {"neighbor_slices": 1, "r_xy_um": 6.0},
    "registration": {"atlas_z_refine_range": 0},
    "outputs": {
        "leaf_csv": "outputs/leaf.csv",
        "hierarchy_csv": "outputs/hier.csv",
        "qc_dir": "outputs/qc",
    },
}


def test_valid_config_no_errors():
    issues = collect_runtime_config_issues(VALID_CFG)
    errors = [i for i in issues if i["severity"] == "error"]
    assert errors == [], f"Unexpected errors: {errors}"


def test_valid_config_refine_range_zero_warning():
    issues = collect_runtime_config_issues(VALID_CFG)
    warnings = [i for i in issues if i["severity"] == "warning"]
    assert any("atlas_z_refine_range" in w["field"] for w in warnings)


def test_missing_project_name_raises_error():
    cfg = {**VALID_CFG, "project": {"name": ""}}
    errors = validate_runtime_config(cfg)
    assert any("project.name" in e for e in errors)


def test_missing_pixel_size_raises_error():
    from copy import deepcopy
    cfg = deepcopy(VALID_CFG)
    cfg["input"]["pixel_size_um_xy"] = None
    errors = validate_runtime_config(cfg)
    assert any("pixel_size_um_xy" in e for e in errors)


def test_negative_pixel_size_raises_error():
    from copy import deepcopy
    cfg = deepcopy(VALID_CFG)
    cfg["input"]["pixel_size_um_xy"] = -1.0
    errors = validate_runtime_config(cfg)
    assert any("pixel_size_um_xy" in e for e in errors)


def test_empty_channel_map_raises_error():
    from copy import deepcopy
    cfg = deepcopy(VALID_CFG)
    cfg["input"]["channel_map"] = {}
    errors = validate_runtime_config(cfg)
    assert any("channel_map" in e for e in errors)


def test_active_channel_not_in_map_raises_error():
    from copy import deepcopy
    cfg = deepcopy(VALID_CFG)
    cfg["input"]["active_channel"] = "green"  # not in channel_map (only "red")
    errors = validate_runtime_config(cfg)
    assert any("active_channel" in e for e in errors)


def test_require_input_dir_flag():
    from copy import deepcopy
    cfg = deepcopy(VALID_CFG)
    # slice_dir not set → error when require_input_dir=True
    errors = validate_runtime_config(cfg, require_input_dir=True)
    assert any("slice_dir" in e for e in errors)


def test_nonnegative_int_negative_neighbor_slices():
    from copy import deepcopy
    cfg = deepcopy(VALID_CFG)
    cfg["dedup"]["neighbor_slices"] = -1
    errors = validate_runtime_config(cfg)
    assert any("neighbor_slices" in e for e in errors)


def test_nonneg_int_non_numeric():
    from copy import deepcopy
    cfg = deepcopy(VALID_CFG)
    cfg["dedup"]["neighbor_slices"] = "abc"
    errors = validate_runtime_config(cfg)
    assert any("neighbor_slices" in e for e in errors)


# ---------------------------------------------------------------------------
# load_config
# ---------------------------------------------------------------------------


def test_load_config_valid(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps({"key": "val"}), encoding="utf-8")
    data = load_config(p)
    assert data["key"] == "val"


def test_load_config_non_dict_raises(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_config(p)
