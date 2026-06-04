from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
from tifffile import imwrite

# Import the CellposeRuntimeError from the same module path that detect.py
# actually uses at runtime, so isinstance checks in pytest.raises match.
import project.scripts.detect as _detect_mod
from project.scripts.detect import (
    _is_cellpose_model,
    _resolve_model_type,
    detect_cells,
    detect_cells_cellpose,
    detect_cells_fallback,
    detect_cells_log_fallback,
)

CellposeRuntimeError = _detect_mod.CellposeRuntimeError
CellposeDetectionError = _detect_mod.CellposeDetectionError


@pytest.fixture()
def tiny_slice(tmp_path: Path) -> Path:
    """Create a small 16-bit TIFF with a few bright spots."""
    img = np.zeros((64, 64), dtype=np.uint16)
    img[20, 20] = 5000
    img[40, 40] = 6000
    p = tmp_path / "slice.tif"
    imwrite(str(p), img)
    return p


# ── detect_cells_cellpose: import failure ────────────────────────────────────


def test_cellpose_raises_on_import_failure(tiny_slice, monkeypatch):
    """When cellpose is not installed, loading the model must raise CellposeDetectionError."""
    monkeypatch.setattr(
        "project.scripts.detect._load_cellpose_model",
        MagicMock(side_effect=ImportError("No module named 'cellpose'")),
    )
    with pytest.raises(CellposeDetectionError, match="failed to load"):
        detect_cells_cellpose(tiny_slice, model_type="cyto2", raise_on_error=True)


# ── detect_cells_cellpose: model init failure ────────────────────────────────


def test_cellpose_raises_on_model_init_failure(tiny_slice, monkeypatch):
    monkeypatch.setattr(
        "project.scripts.detect._load_cellpose_model",
        MagicMock(side_effect=RuntimeError("CUDA OOM")),
    )
    with pytest.raises(CellposeDetectionError, match="failed to load"):
        detect_cells_cellpose(tiny_slice, model_type="cyto2", raise_on_error=True)


# ── detect_cells_cellpose: inference failure ─────────────────────────────────


def test_cellpose_raises_on_inference_failure(tiny_slice, monkeypatch):
    fake_model = MagicMock()
    fake_model.eval.side_effect = RuntimeError("segfault in inference")
    monkeypatch.setattr(
        "project.scripts.detect._load_cellpose_model",
        MagicMock(return_value=(fake_model, False)),
    )
    with pytest.raises(CellposeDetectionError, match="inference failed"):
        detect_cells_cellpose(tiny_slice, model_type="cyto2", raise_on_error=True)


# ── detect_cells: Cellpose failure propagates when auto_switch disabled ──────


def test_detect_cells_propagates_cellpose_error_when_auto_switch_off(tiny_slice, monkeypatch):
    monkeypatch.setattr(
        "project.scripts.detect._load_cellpose_model",
        MagicMock(side_effect=ImportError("no cellpose")),
    )
    cfg = {
        "detection": {
            "primary_model": "cellpose_cyto2",
            "auto_switch_on_distortion": False,
        }
    }
    with pytest.raises((CellposeDetectionError, CellposeRuntimeError)):
        detect_cells(tiny_slice, cfg)


# ── detect_cells: Cellpose failure falls back when auto_switch enabled ───────


def test_detect_cells_falls_back_when_auto_switch_on(tiny_slice, monkeypatch):
    monkeypatch.setattr(
        "project.scripts.detect._load_cellpose_model",
        MagicMock(side_effect=ImportError("no cellpose")),
    )
    cfg = {
        "detection": {
            "primary_model": "cellpose_cyto2",
            "secondary_model": "cellpose_nuclei",
            "auto_switch_on_distortion": True,
            "fallback_model": "log",
        }
    }
    # Should NOT raise — falls back to non-Cellpose detector
    result = detect_cells(tiny_slice, cfg)
    assert isinstance(result, pd.DataFrame)


# ── detect_cells: successful fallback when configured ────────────────────────


def test_detect_cells_uses_fallback_when_configured(tiny_slice):
    """When mode is not cellpose, the fallback detector runs without error."""
    cfg = {
        "detection": {
            "mode": "fallback",
            "primary_model": "log",
            "secondary_model": "log",
            "fallback_model": "log",
        }
    }
    result = detect_cells(tiny_slice, cfg)
    assert isinstance(result, pd.DataFrame)


# ── detect_cells_fallback: basic ─────────────────────────────────────────────


def test_fallback_detector_returns_dataframe(tiny_slice):
    df = detect_cells_fallback(tiny_slice, min_distance=5, threshold_abs=1000.0)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"cell_id", "x", "y", "score", "detector", "area_px"}


# ── detect_cells_log_fallback: basic ─────────────────────────────────────────


def test_log_fallback_returns_dataframe(tiny_slice):
    df = detect_cells_log_fallback(tiny_slice)
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns) >= {"cell_id", "x", "y", "score", "detector", "area_px"}


# ── _resolve_model_type tests ──────────────────────────────────────────────────


def test_resolve_model_type_cpsam():
    assert _resolve_model_type("cpsam") == "cpsam"


def test_resolve_model_type_sam():
    assert _resolve_model_type("sam") == "cpsam"


def test_resolve_model_type_nuclei():
    assert _resolve_model_type("nuclei") == "nuclei"


def test_resolve_model_type_cyto3():
    assert _resolve_model_type("cyto3") == "cyto3"


def test_resolve_model_type_cyto2():
    assert _resolve_model_type("cyto2") == "cyto2"


def test_resolve_model_type_cyto():
    assert _resolve_model_type("cyto") == "cyto"


def test_resolve_model_type_empty_defaults_to_cpsam():
    """An empty name should default to cpsam (Cellpose-SAM v4+)."""
    assert _resolve_model_type("") == "cpsam"


def test_resolve_model_type_custom_path():
    """Custom model paths (not built-in names) pass through unchanged."""
    from project.scripts.detect import _resolve_model_type

    # Absolute path to a custom model
    assert _resolve_model_type("C:/Users/me/.cellpose/models/brainfast_v1") == "brainfast_v1"
    assert _resolve_model_type("/home/user/models/my_custom") == "my_custom"

    # Name that looks like a user-trained model (not a built-in)
    assert _resolve_model_type("brainfast_v1_20260413") == "brainfast_v1_20260413"


def test_resolve_model_type_builtin_names():
    """Built-in model names still resolve correctly."""
    from project.scripts.detect import _resolve_model_type

    assert _resolve_model_type("cpsam") == "cpsam"
    assert _resolve_model_type("cyto3") == "cyto3"
    assert _resolve_model_type("cyto2") == "cyto2"
    assert _resolve_model_type("nuclei") == "nuclei"
    assert _resolve_model_type("cyto") == "cyto"


# ── _is_cellpose_model tests ──────────────────────────────────────────────────


def test_is_cellpose_model_cpsam():
    assert _is_cellpose_model("cpsam") is True


def test_is_cellpose_model_sam():
    assert _is_cellpose_model("sam") is True


def test_is_cellpose_model_cellpose_prefix():
    assert _is_cellpose_model("cellpose_cyto2") is True
    assert _is_cellpose_model("cellpose_nuclei") is True


def test_is_cellpose_model_nuclei():
    assert _is_cellpose_model("nuclei") is True


def test_is_cellpose_model_log_returns_false():
    assert _is_cellpose_model("log") is False


def test_is_cellpose_model_empty_returns_false():
    assert _is_cellpose_model("") is False


def test_is_cellpose_model_fallback_peak_returns_true():
    """'fallback_peak' is not a known non-Cellpose name, so it's treated as a custom model."""
    assert _is_cellpose_model("fallback_peak") is True


# ── _load_cellpose_model: CellposeModel (v4+) branch ──────────────────────────


def test_load_cellpose_model_v4_branch(monkeypatch):
    """When models.CellposeModel exists, it should be called with pretrained_model=..."""
    import project.scripts.detect as dm

    # Clear the model cache so our mock is used
    monkeypatch.setattr(dm, "_CELLPOSE_MODEL_CACHE", {})

    fake_model_instance = MagicMock()
    FakeCellposeModel = MagicMock(return_value=fake_model_instance)

    # Build a fake models module with CellposeModel attribute
    class FakeModelsV4:
        CellposeModel = FakeCellposeModel
        Cellpose = MagicMock()  # also present but CellposeModel takes priority

    fake_models_v4 = FakeModelsV4()

    # Patch sys.modules BEFORE calling _load_cellpose_model so that
    # `from cellpose import models` inside the function resolves to our fake.
    # We must also remove any cached real cellpose.models first.
    saved_cellpose = sys.modules.get("cellpose")
    saved_models = sys.modules.get("cellpose.models")
    try:
        sys.modules["cellpose"] = MagicMock(models=fake_models_v4)
        sys.modules["cellpose.models"] = fake_models_v4
        result = dm._load_cellpose_model("cpsam", use_gpu=False)
    finally:
        # Restore original modules
        if saved_cellpose is not None:
            sys.modules["cellpose"] = saved_cellpose
        else:
            sys.modules.pop("cellpose", None)
        if saved_models is not None:
            sys.modules["cellpose.models"] = saved_models
        else:
            sys.modules.pop("cellpose.models", None)

    FakeCellposeModel.assert_called_once_with(gpu=False, pretrained_model="cpsam")
    assert result == (fake_model_instance, False)  # (model, is_legacy=False)


def test_load_cellpose_model_legacy_branch(monkeypatch):
    """When only models.Cellpose exists (no CellposeModel), use legacy path."""
    import project.scripts.detect as dm

    monkeypatch.setattr(dm, "_CELLPOSE_MODEL_CACHE", {})

    fake_model_instance = MagicMock()
    FakeCellpose = MagicMock(return_value=fake_model_instance)

    # Build a fake models module with only Cellpose (no CellposeModel)
    class FakeModelsLegacy:
        Cellpose = FakeCellpose

    fake_models_legacy = FakeModelsLegacy()

    saved_cellpose = sys.modules.get("cellpose")
    saved_models = sys.modules.get("cellpose.models")
    try:
        sys.modules["cellpose"] = MagicMock(models=fake_models_legacy)
        sys.modules["cellpose.models"] = fake_models_legacy
        result = dm._load_cellpose_model("cyto2", use_gpu=False)
    finally:
        if saved_cellpose is not None:
            sys.modules["cellpose"] = saved_cellpose
        else:
            sys.modules.pop("cellpose", None)
        if saved_models is not None:
            sys.modules["cellpose.models"] = saved_models
        else:
            sys.modules.pop("cellpose.models", None)

    FakeCellpose.assert_called_once_with(gpu=False, model_type="cyto2")
    assert result == (fake_model_instance, True)  # (model, is_legacy=True)


# ── v4 three-value eval() return handling ──────────────────────────────────────


def test_cellpose_v4_three_value_eval_return(tiny_slice, monkeypatch):
    """Cellpose v4 eval() returns a 3-tuple (masks, flows, styles).
    detect_cells_cellpose should handle this correctly via result[0]."""
    import project.scripts.detect as dm

    masks = np.zeros((64, 64), dtype=np.int32)
    masks[18:23, 18:23] = 1  # one cell region
    masks[38:43, 38:43] = 2  # another cell region

    fake_model = MagicMock()
    # v4 returns 3 values
    fake_model.eval.return_value = (masks, "flows", "styles")

    monkeypatch.setattr(dm, "_CELLPOSE_MODEL_CACHE", {})
    monkeypatch.setattr(
        "project.scripts.detect._load_cellpose_model",
        MagicMock(return_value=(fake_model, False)),
    )

    df = detect_cells_cellpose(tiny_slice, model_type="cpsam")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert set(df.columns) >= {"cell_id", "x", "y", "score", "detector", "area_px"}
    assert df["detector"].iloc[0] == "cellpose_cpsam"


def test_cellpose_v4_does_not_pass_channels(tiny_slice, monkeypatch):
    """Cellpose-SAM v4+ ignores `channels`; detect_cells_cellpose must NOT
    include it in kwargs when CellposeModel is present."""
    import project.scripts.detect as dm

    masks = np.zeros((64, 64), dtype=np.int32)
    masks[18:23, 18:23] = 1

    fake_model = MagicMock()
    fake_model.eval.return_value = (masks, "flows", "styles")

    monkeypatch.setattr(dm, "_CELLPOSE_MODEL_CACHE", {})
    monkeypatch.setattr(
        "project.scripts.detect._load_cellpose_model",
        MagicMock(return_value=(fake_model, False)),  # v4+, not legacy
    )

    detect_cells_cellpose(tiny_slice, model_type="cpsam", channels=[0, 0])
    # Verify channels was NOT passed to eval
    call_kwargs = fake_model.eval.call_args[1]
    assert "channels" not in call_kwargs, (
        f"v4 path must NOT pass 'channels' to eval, got: {call_kwargs}"
    )
    assert "tile" not in call_kwargs, (
        f"current CellposeModel.eval does not accept 'tile'; got: {call_kwargs}"
    )


def test_cellpose_v3_passes_channels(tiny_slice, monkeypatch):
    """Legacy Cellpose v2/v3 requires `channels` in eval kwargs."""
    import project.scripts.detect as dm

    masks = np.zeros((64, 64), dtype=np.int32)
    masks[18:23, 18:23] = 1

    fake_model = MagicMock()
    fake_model.eval.return_value = (masks, "flows", "styles", "diams")

    monkeypatch.setattr(dm, "_CELLPOSE_MODEL_CACHE", {})
    monkeypatch.setattr(
        "project.scripts.detect._load_cellpose_model",
        MagicMock(return_value=(fake_model, True)),  # legacy v2/v3
    )

    detect_cells_cellpose(tiny_slice, model_type="cyto2", channels=[1, 0])
    call_kwargs = fake_model.eval.call_args[1]
    assert "channels" in call_kwargs, (
        f"v3 legacy path must pass 'channels' to eval, got: {call_kwargs}"
    )
    assert call_kwargs["channels"] == [1, 0]


def test_masks_to_centroids_resizes_masks_to_intensity_shape():
    """Cellpose v4 may return masks on its internal resized grid."""
    import project.scripts.detect as dm

    masks = np.array([[0, 1], [0, 1]], dtype=np.int32)
    intensity = np.ones((4, 4), dtype=np.float32)

    df = dm._masks_to_centroids(masks, "cellpose_cpsam", intensity_image=intensity)

    assert len(df) == 1
    assert float(df["x"].iloc[0]) > 1.0
    assert float(df["mean_intensity"].iloc[0]) == 1.0


def test_cellpose_v3_four_value_eval_return(tiny_slice, monkeypatch):
    """Cellpose v2/v3 eval() returns a 4-tuple (masks, flows, styles, diams).
    detect_cells_cellpose should handle this correctly via result[0]."""
    import project.scripts.detect as dm

    masks = np.zeros((64, 64), dtype=np.int32)
    masks[18:23, 18:23] = 1

    fake_model = MagicMock()
    # v2/v3 returns 4 values
    fake_model.eval.return_value = (masks, "flows", "styles", "diams")

    monkeypatch.setattr(dm, "_CELLPOSE_MODEL_CACHE", {})
    monkeypatch.setattr(
        "project.scripts.detect._load_cellpose_model",
        MagicMock(return_value=(fake_model, True)),  # legacy v2/v3
    )

    df = detect_cells_cellpose(tiny_slice, model_type="cyto2")
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    assert df["detector"].iloc[0] == "cellpose_cyto2"


# ── No-silent-fallback regression test ─────────────────────────────────────────


def test_no_silent_fallback_when_auto_switch_off_and_cellpose_fails(tiny_slice, monkeypatch):
    """Regression: when auto_switch_on_distortion=False and Cellpose raises,
    the error must propagate -- never silently fall back to LoG/peak."""
    monkeypatch.setattr(
        "project.scripts.detect._load_cellpose_model",
        MagicMock(side_effect=CellposeRuntimeError("GPU exploded")),
    )
    cfg = {
        "detection": {
            "primary_model": "cpsam",
            "secondary_model": "cellpose_nuclei",
            "auto_switch_on_distortion": False,
            "fallback_model": "log",
        }
    }
    with pytest.raises((CellposeDetectionError, CellposeRuntimeError)):
        detect_cells(tiny_slice, cfg)


def test_no_silent_fallback_when_auto_switch_off_and_cellpose_returns_empty(
    tiny_slice, monkeypatch
):
    """Regression: when auto_switch_on_distortion=False and Cellpose returns
    zero cells (empty masks), detect_cells must NOT silently fall through to
    LoG.  With the current code, it returns an empty DataFrame rather than
    raising, which still prevents silent fallback to a different detector."""
    empty_masks = np.zeros((64, 64), dtype=np.int32)
    fake_model = MagicMock()
    fake_model.eval.return_value = (empty_masks, "flows", "styles")

    monkeypatch.setattr(
        "project.scripts.detect._load_cellpose_model",
        MagicMock(return_value=(fake_model, False)),
    )

    cfg = {
        "detection": {
            "primary_model": "cpsam",
            "secondary_model": "cellpose_nuclei",
            "auto_switch_on_distortion": False,
            "fallback_model": "log",
        }
    }
    # With allow_fallback=False, Cellpose returning empty masks yields an
    # empty DataFrame (no silent LoG fallback) rather than raising.
    result = detect_cells(tiny_slice, cfg)
    assert isinstance(result, pd.DataFrame)
    assert result.empty or len(result) == 0


def test_detect_cells_cellpose_returns_masks_when_requested(monkeypatch):
    """When return_masks=True, detect_cells_cellpose returns (df, masks) tuple."""
    from pathlib import Path
    from unittest.mock import MagicMock

    import numpy as np

    from project.scripts.detect import detect_cells_cellpose

    fake_masks = np.array([[0, 0, 1], [0, 1, 1], [2, 2, 0]], dtype=np.int32)

    fake_model = MagicMock()
    fake_model.eval.return_value = (fake_masks, None, None)

    monkeypatch.setattr(
        "project.scripts.detect._load_cellpose_model",
        lambda **kwargs: (fake_model, False),
    )
    monkeypatch.setattr(
        "project.scripts.detect._read_gray",
        lambda p: np.zeros((3, 3), dtype=np.float32),
    )
    monkeypatch.setattr(
        "project.scripts.detect._norm_for_cellpose",
        lambda img: img,
    )

    result = detect_cells_cellpose(Path("fake.tif"), "cyto3", return_masks=True)
    assert isinstance(result, tuple)
    assert len(result) == 2
    df, masks = result
    assert len(df) == 2  # two cells (labels 1 and 2)
    assert masks.shape == (3, 3)
    assert int(masks.max()) == 2


def test_detect_cells_cellpose_default_returns_df_only(monkeypatch):
    """Default behavior (return_masks=False) returns just the DataFrame."""
    from pathlib import Path
    from unittest.mock import MagicMock

    import numpy as np
    import pandas as pd

    from project.scripts.detect import detect_cells_cellpose

    fake_masks = np.array([[0, 1], [1, 0]], dtype=np.int32)
    fake_model = MagicMock()
    fake_model.eval.return_value = (fake_masks, None, None)

    monkeypatch.setattr(
        "project.scripts.detect._load_cellpose_model",
        lambda **kwargs: (fake_model, False),
    )
    monkeypatch.setattr(
        "project.scripts.detect._read_gray",
        lambda p: np.zeros((2, 2), dtype=np.float32),
    )
    monkeypatch.setattr(
        "project.scripts.detect._norm_for_cellpose",
        lambda img: img,
    )

    result = detect_cells_cellpose(Path("fake.tif"), "cyto3")
    assert isinstance(result, pd.DataFrame)
