# Cellpose Parameter Panel — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add interactive Cellpose parameter controls (model selector, diameter, thresholds) to the detection preview, so users can tune detection without editing JSON config files.

**Architecture:** New `/api/cellpose/models` endpoint lists available models. The existing `/api/detect/preview` endpoint accepts optional parameter overrides. Frontend adds a collapsible panel of controls above the Detect Cells button, sending overrides with each detection request.

**Tech Stack:** Python/Flask (backend API), vanilla JS (frontend controls), Cellpose 4.1.1 (`cellpose.models.get_user_models()`, `CellposeModel`)

**Spec:** `docs/superpowers/specs/2026-04-13-cellpose-gui-integration-design.md` §4

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `project/frontend/blueprints/api_cellpose.py` | New blueprint: `/api/cellpose/models` endpoint |
| Create | `project/tests/unit/test_api_cellpose.py` | Tests for the new blueprint |
| Modify | `project/frontend/blueprints/api_detect_preview.py` | Accept `params` override in `/api/detect/preview` |
| Modify | `project/tests/unit/test_detect_preview_api.py` | Test param override behavior |
| Modify | `project/scripts/detect.py` | `_resolve_model_type` supports custom model paths; `detect_cells_cellpose` returns masks optionally |
| Modify | `project/tests/unit/test_detect.py` | Test custom model resolution |
| Modify | `project/frontend/server.py` | Register `api_cellpose` blueprint |
| Modify | `project/frontend/index.html` | Add parameter panel HTML |
| Modify | `project/frontend/app.js` | Add parameter panel JS + send overrides |

---

### Task 1: `/api/cellpose/models` endpoint

**Files:**
- Create: `project/frontend/blueprints/api_cellpose.py`
- Create: `project/tests/unit/test_api_cellpose.py`
- Modify: `project/frontend/server.py:50-72`

- [ ] **Step 1: Write the failing test**

Create `project/tests/unit/test_api_cellpose.py`:

```python
"""Unit tests for the api_cellpose blueprint."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture()
def app(tmp_path):
    """Create a minimal Flask app with the cellpose blueprint."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "outputs").mkdir()

    import project.frontend.server_context as ctx

    ctx.ROOT = project_root / "frontend"
    ctx.PROJECT_ROOT = project_root
    ctx.OUTPUT_DIR = project_root / "outputs"

    from flask import Flask

    from project.frontend.blueprints.api_cellpose import bp

    flask_app = Flask(__name__)
    flask_app.register_blueprint(bp)
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


class TestListModels:
    def test_returns_builtin_models(self, client):
        """Endpoint lists at least the built-in model names."""
        with patch(
            "project.frontend.blueprints.api_cellpose._get_user_models",
            return_value=[],
        ):
            resp = client.get("/api/cellpose/models")
            assert resp.status_code == 200
            data = resp.get_json()
            assert data["ok"] is True
            names = [m["name"] for m in data["models"]]
            assert "cpsam" in names
            assert "cyto3" in names
            assert "cyto2" in names
            assert "nuclei" in names
            assert all(m["type"] == "builtin" for m in data["models"] if m["name"] in names)

    def test_includes_user_models(self, client):
        """User-trained models appear with type='custom'."""
        with patch(
            "project.frontend.blueprints.api_cellpose._get_user_models",
            return_value=["brainfast_v1_20260413", "my_custom"],
        ):
            resp = client.get("/api/cellpose/models")
            data = resp.get_json()
            names = [m["name"] for m in data["models"]]
            assert "brainfast_v1_20260413" in names
            assert "my_custom" in names
            custom_models = [m for m in data["models"] if m["type"] == "custom"]
            assert len(custom_models) == 2

    def test_handles_cellpose_import_error(self, client):
        """Endpoint returns empty list when cellpose is not installed."""
        with patch(
            "project.frontend.blueprints.api_cellpose._get_user_models",
            side_effect=ImportError("No module named 'cellpose'"),
        ):
            resp = client.get("/api/cellpose/models")
            data = resp.get_json()
            assert data["ok"] is True
            # Still returns builtins even if cellpose is not installed
            assert len(data["models"]) >= 4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest project/tests/unit/test_api_cellpose.py -v`
Expected: ImportError — `api_cellpose` module does not exist yet.

- [ ] **Step 3: Write the endpoint**

Create `project/frontend/blueprints/api_cellpose.py`:

```python
"""api_cellpose.py — Cellpose model management and training endpoints.

Provides endpoints for listing available models, parameter validation,
and (in future subsystems) training and model comparison.
"""

from __future__ import annotations

from flask import Blueprint, jsonify

bp = Blueprint("api_cellpose", __name__, url_prefix="/api/cellpose")

_BUILTIN_MODELS = ["cpsam", "cyto3", "cyto2", "nuclei"]


def _get_user_models() -> list[str]:
    """Return names of user-trained Cellpose models.

    Wraps ``cellpose.models.get_user_models()`` with import safety.
    """
    from cellpose.models import get_user_models

    return list(get_user_models())


@bp.get("/models")
def list_models():
    """List all available Cellpose models (built-in + user-trained).

    Returns:
        JSON: ``{"ok": true, "models": [{"name": str, "type": "builtin"|"custom"}, ...]}``
    """
    models = [{"name": n, "type": "builtin"} for n in _BUILTIN_MODELS]

    try:
        user_models = _get_user_models()
    except Exception:
        user_models = []

    for name in user_models:
        models.append({"name": name, "type": "custom"})

    return jsonify({"ok": True, "models": models})
```

- [ ] **Step 4: Register blueprint in server.py**

Add import and registration in `project/frontend/server.py`:

After line 54 (`from project.frontend.blueprints.api_detect_preview import bp as detect_preview_bp`), add:

```python
from project.frontend.blueprints.api_cellpose import bp as cellpose_bp
```

After line 72 (`app.register_blueprint(detect_preview_bp)`), add:

```python
    app.register_blueprint(cellpose_bp)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest project/tests/unit/test_api_cellpose.py -v`
Expected: 3 passed.

- [ ] **Step 6: Commit**

```bash
git add project/frontend/blueprints/api_cellpose.py project/tests/unit/test_api_cellpose.py project/frontend/server.py
git commit -m "feat: add /api/cellpose/models endpoint for listing available models"
```

---

### Task 2: `/api/detect/preview` accepts parameter overrides

**Files:**
- Modify: `project/frontend/blueprints/api_detect_preview.py:90-172`
- Modify: `project/tests/unit/test_detect_preview_api.py`

- [ ] **Step 1: Write the failing test**

Append to `project/tests/unit/test_detect_preview_api.py`:

```python
    def test_param_overrides_applied_to_config(self, app, tmp_path, sample_slice):
        """When request includes 'params', they override config values."""
        import json

        import pandas as pd

        import project.frontend.server_context as ctx

        # Write a base config
        base_cfg = {
            "detection": {
                "primary_model": "cpsam",
                "cellpose_diameter_um": 12.0,
                "cellpose_flow_threshold": 0.4,
                "cellpose_cellprob_threshold": 0.0,
                "cellpose_min_size_px": 8,
            },
            "input": {"pixel_size_um_xy": 5.0},
            "compute": {"device": "cpu"},
        }
        cfg_file = tmp_path / "base_config.json"
        cfg_file.write_text(json.dumps(base_cfg))
        ctx.run_state["config_path"] = str(cfg_file)

        fake_df = pd.DataFrame({
            "x": [10.0],
            "y": [15.0],
            "detector": ["cyto3"],
        })

        captured_cfg = {}

        def mock_detect(slice_path, cfg):
            captured_cfg.update(cfg)
            return fake_df

        from unittest.mock import patch

        with patch(
            "project.frontend.blueprints.api_detect_preview._run_detection",
            side_effect=mock_detect,
        ):
            client = app.test_client()
            resp = client.post(
                "/api/detect/preview",
                data=json.dumps({
                    "slicePath": str(sample_slice),
                    "jobId": "test_params",
                    "params": {
                        "model": "cyto3",
                        "diameter_um": 20.0,
                        "flow_threshold": 0.6,
                        "cellprob_threshold": -2.0,
                        "min_size_px": 15,
                    },
                }),
                content_type="application/json",
            )
            assert resp.status_code == 200
            det = captured_cfg["detection"]
            assert det["primary_model"] == "cyto3"
            assert det["cellpose_diameter_um"] == 20.0
            assert det["cellpose_flow_threshold"] == 0.6
            assert det["cellpose_cellprob_threshold"] == -2.0
            assert det["cellpose_min_size_px"] == 15

    def test_param_overrides_without_params_key_unchanged(self, app, tmp_path, sample_slice):
        """When request has no 'params', config is used as-is (backward compat)."""
        import json

        import pandas as pd

        import project.frontend.server_context as ctx

        base_cfg = {
            "detection": {
                "primary_model": "cpsam",
                "cellpose_diameter_um": 12.0,
            },
            "input": {"pixel_size_um_xy": 5.0},
            "compute": {"device": "cpu"},
        }
        cfg_file = tmp_path / "compat_config.json"
        cfg_file.write_text(json.dumps(base_cfg))
        ctx.run_state["config_path"] = str(cfg_file)

        fake_df = pd.DataFrame({
            "x": [10.0],
            "y": [15.0],
            "detector": ["cpsam"],
        })

        captured_cfg = {}

        def mock_detect(slice_path, cfg):
            captured_cfg.update(cfg)
            return fake_df

        from unittest.mock import patch

        with patch(
            "project.frontend.blueprints.api_detect_preview._run_detection",
            side_effect=mock_detect,
        ):
            client = app.test_client()
            resp = client.post(
                "/api/detect/preview",
                data=json.dumps({
                    "slicePath": str(sample_slice),
                    "jobId": "test_compat",
                }),
                content_type="application/json",
            )
            assert resp.status_code == 200
            assert captured_cfg["detection"]["primary_model"] == "cpsam"
            assert captured_cfg["detection"]["cellpose_diameter_um"] == 12.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest project/tests/unit/test_detect_preview_api.py::TestDetectPreviewEndpoint::test_param_overrides_applied_to_config -v`
Expected: FAIL — the endpoint currently ignores `params`.

- [ ] **Step 3: Add param override logic to the endpoint**

In `project/frontend/blueprints/api_detect_preview.py`, add a helper function after the `_load_active_config` function (after line 56):

```python
def _apply_param_overrides(cfg: dict, params: dict) -> dict:
    """Merge user-supplied parameter overrides into the detection config.

    Maps frontend param names to config keys:
        model          → detection.primary_model
        diameter_um    → detection.cellpose_diameter_um
        flow_threshold → detection.cellpose_flow_threshold
        cellprob_threshold → detection.cellpose_cellprob_threshold
        min_size_px    → detection.cellpose_min_size_px
    """
    if not params:
        return cfg

    det = dict(cfg.get("detection", {}))

    _PARAM_MAP = {
        "model": "primary_model",
        "diameter_um": "cellpose_diameter_um",
        "flow_threshold": "cellpose_flow_threshold",
        "cellprob_threshold": "cellpose_cellprob_threshold",
        "min_size_px": "cellpose_min_size_px",
        "gpu": "cellpose_gpu",
    }

    for param_key, config_key in _PARAM_MAP.items():
        if param_key in params:
            det[config_key] = params[param_key]

    cfg = dict(cfg)
    cfg["detection"] = det
    return cfg
```

Then modify the `detect_preview()` function. After line 120 (`cfg = _load_active_config()`), add:

```python
    # Apply user parameter overrides from the request
    params = payload.get("params")
    if params:
        cfg = _apply_param_overrides(cfg, params)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest project/tests/unit/test_detect_preview_api.py -v`
Expected: All 11 tests pass (9 existing + 2 new).

- [ ] **Step 5: Commit**

```bash
git add project/frontend/blueprints/api_detect_preview.py project/tests/unit/test_detect_preview_api.py
git commit -m "feat: /api/detect/preview accepts param overrides for model, diameter, thresholds"
```

---

### Task 3: `detect.py` supports custom model paths

**Files:**
- Modify: `project/scripts/detect.py:50-69` (`_resolve_model_type`), `130-151` (`_load_cellpose_model`)
- Modify: `project/tests/unit/test_detect.py`

- [ ] **Step 1: Write the failing test**

Append to `project/tests/unit/test_detect.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest project/tests/unit/test_detect.py::test_resolve_model_type_custom_path -v`
Expected: FAIL — `_resolve_model_type("brainfast_v1_20260413")` returns `"cpsam"` (the default fallback).

- [ ] **Step 3: Update `_resolve_model_type` to pass through unknown names**

In `project/scripts/detect.py`, replace the `_resolve_model_type` function (lines 50-69):

```python
def _resolve_model_type(name: str) -> str:
    """Resolve model name to Cellpose model type.

    Built-in aliases are normalized (e.g., "sam" → "cpsam").
    Unknown names (user-trained models or paths) pass through unchanged
    so they can be loaded via ``CellposeModel(pretrained_model=name)``.
    """
    s = str(name or "").strip().lower()
    if not s:
        return "cpsam"
    if "cpsam" in s or s == "sam":
        return "cpsam"
    if "nuclei" in s:
        return "nuclei"
    if "cyto3" in s:
        return "cyto3"
    if "cyto2" in s:
        return "cyto2"
    if "cyto" in s:
        return "cyto"
    # Not a built-in — assume it's a user-trained model name or path.
    # Return the original (non-lowered) name to preserve path casing.
    return str(name or "").strip()
```

- [ ] **Step 4: Update `_is_cellpose_model` to accept custom names**

In `project/scripts/detect.py`, replace the `_is_cellpose_model` function (lines 72-75):

```python
_BUILTIN_CELLPOSE_NAMES = {"cpsam", "sam", "cyto", "cyto2", "cyto3", "nuclei"}


def _is_cellpose_model(name: str) -> bool:
    """Return True if the model name refers to any Cellpose model.

    Returns True for built-in names, names starting with 'cellpose',
    and any name that is not a known non-Cellpose detector
    (like 'log', 'peak', 'threshold', 'reporter_positive', 'none').
    """
    s = str(name or "").strip().lower()
    if not s or s == "none":
        return False
    if s in _BUILTIN_CELLPOSE_NAMES:
        return True
    if s.startswith("cellpose"):
        return True
    # Non-Cellpose detector names
    if s in ("log", "peak", "threshold", "reporter_positive", "fallback"):
        return False
    # Unknown name — assume it's a user-trained Cellpose model
    return True
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest project/tests/unit/test_detect.py -v`
Expected: All tests pass including the 2 new ones.

- [ ] **Step 6: Run full test suite to verify no regressions**

Run: `python -m pytest project/tests -q`
Expected: 200+ passed, 0 failed.

- [ ] **Step 7: Commit**

```bash
git add project/scripts/detect.py project/tests/unit/test_detect.py
git commit -m "feat: detect.py supports custom/user-trained model names in _resolve_model_type"
```

---

### Task 4: Frontend parameter panel HTML

**Files:**
- Modify: `project/frontend/index.html:395-399`

- [ ] **Step 1: Add parameter panel HTML**

In `project/frontend/index.html`, replace the single detect button line (line 398) with a collapsible parameter panel:

```html
          <details id="detectParamsPanel" class="detect-params-panel">
            <summary class="btn-secondary" style="cursor:pointer;display:inline-flex;align-items:center;gap:6px;">
              <i data-lucide="settings" class="btn-icon"></i>
              <span data-i18n="btn.detectParams">Detection Parameters</span>
            </summary>
            <div class="detect-params-grid">
              <label class="field-label">
                <span data-i18n="detect.model">Model</span>
                <select id="detectModelSelect">
                  <option value="cpsam">cpsam (Cellpose-SAM)</option>
                  <option value="cyto3">cyto3</option>
                  <option value="cyto2">cyto2</option>
                  <option value="nuclei">nuclei</option>
                </select>
              </label>
              <label class="field-label">
                <span data-i18n="detect.diameter">Diameter (µm)</span>
                <input id="detectDiameterUm" type="number" min="1" max="100" step="0.5" value="12.0" />
              </label>
              <label class="field-label">
                <span data-i18n="detect.flowThreshold">Flow Threshold</span>
                <input id="detectFlowThreshold" type="range" min="0" max="1" step="0.05" value="0.4" />
                <span id="detectFlowThresholdVal" class="slider-val">0.4</span>
              </label>
              <label class="field-label">
                <span data-i18n="detect.cellprobThreshold">Cell Probability</span>
                <input id="detectCellprobThreshold" type="range" min="-6" max="6" step="0.5" value="0.0" />
                <span id="detectCellprobThresholdVal" class="slider-val">0.0</span>
              </label>
              <label class="field-label">
                <span data-i18n="detect.minSize">Min Size (px)</span>
                <input id="detectMinSizePx" type="number" min="1" max="500" step="1" value="8" />
              </label>
              <label class="field-label" style="flex-direction:row;align-items:center;gap:8px;">
                <input id="detectGpuToggle" type="checkbox" checked style="width:auto;" />
                <span data-i18n="detect.gpu">GPU</span>
              </label>
            </div>
          </details>
          <button id="detectPreviewBtn" class="btn-secondary" data-i18n="btn.detectPreview"><i data-lucide="scan" class="btn-icon"></i> Detect Cells</button>
```

- [ ] **Step 2: Add CSS for the parameter grid**

Append to `project/frontend/styles.css`:

```css
/* Detection parameter panel */
.detect-params-panel {
  margin-bottom: 8px;
}
.detect-params-panel[open] {
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 8px 12px;
  background: var(--bg-secondary, #1a1a2e);
}
.detect-params-grid {
  display: grid;
  grid-template-columns: repeat(auto-fill, minmax(180px, 1fr));
  gap: 8px 16px;
  margin-top: 8px;
}
.detect-params-grid .field-label {
  display: flex;
  flex-direction: column;
  gap: 2px;
  font-size: 0.85em;
}
.detect-params-grid input[type="range"] {
  width: 100%;
}
.slider-val {
  font-size: 0.8em;
  color: var(--text-secondary, #888);
  text-align: right;
  min-width: 36px;
}
```

- [ ] **Step 3: Verify page loads without errors**

Open `http://127.0.0.1:8787` in browser, verify Detection Parameters panel opens/closes, sliders move, no console errors.

- [ ] **Step 4: Commit**

```bash
git add project/frontend/index.html project/frontend/styles.css
git commit -m "feat: add detection parameter panel HTML (model, diameter, thresholds)"
```

---

### Task 5: Frontend JS — populate model dropdown and send overrides

**Files:**
- Modify: `project/frontend/app.js:1187-1246`

- [ ] **Step 1: Add i18n entries**

In `project/frontend/app.js`, add to `LANGS.en` (around line 47):

```javascript
    'btn.detectParams': '<i data-lucide="settings" class="btn-icon"></i> Detection Parameters',
    'detect.model': 'Model',
    'detect.diameter': 'Diameter (µm)',
    'detect.flowThreshold': 'Flow Threshold',
    'detect.cellprobThreshold': 'Cell Probability',
    'detect.minSize': 'Min Size (px)',
    'detect.gpu': 'GPU',
```

Add corresponding Chinese entries in `LANGS.zh` (around line 381):

```javascript
    'btn.detectParams': '<i data-lucide="settings" class="btn-icon"></i> 检测参数',
    'detect.model': '模型',
    'detect.diameter': '直径 (µm)',
    'detect.flowThreshold': '流量阈值',
    'detect.cellprobThreshold': '细胞概率',
    'detect.minSize': '最小面积 (px)',
    'detect.gpu': 'GPU',
```

- [ ] **Step 2: Add model dropdown loader**

Insert after the `runDetectPreview` function definition (after line 1241) and before the `onclick` binding (line 1243):

```javascript
// --- Detection parameter panel logic ---
async function loadCellposeModels() {
  try {
    const res = await fetch('/api/cellpose/models');
    const data = await res.json();
    if (!data.ok) return;

    const select = document.getElementById('detectModelSelect');
    select.innerHTML = '';
    for (const m of data.models) {
      const opt = document.createElement('option');
      opt.value = m.name;
      opt.textContent = m.type === 'custom' ? `${m.name} (custom)` : m.name;
      select.appendChild(opt);
    }
  } catch (err) {
    console.warn('Failed to load Cellpose models:', err);
  }
}

// Slider value displays
document.getElementById('detectFlowThreshold').oninput = function() {
  document.getElementById('detectFlowThresholdVal').textContent = this.value;
};
document.getElementById('detectCellprobThreshold').oninput = function() {
  document.getElementById('detectCellprobThresholdVal').textContent = this.value;
};

// Load models when panel is first opened
document.getElementById('detectParamsPanel').addEventListener('toggle', function() {
  if (this.open) loadCellposeModels();
});
```

- [ ] **Step 3: Modify `runDetectPreview()` to send parameter overrides**

Replace the fetch body in `runDetectPreview()` (line 1206) from:

```javascript
      body: JSON.stringify({ slicePath, jobId: getOverlayJobId() }),
```

to:

```javascript
      body: JSON.stringify({
        slicePath,
        jobId: getOverlayJobId(),
        params: {
          model: document.getElementById('detectModelSelect').value,
          diameter_um: parseFloat(document.getElementById('detectDiameterUm').value) || 12.0,
          flow_threshold: parseFloat(document.getElementById('detectFlowThreshold').value),
          cellprob_threshold: parseFloat(document.getElementById('detectCellprobThreshold').value),
          min_size_px: parseInt(document.getElementById('detectMinSizePx').value, 10) || 8,
          gpu: document.getElementById('detectGpuToggle').checked,
        },
      }),
```

- [ ] **Step 4: Verify end-to-end**

1. Start server: `cd project && python frontend/server.py`
2. Open `http://127.0.0.1:8787`
3. Load a slice, expand "Detection Parameters" panel
4. Verify model dropdown populates (should show cpsam, cyto3, cyto2, nuclei)
5. Adjust diameter slider, click "Detect Cells"
6. Verify detection runs with the overridden parameters (check server log for model/diameter used)

- [ ] **Step 5: Commit**

```bash
git add project/frontend/app.js
git commit -m "feat: frontend sends detection param overrides (model, diameter, thresholds)"
```

---

### Task 6: Full integration test — end-to-end parameter override

**Files:**
- Modify: `project/tests/unit/test_detect_preview_api.py`

- [ ] **Step 1: Write integration-level test**

Append to `project/tests/unit/test_detect_preview_api.py`:

```python
    def test_param_override_model_reaches_detector(self, app, tmp_path, sample_slice):
        """Full chain: param override 'model' changes which model string
        reaches the actual detect_cells function."""
        import json

        import pandas as pd

        import project.frontend.server_context as ctx

        # Config says cpsam, but param override says cyto3
        base_cfg = {
            "detection": {"primary_model": "cpsam"},
            "input": {"pixel_size_um_xy": 5.0},
            "compute": {"device": "cpu"},
        }
        cfg_file = tmp_path / "override_config.json"
        cfg_file.write_text(json.dumps(base_cfg))
        ctx.run_state["config_path"] = str(cfg_file)

        fake_df = pd.DataFrame({
            "x": [10.0], "y": [15.0], "detector": ["cellpose_cyto3"],
        })

        captured_cfg = {}

        def mock_detect(slice_path, cfg):
            captured_cfg.update(cfg)
            return fake_df

        from unittest.mock import patch

        with patch(
            "project.frontend.blueprints.api_detect_preview._run_detection",
            side_effect=mock_detect,
        ):
            client = app.test_client()
            resp = client.post(
                "/api/detect/preview",
                data=json.dumps({
                    "slicePath": str(sample_slice),
                    "jobId": "test_override_model",
                    "params": {"model": "cyto3"},
                }),
                content_type="application/json",
            )
            assert resp.status_code == 200
            # Config passed to detector must have the overridden model
            assert captured_cfg["detection"]["primary_model"] == "cyto3"
```

- [ ] **Step 2: Run full test suite**

Run: `python -m pytest project/tests -q`
Expected: All tests pass (200+ existing + new ones).

- [ ] **Step 3: Commit**

```bash
git add project/tests/unit/test_detect_preview_api.py
git commit -m "test: add integration test for param override reaching detector"
```

---

### Task 7: Final validation

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest project/tests -q`
Expected: All pass, 0 failures.

- [ ] **Step 2: Verify server starts cleanly**

Run: `cd project && python frontend/server.py`
Expected: No import errors, server starts on port 8787.

- [ ] **Step 3: Spot-check in browser**

1. Open `http://127.0.0.1:8787`
2. Load a sample slice
3. Expand "Detection Parameters" → model dropdown loads
4. Change model to `cyto3`, adjust diameter to 20
5. Click "Detect Cells" → verify detection runs (check server terminal for log line)
6. Verify result banner shows cell count and detector name
