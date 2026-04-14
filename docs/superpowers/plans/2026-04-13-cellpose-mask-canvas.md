# Cellpose Mask Annotation Canvas — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a mask annotation canvas where users correct Cellpose detection results (add/delete/paint cells), then save corrected masks as training data for fine-tuning.

**Architecture:** Backend returns full instance masks (not just centroids) via a new `/api/detect/preview/masks` endpoint. A new full-screen modal (`mask-editor.js`) renders the original image + colored mask overlay on a two-layer HTML5 Canvas. Tools (brush, eraser, add cell, delete, merge, split) edit the mask in-memory. "Save to Training Set" sends the corrected mask to a new `/api/cellpose/save-training-sample` endpoint which writes Cellpose-convention files (`{name}.tif` + `{name}_masks.tif`).

**Tech Stack:** HTML5 Canvas 2D, vanilla JS (new `mask-editor.js` file), Flask endpoints, numpy/tifffile for mask I/O

**Spec:** `docs/superpowers/specs/2026-04-13-cellpose-gui-integration-design.md` §5

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Create | `project/frontend/mask-editor.js` | Mask editor canvas, tools, undo/redo, rendering |
| Create | `project/frontend/mask-editor.css` | Mask editor modal styles |
| Modify | `project/frontend/blueprints/api_detect_preview.py` | New `/api/detect/preview/masks` endpoint returning raw masks |
| Modify | `project/frontend/blueprints/api_cellpose.py` | New `/api/cellpose/save-training-sample` endpoint |
| Modify | `project/scripts/detect.py` | `detect_cells_cellpose` returns masks alongside centroids |
| Modify | `project/frontend/index.html` | Add mask editor modal HTML + script tag |
| Modify | `project/frontend/app.js` | "Edit Masks" button handler, open/close modal |
| Create | `project/tests/unit/test_mask_endpoints.py` | Tests for masks + save-training-sample endpoints |

---

### Task 1: `detect_cells_cellpose` returns masks alongside centroids

**Files:**
- Modify: `project/scripts/detect.py:369-434`
- Modify: `project/tests/unit/test_detect.py`

- [ ] **Step 1: Write the failing test**

Append to `project/tests/unit/test_detect.py`:

```python
def test_detect_cells_cellpose_returns_masks_when_requested(monkeypatch):
    """When return_masks=True, detect_cells_cellpose returns (df, masks) tuple."""
    import types
    from pathlib import Path
    from unittest.mock import MagicMock

    import numpy as np

    from project.scripts.detect import detect_cells_cellpose

    fake_masks = np.array([[0, 0, 1], [0, 1, 1], [2, 2, 0]], dtype=np.int32)

    fake_model = MagicMock()
    fake_model.eval.return_value = (fake_masks, None, None)

    monkeypatch.setattr(
        "project.scripts.detect._load_cellpose_model",
        lambda **kwargs: fake_model,
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
        lambda **kwargs: fake_model,
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest project/tests/unit/test_detect.py::test_detect_cells_cellpose_returns_masks_when_requested -v`
Expected: TypeError — `detect_cells_cellpose` doesn't accept `return_masks`.

- [ ] **Step 3: Add `return_masks` parameter to `detect_cells_cellpose`**

In `project/scripts/detect.py`, modify the `detect_cells_cellpose` function signature (line 369) to add `return_masks: bool = False`:

```python
def detect_cells_cellpose(
    slice_path: Path,
    model_type: str = "cyto3",
    diameter_px: float | None = None,
    *,
    use_gpu: bool = False,
    channels: list[int] | None = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    min_size: int = 8,
    return_masks: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
```

Then at the end of the function (replacing the current last line `return _masks_to_centroids(masks, ...)`), change to:

```python
    df = _masks_to_centroids(masks, detector=f"cellpose_{model_type}")
    if return_masks:
        return df, masks
    return df
```

Apply the same pattern to BOTH the normal path AND the retry path inside the function. The easiest approach: extract the masks variable from `result[0]` (already done), then change only the final return at the bottom.

- [ ] **Step 4: Run tests**

Run: `python -m pytest project/tests/unit/test_detect.py -v`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add project/scripts/detect.py project/tests/unit/test_detect.py
git commit -m "feat: detect_cells_cellpose supports return_masks=True for mask editing"
```

---

### Task 2: `/api/detect/preview/masks` endpoint

**Files:**
- Modify: `project/frontend/blueprints/api_detect_preview.py`
- Create: `project/tests/unit/test_mask_endpoints.py`

- [ ] **Step 1: Write the failing test**

Create `project/tests/unit/test_mask_endpoints.py`:

```python
"""Tests for mask-related endpoints (detect preview masks + save training sample)."""

from __future__ import annotations

import io
import json
import zlib
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from tifffile import imwrite


@pytest.fixture()
def app(tmp_path):
    project_root = tmp_path / "project"
    project_root.mkdir()
    (project_root / "outputs").mkdir()

    import project.frontend.server_context as ctx

    ctx.ROOT = project_root / "frontend"
    ctx.PROJECT_ROOT = project_root
    ctx.OUTPUT_DIR = project_root / "outputs"

    from flask import Flask

    from project.frontend.blueprints.api_cellpose import bp as cellpose_bp
    from project.frontend.blueprints.api_detect_preview import bp as detect_bp

    flask_app = Flask(__name__)
    flask_app.register_blueprint(detect_bp)
    flask_app.register_blueprint(cellpose_bp)
    flask_app.config["TESTING"] = True
    return flask_app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def sample_slice(tmp_path):
    slice_path = tmp_path / "test_slice.tif"
    img = np.random.randint(0, 65535, (64, 64), dtype=np.uint16)
    imwrite(str(slice_path), img)
    return slice_path


class TestMasksEndpoint:
    def test_returns_masks_after_detection(self, client, sample_slice):
        """After a detection run, /masks returns the raw mask array."""
        import pandas as pd

        fake_masks = np.array([[0, 1, 1], [0, 2, 2], [0, 0, 0]], dtype=np.int32)
        fake_df = pd.DataFrame({
            "x": [1.5, 1.5], "y": [0.0, 1.0],
            "detector": ["cellpose_cyto3"] * 2,
            "cell_id": [1, 2], "score": [1.0, 1.0], "area_px": [2.0, 2.0],
        })

        def mock_run_detection_with_masks(slice_path, cfg):
            return fake_df, fake_masks

        with patch(
            "project.frontend.blueprints.api_detect_preview._run_detection_with_masks",
            side_effect=mock_run_detection_with_masks,
        ):
            # First run detection
            resp = client.post(
                "/api/detect/preview",
                data=json.dumps({
                    "slicePath": str(sample_slice),
                    "jobId": "mask_test",
                    "returnMasks": True,
                }),
                content_type="application/json",
            )
            assert resp.status_code == 200

            # Then fetch masks
            resp2 = client.get("/api/detect/preview/masks?jobId=mask_test")
            assert resp2.status_code == 200
            data = resp2.get_json()
            assert data["ok"] is True
            assert data["width"] == 3
            assert data["height"] == 3
            assert data["cellCount"] == 2

            # Decompress and verify mask data
            mask_bytes = zlib.decompress(bytes.fromhex(data["maskHex"]))
            mask_arr = np.frombuffer(mask_bytes, dtype=np.int32).reshape(
                data["height"], data["width"]
            )
            assert mask_arr.shape == (3, 3)
            assert int(mask_arr.max()) == 2

    def test_masks_not_available_returns_404(self, client):
        resp = client.get("/api/detect/preview/masks?jobId=nonexistent")
        assert resp.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest project/tests/unit/test_mask_endpoints.py::TestMasksEndpoint::test_masks_not_available_returns_404 -v`
Expected: FAIL — endpoint does not exist yet.

- [ ] **Step 3: Add mask storage and endpoint**

In `project/frontend/blueprints/api_detect_preview.py`:

1. Add `import zlib` at the top (after `import threading`).

2. Add a new dict to cache masks (after `_detect_results` on line ~28):

```python
_detect_masks: dict[str, np.ndarray] = {}
```

3. Add a new helper function (after `_run_detection`):

```python
def _run_detection_with_masks(slice_path: Path, cfg: dict):
    """Run detection and return (DataFrame, masks) if Cellpose, else (DataFrame, None)."""
    try:
        from project.scripts.detect import detect_cells_cellpose, _resolve_model_type, _diameter_px, _use_gpu
    except ImportError:
        from scripts.detect import detect_cells_cellpose, _resolve_model_type, _diameter_px, _use_gpu

    det_cfg = cfg.get("detection", {})
    model_type = _resolve_model_type(str(det_cfg.get("primary_model", "cpsam")))
    d_px = _diameter_px(det_cfg, cfg)
    use_gpu = _use_gpu(cfg, det_cfg)
    flow_thr = float(det_cfg.get("cellpose_flow_threshold", 0.4))
    prob_thr = float(det_cfg.get("cellpose_cellprob_threshold", 0.0))
    min_sz = int(det_cfg.get("cellpose_min_size_px", 8))

    try:
        result = detect_cells_cellpose(
            slice_path, model_type, d_px,
            use_gpu=use_gpu, flow_threshold=flow_thr,
            cellprob_threshold=prob_thr, min_size=min_sz,
            return_masks=True,
        )
        if isinstance(result, tuple):
            return result  # (df, masks)
        return result, None
    except Exception:
        # Fallback to standard detection (no masks)
        df = _run_detection(slice_path, cfg)
        return df, None
```

4. In the `detect_preview()` function, after `cfg = _apply_param_overrides(cfg, params)`, add logic to use mask-returning detection when requested:

```python
    return_masks = bool(payload.get("returnMasks", False))
```

Then replace the try/except block for `cells_df = _run_detection(...)` with:

```python
    try:
        if return_masks:
            cells_df, raw_masks = _run_detection_with_masks(slice_path, cfg)
            if raw_masks is not None:
                with _detect_lock:
                    _detect_masks[job_id] = raw_masks
        else:
            cells_df = _run_detection(slice_path, cfg)
    except Exception as exc:
        return jsonify({"ok": False, "error": str(exc), "runtimeAvailable": False}), 500
```

5. Add the masks endpoint:

```python
@bp.get("/detect/preview/masks")
def detect_preview_masks():
    """Return the raw instance mask array from the last detection.

    The mask is zlib-compressed and hex-encoded for JSON transport.
    Client decompresses and reshapes to (height, width) Int32 array.
    """
    job_id = ctx._query_job_id()
    with _detect_lock:
        masks = _detect_masks.get(job_id)
    if masks is None:
        return jsonify({"ok": False, "error": "no masks available"}), 404

    compressed = zlib.compress(masks.astype(np.int32).tobytes())
    return jsonify({
        "ok": True,
        "width": masks.shape[1],
        "height": masks.shape[0],
        "cellCount": int(masks.max()),
        "maskHex": compressed.hex(),
    })
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest project/tests/unit/test_mask_endpoints.py -v`
Expected: 2 passed.

- [ ] **Step 5: Run full suite**

Run: `python -m pytest project/tests -q`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add project/frontend/blueprints/api_detect_preview.py project/tests/unit/test_mask_endpoints.py
git commit -m "feat: /api/detect/preview/masks returns raw instance masks for editing"
```

---

### Task 3: `/api/cellpose/save-training-sample` endpoint

**Files:**
- Modify: `project/frontend/blueprints/api_cellpose.py`
- Modify: `project/tests/unit/test_mask_endpoints.py`

- [ ] **Step 1: Write the failing test**

Append to `project/tests/unit/test_mask_endpoints.py`:

```python
class TestSaveTrainingSample:
    def test_saves_image_and_mask_files(self, app, client, tmp_path, sample_slice):
        """Saves image + mask in Cellpose convention."""
        import project.frontend.server_context as ctx
        from tifffile import imread

        training_dir = ctx.PROJECT_ROOT / "cellpose_training"

        mask_data = np.array([[0, 1], [2, 0]], dtype=np.int32)
        compressed = zlib.compress(mask_data.tobytes())

        resp = client.post(
            "/api/cellpose/save-training-sample",
            data=json.dumps({
                "imagePath": str(sample_slice),
                "maskHex": compressed.hex(),
                "width": 2,
                "height": 2,
            }),
            content_type="application/json",
        )
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["ok"] is True
        assert data["trainingSetStats"]["totalImages"] >= 1

        # Verify files exist with Cellpose naming convention
        saved_name = Path(data["savedAs"]).stem  # e.g. "test_slice"
        img_path = training_dir / f"{saved_name}.tif"
        mask_path = training_dir / f"{saved_name}_masks.tif"
        assert img_path.exists()
        assert mask_path.exists()

        # Verify mask content
        saved_mask = imread(str(mask_path))
        assert saved_mask.shape == (2, 2)
        assert int(saved_mask.max()) == 2

    def test_rejects_missing_image(self, client):
        resp = client.post(
            "/api/cellpose/save-training-sample",
            data=json.dumps({
                "imagePath": "/nonexistent/path.tif",
                "maskHex": "deadbeef",
                "width": 2,
                "height": 2,
            }),
            content_type="application/json",
        )
        assert resp.status_code == 400

    def test_training_stats_accumulate(self, app, client, tmp_path):
        """Stats count increases as more samples are saved."""
        import project.frontend.server_context as ctx

        for i in range(3):
            slice_path = tmp_path / f"slice_{i}.tif"
            img = np.random.randint(0, 65535, (8, 8), dtype=np.uint16)
            imwrite(str(slice_path), img)

            mask_data = np.ones((8, 8), dtype=np.int32) * (i + 1)
            compressed = zlib.compress(mask_data.tobytes())

            resp = client.post(
                "/api/cellpose/save-training-sample",
                data=json.dumps({
                    "imagePath": str(slice_path),
                    "maskHex": compressed.hex(),
                    "width": 8,
                    "height": 8,
                }),
                content_type="application/json",
            )
            assert resp.status_code == 200

        data = resp.get_json()
        assert data["trainingSetStats"]["totalImages"] == 3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest project/tests/unit/test_mask_endpoints.py::TestSaveTrainingSample -v`
Expected: 404 — endpoint does not exist yet.

- [ ] **Step 3: Implement the endpoint**

In `project/frontend/blueprints/api_cellpose.py`, add imports and the endpoint:

```python
"""api_cellpose.py — Cellpose model management and training endpoints.

Provides endpoints for listing available models, parameter validation,
and (in future subsystems) training and model comparison.
"""

from __future__ import annotations

import shutil
import zlib
from pathlib import Path

import numpy as np
from flask import Blueprint, jsonify, request
from tifffile import imread, imwrite

import project.frontend.server_context as ctx

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


def _training_dir() -> Path:
    """Return the Cellpose training data directory, creating it if needed."""
    d = ctx.PROJECT_ROOT / "cellpose_training"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _training_set_stats(training_dir: Path) -> dict:
    """Compute stats for the current training set."""
    mask_files = sorted(training_dir.glob("*_masks.tif"))
    total_images = len(mask_files)
    total_cells = 0
    for mf in mask_files:
        try:
            m = imread(str(mf))
            total_cells += int(m.max())
        except Exception:
            pass
    return {
        "totalImages": total_images,
        "totalCells": total_cells,
        "avgCellsPerImage": round(total_cells / max(total_images, 1), 1),
    }


@bp.post("/save-training-sample")
def save_training_sample():
    """Save a corrected mask + original image as a Cellpose training pair.

    Request JSON:
        imagePath: str — path to the original TIFF slice
        maskHex: str — zlib-compressed, hex-encoded Int32 mask array
        width: int — mask width
        height: int — mask height

    Saves:
        cellpose_training/{name}.tif — copy of original image
        cellpose_training/{name}_masks.tif — corrected instance mask (uint16)
    """
    payload = request.get_json(force=True)
    image_path = Path(payload.get("imagePath", ""))
    if not image_path.is_absolute():
        image_path = ctx.PROJECT_ROOT / image_path
    if not image_path.exists():
        return jsonify({"ok": False, "error": f"image not found: {image_path}"}), 400

    try:
        mask_bytes = zlib.decompress(bytes.fromhex(payload["maskHex"]))
        width = int(payload["width"])
        height = int(payload["height"])
        mask = np.frombuffer(mask_bytes, dtype=np.int32).reshape(height, width)
    except Exception as exc:
        return jsonify({"ok": False, "error": f"invalid mask data: {exc}"}), 400

    td = _training_dir()
    stem = image_path.stem

    # Copy original image
    dst_img = td / f"{stem}.tif"
    if not dst_img.exists() or dst_img.resolve() != image_path.resolve():
        shutil.copy2(str(image_path), str(dst_img))

    # Save mask as uint16 TIFF (Cellpose convention)
    dst_mask = td / f"{stem}_masks.tif"
    imwrite(str(dst_mask), mask.astype(np.uint16))

    stats = _training_set_stats(td)

    return jsonify({
        "ok": True,
        "savedAs": f"cellpose_training/{stem}.tif",
        "trainingSetStats": stats,
    })
```

- [ ] **Step 4: Run tests**

Run: `python -m pytest project/tests/unit/test_mask_endpoints.py -v`
Expected: 5 passed (2 from Task 2 + 3 new).

- [ ] **Step 5: Run full suite**

Run: `python -m pytest project/tests -q`

- [ ] **Step 6: Commit**

```bash
git add project/frontend/blueprints/api_cellpose.py project/tests/unit/test_mask_endpoints.py
git commit -m "feat: /api/cellpose/save-training-sample saves corrected masks for training"
```

---

### Task 4: Mask editor HTML modal shell

**Files:**
- Create: `project/frontend/mask-editor.css`
- Modify: `project/frontend/index.html`

- [ ] **Step 1: Create mask editor CSS**

Create `project/frontend/mask-editor.css`:

```css
/* Mask Editor Modal */
.mask-editor-modal {
  display: none;
  position: fixed;
  top: 0; left: 0; right: 0; bottom: 0;
  z-index: 9999;
  background: var(--bg, #0d1117);
}
.mask-editor-modal.active { display: flex; }

.mask-editor-layout {
  display: flex;
  width: 100%;
  height: 100%;
}

/* Left toolbar */
.me-toolbar {
  width: 52px;
  background: var(--bg-secondary, #1a1a2e);
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  padding: 10px 4px;
  border-right: 1px solid var(--border, #333);
}
.me-tool-btn {
  width: 38px; height: 38px;
  background: var(--bg-tertiary, #333);
  border: 1px solid transparent;
  border-radius: 6px;
  display: flex; align-items: center; justify-content: center;
  color: var(--text-secondary, #aaa);
  font-size: 16px;
  cursor: pointer;
  transition: background 0.15s;
}
.me-tool-btn:hover { background: var(--bg-hover, #444); }
.me-tool-btn.active {
  background: var(--accent, #e94560);
  color: #fff;
  border-color: var(--accent, #e94560);
}
.me-tool-sep {
  width: 28px; height: 1px;
  background: var(--border, #555);
  margin: 2px 0;
}

/* Center canvas area */
.me-canvas-area {
  flex: 1;
  background: #111;
  position: relative;
  overflow: hidden;
  display: flex;
  align-items: center;
  justify-content: center;
}
.me-canvas-wrap {
  position: relative;
  transform-origin: 0 0;
}
.me-canvas-wrap canvas {
  position: absolute;
  top: 0; left: 0;
  image-rendering: pixelated;
}

/* Right panel */
.me-right-panel {
  width: 200px;
  background: var(--bg-secondary, #16213e);
  padding: 12px;
  border-left: 1px solid var(--border, #333);
  font-size: 13px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 12px;
}
.me-panel-section { display: flex; flex-direction: column; gap: 6px; }
.me-panel-label {
  color: var(--accent, #e94560);
  font-weight: 600;
  font-size: 12px;
  text-transform: uppercase;
}
.me-slider-row {
  display: flex; align-items: center; gap: 8px;
}
.me-slider-row input[type="range"] { flex: 1; }
.me-slider-row .me-slider-val {
  min-width: 28px; text-align: right;
  color: var(--text-secondary, #888);
  font-size: 12px;
}

/* Cell list */
.me-cell-list {
  flex: 1; overflow-y: auto;
  display: flex; flex-direction: column; gap: 3px;
}
.me-cell-item {
  padding: 4px 8px;
  border-radius: 4px;
  font-size: 12px;
  cursor: pointer;
  display: flex; align-items: center; gap: 6px;
  background: var(--bg-tertiary, #1a1a2e);
}
.me-cell-item:hover { background: var(--bg-hover, #222); }
.me-cell-item.selected {
  border: 1px solid currentColor;
  font-weight: 600;
}
.me-cell-dot {
  width: 10px; height: 10px;
  border-radius: 50%;
  display: inline-block;
}

/* Status bar */
.me-status-bar {
  position: absolute;
  bottom: 0; left: 0; right: 0;
  background: var(--bg-secondary, #1a1a2e);
  padding: 4px 12px;
  font-size: 11px;
  color: var(--text-secondary, #888);
  display: flex; justify-content: space-between;
  border-top: 1px solid var(--border, #333);
}

/* Action buttons */
.me-actions {
  display: flex; gap: 8px;
  padding-top: 8px;
  border-top: 1px solid var(--border, #333);
}
.me-btn {
  flex: 1;
  padding: 8px;
  border: none;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 600;
  cursor: pointer;
  text-align: center;
}
.me-btn-primary {
  background: var(--accent, #e94560);
  color: #fff;
}
.me-btn-primary:hover { opacity: 0.9; }
.me-btn-secondary {
  background: var(--bg-tertiary, #333);
  color: var(--text, #eee);
}
.me-btn-secondary:hover { background: var(--bg-hover, #444); }

/* Top bar with close button */
.me-top-bar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 6px 12px;
  background: var(--bg-secondary, #1a1a2e);
  border-bottom: 1px solid var(--border, #333);
}
.me-top-bar h3 { margin: 0; font-size: 14px; color: var(--text); }
.me-close-btn {
  background: none; border: none;
  color: var(--text-secondary, #888);
  font-size: 20px; cursor: pointer;
  padding: 4px 8px;
}
.me-close-btn:hover { color: var(--text); }
```

- [ ] **Step 2: Add modal HTML to index.html**

In `project/frontend/index.html`, add the CSS link in the `<head>` section (after the existing `styles.css` link):

```html
<link rel="stylesheet" href="mask-editor.css">
```

Add the modal HTML before the closing `</body>` tag, and the script tag:

```html
  <!-- Mask Editor Modal -->
  <div id="maskEditorModal" class="mask-editor-modal">
    <div style="display:flex;flex-direction:column;width:100%;height:100%;">
      <div class="me-top-bar">
        <h3 data-i18n="maskEditor.title">Mask Editor</h3>
        <button id="maskEditorClose" class="me-close-btn" title="Close">&times;</button>
      </div>
      <div class="mask-editor-layout">
        <!-- Left toolbar -->
        <div class="me-toolbar">
          <button class="me-tool-btn active" data-me-tool="brush" title="Brush (B)">🖌</button>
          <button class="me-tool-btn" data-me-tool="eraser" title="Eraser (E)">🧹</button>
          <button class="me-tool-btn" data-me-tool="addcell" title="New Cell (N)">➕</button>
          <button class="me-tool-btn" data-me-tool="delete" title="Delete Cell (D)">🗑</button>
          <button class="me-tool-btn" data-me-tool="merge" title="Merge Cells">🔗</button>
          <button class="me-tool-btn" data-me-tool="split" title="Split Cell">✂️</button>
          <div class="me-tool-sep"></div>
          <button class="me-tool-btn" data-me-tool="pan" title="Pan (Space+drag)">✋</button>
        </div>
        <!-- Canvas area -->
        <div class="me-canvas-area" id="meCanvasArea">
          <div class="me-canvas-wrap" id="meCanvasWrap">
            <canvas id="meImageCanvas"></canvas>
            <canvas id="meMaskCanvas"></canvas>
          </div>
          <div class="me-status-bar">
            <span id="meStatusLeft">Brush: 10px</span>
            <span id="meStatusRight">Zoom: 100% | 0 cells</span>
          </div>
        </div>
        <!-- Right panel -->
        <div class="me-right-panel">
          <div class="me-panel-section">
            <span class="me-panel-label" data-i18n="maskEditor.brushSize">Brush Size</span>
            <div class="me-slider-row">
              <input type="range" id="meBrushSize" min="1" max="50" value="10" />
              <span class="me-slider-val" id="meBrushSizeVal">10</span>
            </div>
          </div>
          <div class="me-panel-section">
            <span class="me-panel-label" data-i18n="maskEditor.opacity">Mask Opacity</span>
            <div class="me-slider-row">
              <input type="range" id="meMaskOpacity" min="0" max="100" value="50" />
              <span class="me-slider-val" id="meMaskOpacityVal">50%</span>
            </div>
          </div>
          <div class="me-panel-section">
            <span class="me-panel-label" data-i18n="maskEditor.cells">Cells</span>
            <div class="me-cell-list" id="meCellList">
              <!-- Populated dynamically -->
            </div>
          </div>
          <div class="me-actions">
            <button id="meSaveTrainingBtn" class="me-btn me-btn-primary" data-i18n="maskEditor.save">Save to Training Set</button>
          </div>
          <div class="me-actions">
            <button id="meUndoBtn" class="me-btn me-btn-secondary">Undo</button>
            <button id="meRedoBtn" class="me-btn me-btn-secondary">Redo</button>
          </div>
        </div>
      </div>
    </div>
  </div>
  <script src="mask-editor.js"></script>
```

- [ ] **Step 3: Add "Edit Masks" button to detection result banner**

In `project/frontend/index.html`, find the detect actions div (around line 514-517):

```html
              <div class="detect-actions">
                <a id="detectCsvLink" class="btn-secondary btn-sm" download="detect_preview.csv">⬇ Download CSV</a>
                <button id="detectToggleOverlay" class="btn-secondary btn-sm">Toggle Overlay</button>
              </div>
```

Add the Edit Masks button:

```html
              <div class="detect-actions">
                <a id="detectCsvLink" class="btn-secondary btn-sm" download="detect_preview.csv">⬇ Download CSV</a>
                <button id="detectToggleOverlay" class="btn-secondary btn-sm">Toggle Overlay</button>
                <button id="editMasksBtn" class="btn-primary btn-sm" data-i18n="btn.editMasks">✏️ Edit Masks</button>
              </div>
```

- [ ] **Step 4: Commit**

```bash
git add project/frontend/mask-editor.css project/frontend/index.html
git commit -m "feat: mask editor modal HTML shell with toolbar, canvas, right panel"
```

---

### Task 5: Mask editor core JS — canvas rendering + brush/eraser tools

**Files:**
- Create: `project/frontend/mask-editor.js`
- Modify: `project/frontend/app.js` (minimal: "Edit Masks" click handler + i18n)

This is the largest task. The JS file handles:
- Loading image + masks from the backend
- Two-layer canvas rendering (image underneath, colored mask overlay on top)
- Brush and eraser tools
- Cell selection, color LUT
- Undo/redo stack
- Open/close modal
- Save to training set

- [ ] **Step 1: Create mask-editor.js**

Create `project/frontend/mask-editor.js`:

```javascript
/**
 * mask-editor.js — Cellpose mask annotation canvas
 *
 * Two-layer HTML5 Canvas editor for correcting cell instance masks.
 * Tools: brush, eraser, add cell, delete, merge, split, pan.
 * Saves corrected masks to the backend for Cellpose training.
 */

/* global showToast, t, getOverlayJobId */

const MaskEditor = (() => {
  'use strict';

  // --- State ---
  let state = {
    imageData: null,       // Uint8Array grayscale (display-resolution)
    maskData: null,        // Int32Array instance mask
    width: 0,
    height: 0,
    nextCellId: 1,
    selectedCell: 0,
    currentTool: 'brush',
    brushSize: 10,
    maskOpacity: 0.5,
    zoom: 1.0,
    panX: 0, panY: 0,
    isPanning: false,
    isDrawing: false,
    lastX: -1, lastY: -1,
    imagePath: '',
    undoStack: [],
    redoStack: [],
    maxUndo: 20,
    mergeFirst: null,       // first cell ID for merge operation
  };

  // Color LUT: cell ID → [r,g,b]
  const COLOR_LUT = [];
  function ensureColor(id) {
    while (COLOR_LUT.length <= id) {
      // Generate distinct colors using golden-ratio hue spacing
      const hue = (COLOR_LUT.length * 137.508) % 360;
      const s = 0.7, l = 0.55;
      const c = (1 - Math.abs(2*l - 1)) * s;
      const x = c * (1 - Math.abs((hue/60) % 2 - 1));
      const m = l - c/2;
      let r, g, b;
      if (hue < 60)       { r=c; g=x; b=0; }
      else if (hue < 120) { r=x; g=c; b=0; }
      else if (hue < 180) { r=0; g=c; b=x; }
      else if (hue < 240) { r=0; g=x; b=c; }
      else if (hue < 300) { r=x; g=0; b=c; }
      else                { r=c; g=0; b=x; }
      COLOR_LUT.push([
        Math.round((r+m)*255),
        Math.round((g+m)*255),
        Math.round((b+m)*255),
      ]);
    }
    return COLOR_LUT[id];
  }

  // --- DOM refs (resolved on first open) ---
  let dom = {};
  function resolveDom() {
    dom.modal = document.getElementById('maskEditorModal');
    dom.imageCanvas = document.getElementById('meImageCanvas');
    dom.maskCanvas = document.getElementById('meMaskCanvas');
    dom.canvasWrap = document.getElementById('meCanvasWrap');
    dom.canvasArea = document.getElementById('meCanvasArea');
    dom.brushSize = document.getElementById('meBrushSize');
    dom.brushSizeVal = document.getElementById('meBrushSizeVal');
    dom.maskOpacity = document.getElementById('meMaskOpacity');
    dom.maskOpacityVal = document.getElementById('meMaskOpacityVal');
    dom.cellList = document.getElementById('meCellList');
    dom.statusLeft = document.getElementById('meStatusLeft');
    dom.statusRight = document.getElementById('meStatusRight');
    dom.undoBtn = document.getElementById('meUndoBtn');
    dom.redoBtn = document.getElementById('meRedoBtn');
    dom.saveBtn = document.getElementById('meSaveTrainingBtn');
    dom.closeBtn = document.getElementById('maskEditorClose');
  }

  // --- Canvas Rendering ---
  function renderImage() {
    if (!state.imageData) return;
    const ctx = dom.imageCanvas.getContext('2d');
    const imgData = ctx.createImageData(state.width, state.height);
    for (let i = 0; i < state.imageData.length; i++) {
      const v = state.imageData[i];
      imgData.data[i*4] = v;
      imgData.data[i*4+1] = v;
      imgData.data[i*4+2] = v;
      imgData.data[i*4+3] = 255;
    }
    ctx.putImageData(imgData, 0, 0);
  }

  function renderMask() {
    if (!state.maskData) return;
    const ctx = dom.maskCanvas.getContext('2d');
    const imgData = ctx.createImageData(state.width, state.height);
    const alpha = Math.round(state.maskOpacity * 255);
    for (let i = 0; i < state.maskData.length; i++) {
      const cellId = state.maskData[i];
      if (cellId <= 0) {
        imgData.data[i*4+3] = 0; // transparent background
      } else {
        const c = ensureColor(cellId);
        imgData.data[i*4] = c[0];
        imgData.data[i*4+1] = c[1];
        imgData.data[i*4+2] = c[2];
        imgData.data[i*4+3] = (cellId === state.selectedCell) ? Math.min(alpha + 60, 255) : alpha;
      }
    }
    ctx.putImageData(imgData, 0, 0);
  }

  function updateTransform() {
    dom.canvasWrap.style.transform =
      `translate(${state.panX}px, ${state.panY}px) scale(${state.zoom})`;
  }

  function render() {
    renderMask();
    updateStatus();
    updateCellList();
  }

  // --- Undo/Redo ---
  function pushUndo() {
    // Store sparse diff for memory efficiency
    if (state.undoStack.length >= state.maxUndo) {
      state.undoStack.shift();
    }
    state.undoStack.push(new Int32Array(state.maskData));
    state.redoStack = [];
  }

  function undo() {
    if (state.undoStack.length === 0) return;
    state.redoStack.push(new Int32Array(state.maskData));
    state.maskData = state.undoStack.pop();
    state.nextCellId = Math.max(1, maxCellId() + 1);
    render();
  }

  function redo() {
    if (state.redoStack.length === 0) return;
    state.undoStack.push(new Int32Array(state.maskData));
    state.maskData = state.redoStack.pop();
    state.nextCellId = Math.max(1, maxCellId() + 1);
    render();
  }

  function maxCellId() {
    let max = 0;
    for (let i = 0; i < state.maskData.length; i++) {
      if (state.maskData[i] > max) max = state.maskData[i];
    }
    return max;
  }

  // --- Tools ---
  function paintCircle(cx, cy, radius, value) {
    const r2 = radius * radius;
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        if (dx*dx + dy*dy > r2) continue;
        const px = cx + dx, py = cy + dy;
        if (px < 0 || px >= state.width || py < 0 || py >= state.height) continue;
        state.maskData[py * state.width + px] = value;
      }
    }
  }

  function paintLine(x0, y0, x1, y1, radius, value) {
    // Bresenham-style interpolation for smooth strokes
    const dx = Math.abs(x1-x0), dy = Math.abs(y1-y0);
    const steps = Math.max(dx, dy, 1);
    for (let i = 0; i <= steps; i++) {
      const t = i / steps;
      const px = Math.round(x0 + (x1-x0)*t);
      const py = Math.round(y0 + (y1-y0)*t);
      paintCircle(px, py, radius, value);
    }
  }

  function deleteCellById(cellId) {
    pushUndo();
    for (let i = 0; i < state.maskData.length; i++) {
      if (state.maskData[i] === cellId) state.maskData[i] = 0;
    }
    if (state.selectedCell === cellId) state.selectedCell = 0;
    render();
  }

  function mergeCells(idA, idB) {
    if (idA === idB) return;
    pushUndo();
    const target = Math.min(idA, idB);
    const source = Math.max(idA, idB);
    for (let i = 0; i < state.maskData.length; i++) {
      if (state.maskData[i] === source) state.maskData[i] = target;
    }
    state.selectedCell = target;
    render();
  }

  function splitCellAtLine(x0, y0, x1, y1) {
    // Find the cell under the line start
    const idx = y0 * state.width + x0;
    const cellId = state.maskData[idx];
    if (cellId <= 0) return;

    pushUndo();

    // Erase the line through the cell
    paintLine(x0, y0, x1, y1, 1, 0);

    // Flood-fill connected components from the original cell
    const visited = new Uint8Array(state.maskData.length);
    const newId = state.nextCellId++;
    let foundFirst = false;

    for (let i = 0; i < state.maskData.length; i++) {
      if (state.maskData[i] !== cellId || visited[i]) continue;

      // BFS flood fill
      const queue = [i];
      visited[i] = 1;
      const component = [];
      while (queue.length > 0) {
        const ci = queue.shift();
        component.push(ci);
        const cx = ci % state.width, cy = Math.floor(ci / state.width);
        for (const [dx, dy] of [[-1,0],[1,0],[0,-1],[0,1]]) {
          const nx = cx+dx, ny = cy+dy;
          if (nx<0 || nx>=state.width || ny<0 || ny>=state.height) continue;
          const ni = ny*state.width + nx;
          if (!visited[ni] && state.maskData[ni] === cellId) {
            visited[ni] = 1;
            queue.push(ni);
          }
        }
      }

      if (!foundFirst) {
        foundFirst = true; // keep first component as original ID
      } else {
        // Relabel subsequent components
        for (const pi of component) {
          state.maskData[pi] = newId;
        }
      }
    }
    render();
  }

  // --- Mouse Handlers ---
  function canvasCoords(e) {
    const rect = dom.maskCanvas.getBoundingClientRect();
    const x = Math.floor((e.clientX - rect.left) / state.zoom);
    const y = Math.floor((e.clientY - rect.top) / state.zoom);
    return { x, y };
  }

  function onMouseDown(e) {
    if (e.button === 1 || (e.button === 0 && state.currentTool === 'pan') || e.shiftKey) {
      state.isPanning = true;
      state.lastX = e.clientX;
      state.lastY = e.clientY;
      return;
    }

    const { x, y } = canvasCoords(e);

    if (state.currentTool === 'delete') {
      const cellId = state.maskData[y * state.width + x];
      if (cellId > 0) deleteCellById(cellId);
      return;
    }

    if (state.currentTool === 'merge') {
      const cellId = state.maskData[y * state.width + x];
      if (cellId <= 0) return;
      if (state.mergeFirst === null) {
        state.mergeFirst = cellId;
        state.selectedCell = cellId;
        render();
        updateStatus();
        return;
      } else {
        mergeCells(state.mergeFirst, cellId);
        state.mergeFirst = null;
        return;
      }
    }

    if (state.currentTool === 'split') {
      state.isDrawing = true;
      state.lastX = x;
      state.lastY = y;
      return;
    }

    if (state.currentTool === 'brush' || state.currentTool === 'eraser' || state.currentTool === 'addcell') {
      state.isDrawing = true;
      pushUndo();

      if (state.currentTool === 'addcell') {
        state.selectedCell = state.nextCellId++;
      }

      const value = state.currentTool === 'eraser' ? 0 : state.selectedCell;
      if (value <= 0 && state.currentTool !== 'eraser') {
        // Select cell under cursor if none selected
        const cellId = state.maskData[y * state.width + x];
        if (cellId > 0) {
          state.selectedCell = cellId;
        } else {
          state.selectedCell = state.nextCellId++;
        }
      }

      const paintVal = state.currentTool === 'eraser' ? 0 : state.selectedCell;
      paintCircle(x, y, state.brushSize, paintVal);
      state.lastX = x;
      state.lastY = y;
      renderMask();
    }
  }

  function onMouseMove(e) {
    if (state.isPanning) {
      state.panX += e.clientX - state.lastX;
      state.panY += e.clientY - state.lastY;
      state.lastX = e.clientX;
      state.lastY = e.clientY;
      updateTransform();
      return;
    }

    if (!state.isDrawing) return;
    const { x, y } = canvasCoords(e);

    if (state.currentTool === 'split') {
      // Preview line (could add visual feedback later)
      state.lastX = state.lastX; // keep start point
      return;
    }

    const value = state.currentTool === 'eraser' ? 0 : state.selectedCell;
    paintLine(state.lastX, state.lastY, x, y, state.brushSize, value);
    state.lastX = x;
    state.lastY = y;
    renderMask();
  }

  function onMouseUp(e) {
    if (state.isPanning) {
      state.isPanning = false;
      return;
    }

    if (state.isDrawing && state.currentTool === 'split') {
      const { x, y } = canvasCoords(e);
      splitCellAtLine(state.lastX, state.lastY, x, y);
    }

    state.isDrawing = false;
    if (state.currentTool !== 'split') {
      updateCellList();
      updateStatus();
    }
  }

  function onWheel(e) {
    e.preventDefault();
    const delta = e.deltaY > 0 ? 0.9 : 1.1;
    const newZoom = Math.max(0.1, Math.min(10, state.zoom * delta));

    // Zoom towards cursor
    const rect = dom.canvasArea.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    state.panX = cx - (cx - state.panX) * (newZoom / state.zoom);
    state.panY = cy - (cy - state.panY) * (newZoom / state.zoom);
    state.zoom = newZoom;
    updateTransform();
    updateStatus();
  }

  // --- UI Updates ---
  function updateStatus() {
    const toolName = state.currentTool.charAt(0).toUpperCase() + state.currentTool.slice(1);
    let left = `${toolName}: ${state.brushSize}px`;
    if (state.selectedCell > 0) left += ` | Cell #${state.selectedCell}`;
    if (state.currentTool === 'merge' && state.mergeFirst !== null) {
      left += ` | Click second cell to merge with #${state.mergeFirst}`;
    }
    dom.statusLeft.textContent = left;
    dom.statusRight.textContent = `Zoom: ${Math.round(state.zoom * 100)}% | ${maxCellId()} cells`;
  }

  function updateCellList() {
    const cellIds = new Set();
    for (let i = 0; i < state.maskData.length; i++) {
      if (state.maskData[i] > 0) cellIds.add(state.maskData[i]);
    }
    dom.cellList.innerHTML = '';
    const sorted = Array.from(cellIds).sort((a,b) => a-b);
    for (const id of sorted) {
      const c = ensureColor(id);
      const div = document.createElement('div');
      div.className = 'me-cell-item' + (id === state.selectedCell ? ' selected' : '');
      div.innerHTML = `<span class="me-cell-dot" style="background:rgb(${c[0]},${c[1]},${c[2]})"></span> Cell ${id}`;
      div.onclick = () => { state.selectedCell = id; render(); };
      dom.cellList.appendChild(div);
    }
  }

  // --- Keyboard Shortcuts ---
  function onKeyDown(e) {
    if (!dom.modal.classList.contains('active')) return;

    const key = e.key.toLowerCase();
    if (key === 'b') setTool('brush');
    else if (key === 'e') setTool('eraser');
    else if (key === 'n') setTool('addcell');
    else if (key === 'd') setTool('delete');
    else if (key === '[') {
      state.brushSize = Math.max(1, state.brushSize - 1);
      dom.brushSize.value = state.brushSize;
      dom.brushSizeVal.textContent = state.brushSize;
      updateStatus();
    }
    else if (key === ']') {
      state.brushSize = Math.min(50, state.brushSize + 1);
      dom.brushSize.value = state.brushSize;
      dom.brushSizeVal.textContent = state.brushSize;
      updateStatus();
    }
    else if (key === 'z' && (e.ctrlKey || e.metaKey)) {
      e.preventDefault();
      if (e.shiftKey) redo();
      else undo();
    }
    else if (key === 'escape') close();
  }

  function setTool(tool) {
    state.currentTool = tool;
    state.mergeFirst = null;
    document.querySelectorAll('.me-tool-btn').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.meTool === tool);
    });
    updateStatus();
  }

  // --- Save to Training Set ---
  async function saveToTrainingSet() {
    if (!state.maskData || !state.imagePath) return;

    const compressed = pako.deflate(new Uint8Array(state.maskData.buffer));
    const hexStr = Array.from(compressed, b => b.toString(16).padStart(2, '0')).join('');

    try {
      const res = await fetch('/api/cellpose/save-training-sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          imagePath: state.imagePath,
          maskHex: hexStr,
          width: state.width,
          height: state.height,
        }),
      });
      const data = await res.json();
      if (data.ok) {
        if (typeof showToast === 'function') {
          showToast(`Saved! Training set: ${data.trainingSetStats.totalImages} images, ${data.trainingSetStats.totalCells} cells`, 'success', 4000);
        }
      } else {
        if (typeof showToast === 'function') showToast(`Save failed: ${data.error}`, 'error');
      }
    } catch (err) {
      if (typeof showToast === 'function') showToast(`Save failed: ${err.message}`, 'error');
    }
  }

  // --- Open/Close ---
  async function open(imagePath, jobId) {
    resolveDom();
    state.imagePath = imagePath;
    state.undoStack = [];
    state.redoStack = [];
    state.mergeFirst = null;
    state.zoom = 1.0;
    state.panX = 0;
    state.panY = 0;
    state.selectedCell = 1;

    // Fetch masks from backend
    try {
      const res = await fetch(`/api/detect/preview/masks?jobId=${encodeURIComponent(jobId)}`);
      const data = await res.json();
      if (!data.ok) {
        if (typeof showToast === 'function') showToast('No masks available. Run detection with returnMasks first.', 'warning');
        return;
      }

      state.width = data.width;
      state.height = data.height;
      state.nextCellId = data.cellCount + 1;

      // Decompress mask
      const hexBytes = new Uint8Array(data.maskHex.match(/.{1,2}/g).map(b => parseInt(b, 16)));
      const decompressed = pako.inflate(hexBytes);
      state.maskData = new Int32Array(decompressed.buffer);

    } catch (err) {
      if (typeof showToast === 'function') showToast(`Failed to load masks: ${err.message}`, 'error');
      return;
    }

    // Fetch original image as grayscale
    try {
      const imgRes = await fetch(`/api/detect/preview/overlay?jobId=${encodeURIComponent(jobId)}`);
      const blob = await imgRes.blob();
      const bmp = await createImageBitmap(blob);

      // Draw to offscreen canvas to get pixel data
      const offscreen = document.createElement('canvas');
      offscreen.width = bmp.width;
      offscreen.height = bmp.height;
      const offCtx = offscreen.getContext('2d');
      offCtx.drawImage(bmp, 0, 0);
      const pixels = offCtx.getImageData(0, 0, bmp.width, bmp.height);

      // Extract grayscale (use red channel)
      state.imageData = new Uint8Array(state.width * state.height);
      // If overlay image is different size from mask, we use mask dimensions
      const srcW = bmp.width, srcH = bmp.height;
      for (let y = 0; y < state.height; y++) {
        for (let x = 0; x < state.width; x++) {
          const sx = Math.floor(x * srcW / state.width);
          const sy = Math.floor(y * srcH / state.height);
          state.imageData[y * state.width + x] = pixels.data[(sy * srcW + sx) * 4];
        }
      }
    } catch (err) {
      // If overlay not available, use black background
      state.imageData = new Uint8Array(state.width * state.height);
    }

    // Setup canvases
    dom.imageCanvas.width = state.width;
    dom.imageCanvas.height = state.height;
    dom.maskCanvas.width = state.width;
    dom.maskCanvas.height = state.height;
    dom.canvasWrap.style.width = state.width + 'px';
    dom.canvasWrap.style.height = state.height + 'px';

    // Center in viewport
    const area = dom.canvasArea.getBoundingClientRect();
    const fitZoom = Math.min(area.width / state.width, (area.height - 30) / state.height) * 0.9;
    state.zoom = fitZoom;
    state.panX = (area.width - state.width * fitZoom) / 2;
    state.panY = (area.height - state.height * fitZoom) / 2;

    renderImage();
    render();
    updateTransform();
    setTool('brush');

    // Show modal
    dom.modal.classList.add('active');

    // Bind events
    dom.maskCanvas.addEventListener('mousedown', onMouseDown);
    dom.maskCanvas.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
    dom.canvasArea.addEventListener('wheel', onWheel, { passive: false });
  }

  function close() {
    dom.modal.classList.remove('active');
    dom.maskCanvas.removeEventListener('mousedown', onMouseDown);
    dom.maskCanvas.removeEventListener('mousemove', onMouseMove);
    window.removeEventListener('mouseup', onMouseUp);
    dom.canvasArea.removeEventListener('wheel', onWheel);
    state.maskData = null;
    state.imageData = null;
  }

  // --- Event Bindings (deferred until DOM ready) ---
  function init() {
    resolveDom();

    // Tool buttons
    document.querySelectorAll('.me-tool-btn').forEach(btn => {
      btn.addEventListener('click', () => setTool(btn.dataset.meTool));
    });

    // Sliders
    dom.brushSize.oninput = function() {
      state.brushSize = parseInt(this.value, 10);
      dom.brushSizeVal.textContent = this.value;
      updateStatus();
    };
    dom.maskOpacity.oninput = function() {
      state.maskOpacity = parseInt(this.value, 10) / 100;
      dom.maskOpacityVal.textContent = this.value + '%';
      renderMask();
    };

    // Buttons
    dom.undoBtn.addEventListener('click', undo);
    dom.redoBtn.addEventListener('click', redo);
    dom.saveBtn.addEventListener('click', saveToTrainingSet);
    dom.closeBtn.addEventListener('click', close);

    // Keyboard
    document.addEventListener('keydown', onKeyDown);
  }

  // Init when DOM is ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  return { open, close };
})();
```

- [ ] **Step 2: Add pako dependency for zlib in browser**

In `project/frontend/index.html`, add before the `mask-editor.js` script tag:

```html
  <script src="https://cdn.jsdelivr.net/npm/pako@2.1.0/dist/pako.min.js"></script>
```

- [ ] **Step 3: Add "Edit Masks" handler and i18n to app.js**

In `project/frontend/app.js`, add i18n entries to `LANGS.en`:

```javascript
    'btn.editMasks': '✏️ Edit Masks',
    'maskEditor.title': 'Mask Editor',
    'maskEditor.brushSize': 'Brush Size',
    'maskEditor.opacity': 'Mask Opacity',
    'maskEditor.cells': 'Cells',
    'maskEditor.save': 'Save to Training Set',
```

Add to `LANGS.zh`:

```javascript
    'btn.editMasks': '✏️ 编辑掩码',
    'maskEditor.title': '掩码编辑器',
    'maskEditor.brushSize': '画笔大小',
    'maskEditor.opacity': '掩码透明度',
    'maskEditor.cells': '细胞',
    'maskEditor.save': '保存到训练集',
```

Add click handler (after the detection result close handler, around line 1245):

```javascript
document.getElementById('editMasksBtn').onclick = async function() {
  const slicePath = document.getElementById('realSlicePath').value;
  if (!slicePath) { showToast('No slice loaded', 'warning'); return; }

  // Re-run detection with returnMasks=true
  const btn = this;
  btn.disabled = true;
  btn.textContent = 'Loading...';

  try {
    const res = await fetch('/api/detect/preview', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        slicePath,
        jobId: getOverlayJobId(),
        returnMasks: true,
        params: {
          model: document.getElementById('detectModelSelect').value,
          diameter_um: parseFloat(document.getElementById('detectDiameterUm').value) || 12.0,
          flow_threshold: parseFloat(document.getElementById('detectFlowThreshold').value),
          cellprob_threshold: parseFloat(document.getElementById('detectCellprobThreshold').value),
          min_size_px: parseInt(document.getElementById('detectMinSizePx').value, 10) || 8,
          gpu: document.getElementById('detectGpuToggle').checked,
        },
      }),
    });
    const data = await res.json();
    if (!data.ok) {
      showToast('Detection failed: ' + (data.error || 'unknown'), 'error');
      return;
    }
    MaskEditor.open(slicePath, getOverlayJobId());
  } catch (err) {
    showToast('Failed: ' + err.message, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = '✏️ Edit Masks';
  }
};
```

- [ ] **Step 4: Commit**

```bash
git add project/frontend/mask-editor.js project/frontend/app.js project/frontend/index.html
git commit -m "feat: mask editor canvas with brush, eraser, add/delete/merge/split tools"
```

---

### Task 6: Final validation

- [ ] **Step 1: Run full test suite**

Run: `python -m pytest project/tests -q`
Expected: All pass.

- [ ] **Step 2: Verify server starts**

Run: `cd project && python frontend/server.py`
Expected: No import errors.

- [ ] **Step 3: Manual browser test**

1. Open `http://127.0.0.1:8787`
2. Load a slice, click "Detect Cells"
3. Click "Edit Masks" → mask editor modal opens full-screen
4. Verify: original image visible under colored mask overlay
5. Test brush tool: paint on canvas → mask updates
6. Test eraser: erase mask pixels
7. Test add cell: creates new cell ID
8. Test delete: click cell → removes it
9. Test undo (Ctrl+Z) / redo (Ctrl+Shift+Z)
10. Test zoom (scroll wheel) and pan (middle click or space+drag)
11. Test "Save to Training Set" → check `project/cellpose_training/` directory
12. Close modal with X or Escape
