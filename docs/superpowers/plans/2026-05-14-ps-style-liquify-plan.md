# PS-Style Liquify Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade Brainfast Liquify from two-click landmark placement to a Photoshop-like brush workflow where the user drags anatomy directly, sees immediate local feedback, and still gets the existing 3D Laplacian refinement and final cell-count export.

**Architecture:** Keep the current landmark-pair system as the precision/backward-compatible path. Add a stroke-based brush layer that records pointer strokes, converts them into dense-enough control correspondences, and feeds those correspondences into the same 3D Laplacian solver used today. The first release focuses on fluid 2D/3D interaction and compatibility with existing apply/finalize/class-prior behavior; full learned stroke-prior visualization can follow after this foundation is stable.

**Tech Stack:** Flask blueprints, vanilla JS canvas/pointer events, NumPy/SciPy label warping, NIfTI via nibabel, pytest unit tests, static frontend regression tests.

---

## Current State

Current 2D preview Liquify already has the core idea of drag-based warping:

- `project/frontend/app.js` has `applyLiquifyDrag(x1, y1, x2, y2)`.
- `project/frontend/blueprints/api_overlay.py` exposes `POST /api/overlay/liquify-drag`.
- `project/frontend/server_context.py` has `_apply_liquify_drags(label, drags, tissue_mask)`.
- `project/frontend/services/overlay_service.py` calls `_apply_liquify_drags()` and re-renders the overlay.

Current 3D Liquify is landmark-pair based:

- `project/frontend/app.js` listens for `click` on `liq3dCanvas`, alternates `atlas` and `real`, and posts `/api/liquify-3d/add-pair`.
- `project/scripts/liquify_3d.py` stores `LandmarkPair` rows in `landmarks_3d.csv`.
- `project/frontend/blueprints/api_liquify_3d.py` applies those pairs through `refine_annotation_with_landmarks()`.
- Class prior stores landmark-pair running means in `project/scripts/class_prior.py`.

The user-facing problem is therefore structural: the UI and persistence model make the user select points rather than push pixels/regions.

## Target Interaction

The default Liquify mode should behave like this:

1. User loads a completed 3D job and slice in the Liquify tab.
2. User selects `Brush` mode.
3. User sets brush radius and strength.
4. User presses on the overlay and drags an atlas boundary toward the real anatomy.
5. During drag, the canvas shows a brush circle, drag trail, and deformation guide; after release, the saved stroke updates the current slice preview and feeds the 3D apply path.
6. On release, the stroke is saved as one undoable action.
7. `Apply 3D warp` converts strokes plus any legacy landmarks into a smooth 3D Laplacian field.
8. `Finalize` remains unchanged and produces the refined cell counts.

Landmark mode remains available for precise sparse correction and for existing class-prior data.

## File Structure

### Modify

- `project/frontend/index.html`
  - Add Liquify mode segmented controls, brush radius/strength controls, stroke count labels, and a compact stroke history table.

- `project/frontend/styles.css`
  - Add responsive canvas rules and brush UI styling.

- `project/frontend/app.js`
  - Replace 3D Liquify click-only interaction with mode-aware pointer events.
  - Add local stroke state, brush preview drawing, save/undo/delete stroke calls.
  - Keep existing landmark click flow behind `Landmark` mode.

- `project/frontend/blueprints/api_liquify_3d.py`
  - Add stroke endpoints.
  - Include strokes in state.
  - Include stroke-derived pairs during apply.
  - Clear strokes with `clear`.

- `project/scripts/liquify_3d.py`
  - Add `LiquifyStroke`, `LiquifyStrokeStore`, and `strokes_to_landmark_pairs`.
  - Add a helper that combines CSV landmarks with stroke-derived pairs.

- `project/scripts/class_prior.py`
  - Keep existing landmark-prior format for MVP, but accept derived stroke pairs when saving a job.
  - Add metadata in sample logs indicating whether contribution came from direct pairs, strokes, or both.

- `project/tests/unit/test_liquify_3d.py`
  - Add storage and stroke-to-pair sampling tests.

- `project/tests/unit/test_api_liquify_3d.py`
  - Add stroke endpoint, state, clear, and apply input tests.

- `project/tests/unit/test_frontend_regressions.py`
  - Add static tests for mode controls, pointer handlers, and responsive canvas rules.

### Verify If Task 1 Changes Service Shape

- `project/tests/unit/test_services.py`
  - Existing `apply_liquify_and_render` coverage should still pass because the backend already accepts `drags[]`.

---

## Task 1: Smooth 2D Preview Liquify Stroke Batching

**Files:**
- Modify: `project/frontend/app.js`
- Test: `project/tests/unit/test_frontend_regressions.py`

This task makes the existing 2D Liquify tool feel continuous instead of "drag once, wait, drag once, wait." It does not alter the 3D tab yet.

- [ ] **Step 1: Write frontend regression tests for batched 2D drags**

Add this test to `project/tests/unit/test_frontend_regressions.py`:

```python
def test_2d_liquify_batches_pointer_stroke_before_posting():
    js = (_FRONTEND_DIR / "app.js").read_text(encoding="utf-8", errors="replace")

    assert "liquifyStrokePoints" in js
    assert "sampleLiquifyStrokePoint" in js
    assert "buildLiquifyDragBatch" in js
    assert "payload.drags = buildLiquifyDragBatch" in js
    assert "drawCanvas.addEventListener('pointerdown'" in js
    assert "drawCanvas.addEventListener('pointermove'" in js
    assert "drawCanvas.addEventListener('pointerup'" in js
```

- [ ] **Step 2: Run the focused failing test**

Run:

```powershell
python -m pytest project/tests/unit/test_frontend_regressions.py::test_2d_liquify_batches_pointer_stroke_before_posting -q
```

Expected: FAIL because these helpers do not exist yet.

- [ ] **Step 3: Add stroke batching state and helpers**

In `project/frontend/app.js`, near the drawing editor state, add:

```javascript
let liquifyStrokePoints = [];

function sampleLiquifyStrokePoint(x, y) {
  const last = liquifyStrokePoints[liquifyStrokePoints.length - 1];
  if (last && Math.hypot(x - last.x, y - last.y) < 3) return;
  liquifyStrokePoints.push({ x: Number(x), y: Number(y) });
}

function buildLiquifyDragBatch(points, radius, strength) {
  const drags = [];
  for (let i = 1; i < points.length; i += 1) {
    const a = points[i - 1];
    const b = points[i];
    const dist = Math.hypot(b.x - a.x, b.y - a.y);
    if (dist < 1.5) continue;
    drags.push({
      x1: a.x,
      y1: a.y,
      x2: b.x,
      y2: b.y,
      radius,
      strength,
    });
  }
  return drags;
}
```

- [ ] **Step 4: Change 2D canvas mouse handlers to pointer handlers**

Replace the 2D `mousedown`, `mousemove`, and `mouseup` Liquify path with pointer-aware logic while preserving existing behavior for line, arrow, scalebar, text, and select. The Liquify-specific path should:

```javascript
drawCanvas.addEventListener('pointerdown', e => {
  if (currentTool === 'select') return;
  const { x, y } = canvasCoords(e);
  if (currentTool === 'liquify') {
    isDrawing = true;
    liquifyStrokePoints = [];
    sampleLiquifyStrokePoint(x, y);
    drawStartX = x;
    drawStartY = y;
    drawCanvas.setPointerCapture?.(e.pointerId);
    return;
  }
  // Keep existing text/scalebar/annotation start logic here.
});

drawCanvas.addEventListener('pointermove', e => {
  const { x, y } = canvasCoords(e);
  if (isDrawing && currentTool === 'liquify') {
    sampleLiquifyStrokePoint(x, y);
    drawPreviewStroke(x, y);
    hideRegionTooltip();
    return;
  }
  // Keep existing hover/annotation preview logic here.
});

drawCanvas.addEventListener('pointerup', e => {
  if (!isDrawing) return;
  const { x, y } = canvasCoords(e);
  if (currentTool === 'liquify') {
    sampleLiquifyStrokePoint(x, y);
    isDrawing = false;
    drawCanvas.releasePointerCapture?.(e.pointerId);
    const drags = buildLiquifyDragBatch(
      liquifyStrokePoints,
      getLiquifyRadius(),
      getLiquifyStrength(),
    );
    liquifyStrokePoints = [];
    if (drags.length) applyLiquifyDragBatch(drags);
    redrawAnnotations();
    return;
  }
  // Keep existing mouseup annotation logic here.
});
```

- [ ] **Step 5: Add `applyLiquifyDragBatch` and keep the old helper as a wrapper**

In `project/frontend/app.js`, replace the body of `applyLiquifyDrag()` with a wrapper and add:

```javascript
async function applyLiquifyDragBatch(drags) {
  if (liquifyBusy || !Array.isArray(drags) || !drags.length) return;

  const payload = buildOverlayRequestPayload();
  if (!payload.realPath) {
    showToast(t('toast.setRealSliceFirst'), 'warning');
    return;
  }
  payload.drags = drags;
  payload.jobId = getOverlayJobId();

  liquifyBusy = true;
  try {
    const res = await fetch('/api/overlay/liquify-drag', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    }).then(r => r.json());
    if (!res.ok) {
      showToast(`Liquify failed: ${res.error || '?'}`, 'error');
      return;
    }
    syncOverlayJobId(res);
    if (res.correctedLabelPath) {
      document.getElementById('atlasLabelPath').value = res.correctedLabelPath;
    }
    await loadPreviewIntoCanvas();
    hoverLastPixelKey = '';
    hideRegionTooltip();
    showToast(`Liquify applied (${drags.length} segment${drags.length === 1 ? '' : 's'}).`, 'success', 1600);
  } catch (e) {
    showToast(`Liquify failed: ${e?.message || '?'}`, 'error');
  } finally {
    liquifyBusy = false;
  }
}

async function applyLiquifyDrag(x1, y1, x2, y2) {
  const dist = Math.hypot(x2 - x1, y2 - y1);
  if (dist < 2.0) return;
  await applyLiquifyDragBatch([{
    x1: Number(x1),
    y1: Number(y1),
    x2: Number(x2),
    y2: Number(y2),
    radius: getLiquifyRadius(),
    strength: getLiquifyStrength(),
  }]);
}
```

- [ ] **Step 6: Run the test**

Run:

```powershell
python -m pytest project/tests/unit/test_frontend_regressions.py::test_2d_liquify_batches_pointer_stroke_before_posting -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add project/frontend/app.js project/tests/unit/test_frontend_regressions.py
git commit -m "feat(liquify): batch 2d brush strokes"
```

---

## Task 2: 3D Stroke Data Model And Sampling

**Files:**
- Modify: `project/scripts/liquify_3d.py`
- Test: `project/tests/unit/test_liquify_3d.py`

This task introduces stroke persistence without touching Flask routes.

- [ ] **Step 1: Write stroke storage and sampling tests**

Add to `project/tests/unit/test_liquify_3d.py`:

```python
def test_stroke_store_add_list_remove_clear(tmp_path):
    from project.scripts.liquify_3d import LiquifyStrokeStore

    store = LiquifyStrokeStore(tmp_path / "strokes_3d.jsonl")
    stroke = store.add_stroke(
        z=12,
        points=[{"x": 10, "y": 20}, {"x": 18, "y": 21}, {"x": 25, "y": 26}],
        radius=40,
        strength=0.8,
        image_dims_yx=(100, 200),
    )

    assert stroke.z == 12
    assert len(store.list_strokes()) == 1
    assert store.list_strokes()[0].points[0] == (20.0, 10.0)

    removed = store.remove_stroke(0)
    assert removed.z == 12
    assert store.list_strokes() == []

    store.add_stroke(
        z=1,
        points=[{"x": 1, "y": 2}, {"x": 5, "y": 6}],
        radius=20,
        strength=0.5,
        image_dims_yx=(10, 10),
    )
    store.clear()
    assert store.list_strokes() == []


def test_strokes_to_landmark_pairs_samples_drag_segments():
    from project.scripts.liquify_3d import LiquifyStroke, strokes_to_landmark_pairs

    strokes = [
        LiquifyStroke(
            z=4,
            points=[(10.0, 10.0), (10.0, 20.0), (15.0, 25.0)],
            radius=30.0,
            strength=1.0,
            image_dims_yx=(100.0, 100.0),
            created_at="2026-05-14T12:00:00Z",
        )
    ]

    pairs = strokes_to_landmark_pairs(strokes, annotation_shape=(20, 50, 50))

    assert len(pairs) > 2
    assert all(p.z == 4 for p in pairs)
    central = pairs[0]
    assert central.atlas == (5.0, 5.0)
    assert central.real[1] > central.atlas[1]
```

- [ ] **Step 2: Run the focused failing tests**

Run:

```powershell
python -m pytest project/tests/unit/test_liquify_3d.py::test_stroke_store_add_list_remove_clear project/tests/unit/test_liquify_3d.py::test_strokes_to_landmark_pairs_samples_drag_segments -q
```

Expected: FAIL because the stroke classes do not exist yet.

- [ ] **Step 3: Add stroke dataclass**

In `project/scripts/liquify_3d.py`, below `LandmarkPair`, add:

```python
@dataclass(frozen=True)
class LiquifyStroke:
    """One PS-style brush stroke on a single z slice.

    points are stored as (y, x) coordinates in annotation-grid voxels.
    """

    z: int
    points: list[tuple[float, float]]
    radius: float
    strength: float
    image_dims_yx: tuple[float, float] | None
    created_at: str
```

- [ ] **Step 4: Add `LiquifyStrokeStore`**

In `project/scripts/liquify_3d.py`, add imports:

```python
import datetime as _dt
import json
```

Then add:

```python
class LiquifyStrokeStore:
    """JSONL-backed store of PS-style brush strokes for one job."""

    def __init__(self, jsonl_path: Path | str) -> None:
        self._path = Path(jsonl_path)

    @property
    def path(self) -> Path:
        return self._path

    def list_strokes(self) -> list[LiquifyStroke]:
        if not self._path.exists():
            return []
        strokes: list[LiquifyStroke] = []
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                pts = [(float(p["y"]), float(p["x"])) for p in row.get("points", [])]
                dims = row.get("image_dims_yx")
                image_dims = None
                if isinstance(dims, list | tuple) and len(dims) == 2:
                    image_dims = (float(dims[0]), float(dims[1]))
                strokes.append(
                    LiquifyStroke(
                        z=int(row["z"]),
                        points=pts,
                        radius=float(row.get("radius", 80.0)),
                        strength=float(row.get("strength", 0.72)),
                        image_dims_yx=image_dims,
                        created_at=str(row.get("created_at", "")),
                    )
                )
        return strokes

    def add_stroke(
        self,
        *,
        z: int,
        points: list[dict] | list[tuple[float, float]],
        radius: float,
        strength: float,
        image_dims_yx: tuple[float, float] | None = None,
    ) -> LiquifyStroke:
        parsed: list[tuple[float, float]] = []
        for p in points:
            if isinstance(p, dict):
                parsed.append((float(p["y"]), float(p["x"])))
            else:
                parsed.append((float(p[0]), float(p[1])))
        if len(parsed) < 2:
            raise ValueError("stroke must contain at least two points")
        stroke = LiquifyStroke(
            z=int(z),
            points=parsed,
            radius=float(radius),
            strength=float(strength),
            image_dims_yx=image_dims_yx,
            created_at=_dt.datetime.now(_dt.UTC).isoformat(),
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "z": stroke.z,
            "points": [{"y": y, "x": x} for y, x in stroke.points],
            "radius": stroke.radius,
            "strength": stroke.strength,
            "image_dims_yx": list(stroke.image_dims_yx) if stroke.image_dims_yx else None,
            "created_at": stroke.created_at,
        }
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        return stroke

    def remove_stroke(self, index: int) -> LiquifyStroke:
        strokes = self.list_strokes()
        if not 0 <= index < len(strokes):
            raise IndexError(f"stroke index {index} out of range [0, {len(strokes)})")
        removed = strokes[index]
        self._rewrite([s for i, s in enumerate(strokes) if i != index])
        return removed

    def clear(self) -> None:
        if self._path.exists():
            self._path.unlink()

    def _rewrite(self, strokes: Iterable[LiquifyStroke]) -> None:
        import os as _os
        import tempfile as _tempfile

        self._path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = _tempfile.mkstemp(
            prefix=self._path.name + ".",
            suffix=".tmp",
            dir=str(self._path.parent),
        )
        tmp_path = Path(tmp_name)
        try:
            with _os.fdopen(fd, "w", encoding="utf-8") as fh:
                for stroke in strokes:
                    row = {
                        "z": stroke.z,
                        "points": [{"y": y, "x": x} for y, x in stroke.points],
                        "radius": stroke.radius,
                        "strength": stroke.strength,
                        "image_dims_yx": list(stroke.image_dims_yx) if stroke.image_dims_yx else None,
                        "created_at": stroke.created_at,
                    }
                    fh.write(json.dumps(row, separators=(",", ":")) + "\n")
            tmp_path.replace(self._path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()
```

- [ ] **Step 5: Add stroke-to-landmark sampling**

In `project/scripts/liquify_3d.py`, add:

```python
def _rescale_stroke_point(
    y: float,
    x: float,
    *,
    image_dims_yx: tuple[float, float] | None,
    annotation_shape: tuple[int, int, int] | None,
) -> tuple[float, float]:
    if not image_dims_yx or not annotation_shape:
        return float(y), float(x)
    img_h, img_w = image_dims_yx
    if img_h <= 0 or img_w <= 0:
        return float(y), float(x)
    _d, ann_h, ann_w = annotation_shape
    return float(y) * (ann_h / img_h), float(x) * (ann_w / img_w)


def strokes_to_landmark_pairs(
    strokes: list[LiquifyStroke],
    *,
    annotation_shape: tuple[int, int, int] | None = None,
    min_segment_px: float = 2.0,
) -> list[LandmarkPair]:
    """Convert brush strokes into source/target correspondences.

    A drag from A to B means the atlas feature currently at A should move
    toward real anatomy at B. Radius and strength create neighbouring
    controls so the Laplacian solve behaves like a soft brush rather than a
    single-point pull.
    """

    pairs: list[LandmarkPair] = []
    offsets = (
        (0.0, 0.0),
        (0.35, 0.0),
        (-0.35, 0.0),
        (0.0, 0.35),
        (0.0, -0.35),
        (0.25, 0.25),
        (0.25, -0.25),
        (-0.25, 0.25),
        (-0.25, -0.25),
    )
    for stroke in strokes:
        pts = stroke.points
        if len(pts) < 2:
            continue
        radius = max(1.0, float(stroke.radius))
        sigma = max(1.0, radius * 0.46)
        strength = float(np.clip(stroke.strength, 0.05, 1.5))
        for i in range(1, len(pts)):
            y0, x0 = pts[i - 1]
            y1, x1 = pts[i]
            dy = y1 - y0
            dx = x1 - x0
            dist = float(np.hypot(y1 - y0, x1 - x0))
            if dist < min_segment_px:
                continue
            for off_y_frac, off_x_frac in offsets:
                off_y = off_y_frac * radius
                off_x = off_x_frac * radius
                off_dist = float(np.hypot(off_y, off_x))
                falloff = float(np.exp(-0.5 * (off_dist * off_dist) / (sigma * sigma)))
                pull = falloff * strength
                if pull < 0.04:
                    continue
                atlas_y = y0 + off_y
                atlas_x = x0 + off_x
                real_y = atlas_y + dy * pull
                real_x = atlas_x + dx * pull
                ay, ax = _rescale_stroke_point(
                    atlas_y,
                    atlas_x,
                    image_dims_yx=stroke.image_dims_yx,
                    annotation_shape=annotation_shape,
                )
                ry, rx = _rescale_stroke_point(
                    real_y,
                    real_x,
                    image_dims_yx=stroke.image_dims_yx,
                    annotation_shape=annotation_shape,
                )
                if annotation_shape:
                    _d, ann_h, ann_w = annotation_shape
                    ay = float(np.clip(ay, 0, ann_h - 1))
                    ax = float(np.clip(ax, 0, ann_w - 1))
                    ry = float(np.clip(ry, 0, ann_h - 1))
                    rx = float(np.clip(rx, 0, ann_w - 1))
                pairs.append(
                    LandmarkPair(
                        z=int(stroke.z),
                        real=(float(ry), float(rx)),
                        atlas=(float(ay), float(ax)),
                    )
                )
    return pairs
```

- [ ] **Step 6: Run the focused tests**

Run:

```powershell
python -m pytest project/tests/unit/test_liquify_3d.py::test_stroke_store_add_list_remove_clear project/tests/unit/test_liquify_3d.py::test_strokes_to_landmark_pairs_samples_drag_segments -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add project/scripts/liquify_3d.py project/tests/unit/test_liquify_3d.py
git commit -m "feat(liquify): add 3d brush stroke model"
```

---

## Task 3: 3D Stroke API Endpoints

**Files:**
- Modify: `project/frontend/blueprints/api_liquify_3d.py`
- Modify: `project/scripts/liquify_3d.py`
- Test: `project/tests/unit/test_api_liquify_3d.py`

This task lets the frontend save, list, delete, and clear brush strokes.

- [ ] **Step 1: Write API tests**

Add to `project/tests/unit/test_api_liquify_3d.py`:

```python
def test_add_stroke_then_state_lists_strokes(client):
    resp = client.post(
        "/api/liquify-3d/stroke",
        data=json.dumps(
            {
                "jobId": "strokeJob",
                "z": 7,
                "points": [{"x": 10, "y": 20}, {"x": 15, "y": 24}, {"x": 18, "y": 28}],
                "radius": 55,
                "strength": 0.9,
                "image_dims_yx": [100, 200],
            }
        ),
        content_type="application/json",
    )
    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    assert data["ok"] is True
    assert data["stroke_count"] == 1

    state = client.get("/api/liquify-3d/state?job=strokeJob").get_json()
    assert state["stroke_count"] == 1
    assert state["strokes"][0]["z"] == 7
    assert state["strokes"][0]["point_count"] == 3
    assert state["derived_pair_count"] >= 2


def test_remove_stroke(client):
    client.post(
        "/api/liquify-3d/stroke",
        data=json.dumps(
            {
                "jobId": "removeStrokeJob",
                "z": 2,
                "points": [{"x": 1, "y": 2}, {"x": 6, "y": 7}],
                "radius": 20,
                "strength": 0.5,
            }
        ),
        content_type="application/json",
    )

    resp = client.delete("/api/liquify-3d/stroke/0?job=removeStrokeJob")
    assert resp.status_code == 200
    assert resp.get_json()["stroke_count"] == 0

    state = client.get("/api/liquify-3d/state?job=removeStrokeJob").get_json()
    assert state["stroke_count"] == 0


def test_clear_empties_pairs_and_strokes(client):
    client.post(
        "/api/liquify-3d/add-pair",
        data=json.dumps({"jobId": "clearMixed", "z": 1, "real": [2, 3], "atlas": [4, 5]}),
        content_type="application/json",
    )
    client.post(
        "/api/liquify-3d/stroke",
        data=json.dumps(
            {
                "jobId": "clearMixed",
                "z": 1,
                "points": [{"x": 1, "y": 2}, {"x": 4, "y": 8}],
                "radius": 20,
                "strength": 0.5,
            }
        ),
        content_type="application/json",
    )

    resp = client.post(
        "/api/liquify-3d/clear",
        data=json.dumps({"jobId": "clearMixed"}),
        content_type="application/json",
    )
    assert resp.status_code == 200

    state = client.get("/api/liquify-3d/state?job=clearMixed").get_json()
    assert state["pair_count"] == 0
    assert state["stroke_count"] == 0
```

- [ ] **Step 2: Run focused failing tests**

Run:

```powershell
python -m pytest project/tests/unit/test_api_liquify_3d.py::test_add_stroke_then_state_lists_strokes project/tests/unit/test_api_liquify_3d.py::test_remove_stroke project/tests/unit/test_api_liquify_3d.py::test_clear_empties_pairs_and_strokes -q
```

Expected: FAIL because routes do not exist yet.

- [ ] **Step 3: Import stroke helpers**

In `project/frontend/blueprints/api_liquify_3d.py`, extend imports from `project.scripts.liquify_3d`:

```python
from project.scripts.liquify_3d import (
    LandmarkStore,
    LiquifyStroke,
    LiquifyStrokeStore,
    refine_annotation_with_landmarks,
    strokes_to_landmark_pairs,
)
```

Mirror the same names in the fallback import block.

- [ ] **Step 4: Add stroke filename and helper functions**

Near `_LANDMARKS_FILENAME`, add:

```python
_STROKES_FILENAME = "liquify_strokes_3d.jsonl"
```

Add:

```python
def _stroke_store_for(job_id: str) -> LiquifyStrokeStore:
    return LiquifyStrokeStore(_job_file_for(job_id, _STROKES_FILENAME))


def _stroke_to_dict(stroke: LiquifyStroke) -> dict:
    return {
        "z": stroke.z,
        "points": [{"y": y, "x": x} for y, x in stroke.points],
        "point_count": len(stroke.points),
        "radius": stroke.radius,
        "strength": stroke.strength,
        "image_dims_yx": list(stroke.image_dims_yx) if stroke.image_dims_yx else None,
        "created_at": stroke.created_at,
    }
```

- [ ] **Step 5: Extend state response**

In `liquify_3d_state()`, after loading pairs, load strokes:

```python
stroke_store = _stroke_store_for(job_id)
strokes = stroke_store.list_strokes()
derived_pairs = strokes_to_landmark_pairs(strokes, annotation_shape=annotation_shape)
```

Return these fields:

```python
"strokes": [_stroke_to_dict(s) for s in strokes],
"stroke_count": len(strokes),
"derived_pair_count": len(derived_pairs),
"total_control_count": len(pairs) + len(derived_pairs),
```

- [ ] **Step 6: Add `POST /liquify-3d/stroke`**

Add route:

```python
@bp.post("/liquify-3d/stroke")
def liquify_3d_add_stroke():
    payload = request.get_json(silent=True) or {}
    job_id = ctx._payload_job_id(payload)
    try:
        z = int(payload["z"])
        points = payload["points"]
        radius = float(payload.get("radius", 80.0))
        strength = float(payload.get("strength", 0.72))
    except (KeyError, TypeError, ValueError) as exc:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": f"invalid payload: {exc}",
                    "error_code": ERR_INVALID_INPUT,
                    "required": "{jobId, z:int, points:[{x,y}], radius:number, strength:number}",
                }
            ),
            400,
        )

    image_dims = payload.get("image_dims_yx")
    image_dims_yx = None
    if isinstance(image_dims, list | tuple) and len(image_dims) == 2:
        image_dims_yx = (float(image_dims[0]), float(image_dims[1]))

    store = _stroke_store_for(job_id)
    try:
        stroke = store.add_stroke(
            z=z,
            points=points,
            radius=radius,
            strength=strength,
            image_dims_yx=image_dims_yx,
        )
    except (KeyError, TypeError, ValueError) as exc:
        return jsonify({"ok": False, "error": str(exc), "error_code": ERR_INVALID_INPUT}), 400

    return jsonify(
        {
            "ok": True,
            "jobId": job_id,
            "stroke": _stroke_to_dict(stroke),
            "stroke_count": len(store.list_strokes()),
        }
    )
```

- [ ] **Step 7: Add `DELETE /liquify-3d/stroke/<index>`**

Add route:

```python
@bp.delete("/liquify-3d/stroke/<int:index>")
def liquify_3d_remove_stroke(index: int):
    job_id = ctx._query_job_id()
    store = _stroke_store_for(job_id)
    try:
        removed = store.remove_stroke(index)
    except IndexError as exc:
        return jsonify({"ok": False, "error": str(exc), "error_code": ERR_INVALID_INPUT}), 400
    return jsonify(
        {
            "ok": True,
            "jobId": job_id,
            "removed": _stroke_to_dict(removed),
            "stroke_count": len(store.list_strokes()),
        }
    )
```

- [ ] **Step 8: Clear strokes in `/clear`**

Update `liquify_3d_clear()`:

```python
store = _landmark_store_for(job_id)
store.clear()
_stroke_store_for(job_id).clear()
return jsonify({"ok": True, "jobId": job_id, "pair_count": 0, "stroke_count": 0})
```

- [ ] **Step 9: Run API tests**

Run:

```powershell
python -m pytest project/tests/unit/test_api_liquify_3d.py::test_add_stroke_then_state_lists_strokes project/tests/unit/test_api_liquify_3d.py::test_remove_stroke project/tests/unit/test_api_liquify_3d.py::test_clear_empties_pairs_and_strokes -q
```

Expected: PASS.

- [ ] **Step 10: Commit**

```powershell
git add project/frontend/blueprints/api_liquify_3d.py project/scripts/liquify_3d.py project/tests/unit/test_api_liquify_3d.py
git commit -m "feat(liquify): expose 3d brush stroke endpoints"
```

---

## Task 4: Feed Brush Strokes Into 3D Apply

**Files:**
- Modify: `project/scripts/liquify_3d.py`
- Modify: `project/frontend/blueprints/api_liquify_3d.py`
- Test: `project/tests/unit/test_api_liquify_3d.py`
- Test: `project/tests/unit/test_liquify_3d.py`

This task makes brush strokes affect the actual refined annotation instead of only being listed.

- [ ] **Step 1: Write helper test for combined controls**

Add to `project/tests/unit/test_liquify_3d.py`:

```python
def test_landmarks_to_point_arrays_accepts_stroke_derived_pairs():
    from project.scripts.liquify_3d import LandmarkPair, landmarks_to_point_arrays

    pairs = [
        LandmarkPair(z=1, atlas=(10.0, 10.0), real=(10.0, 18.0)),
        LandmarkPair(z=2, atlas=(20.0, 20.0), real=(24.0, 20.0)),
    ]
    source, target = landmarks_to_point_arrays(pairs)

    assert source.shape == (2, 3)
    assert target.shape == (2, 3)
    assert source[0].tolist() == [1.0, 10.0, 18.0]
    assert target[0].tolist() == [1.0, 10.0, 10.0]
```

- [ ] **Step 2: Write apply test for stroke-only job**

Add to `project/tests/unit/test_api_liquify_3d.py`:

```python
def test_apply_accepts_stroke_only_controls(client, tmp_path, monkeypatch):
    job_id = "stroke_only_apply"
    job_dir = tmp_path / "outputs" / "jobs" / job_id
    ann_dir = job_dir / "ants_registration"
    ann_dir.mkdir(parents=True)
    ann = np.zeros((6, 24, 24), dtype=np.int16)
    ann[:, 6:18, 6:18] = 1
    nib.save(nib.Nifti1Image(ann, np.eye(4)), str(ann_dir / "annotation_registered.nii.gz"))

    monkeypatch.setattr(ctx, "OUTPUT_DIR", tmp_path / "outputs")

    stroke_resp = client.post(
        "/api/liquify-3d/stroke",
        data=json.dumps(
            {
                "jobId": job_id,
                "z": 3,
                "points": [{"x": 10, "y": 10}, {"x": 13, "y": 10}, {"x": 16, "y": 11}],
                "radius": 30,
                "strength": 0.8,
                "image_dims_yx": [24, 24],
            }
        ),
        content_type="application/json",
    )
    assert stroke_resp.status_code == 200

    resp = client.post(
        "/api/liquify-3d/apply",
        data=json.dumps({"jobId": job_id, "sync": True, "maxiter": 20}),
        content_type="application/json",
    )
    assert resp.status_code == 200, resp.get_json()
    data = resp.get_json()
    assert data["ok"] is True
    assert data["pair_count"] >= 2
    assert (job_dir / "annotation_refined_liquify3d.nii.gz").exists()
```

- [ ] **Step 3: Run focused failing test**

Run:

```powershell
python -m pytest project/tests/unit/test_api_liquify_3d.py::test_apply_accepts_stroke_only_controls -q
```

Expected: FAIL because `/apply` still only reads `landmarks_3d.csv`.

- [ ] **Step 4: Add `refine_annotation_with_pairs` helper**

In `project/scripts/liquify_3d.py`, extract the inner part of `refine_annotation_with_landmarks()` into:

```python
def refine_annotation_with_pairs(
    annotation_path: Path | str,
    pairs: list[LandmarkPair],
    output_path: Path | str,
    *,
    spacing: tuple[float, float, float] | None = None,
    rtol: float = 1e-2,
    maxiter: int = 500,
    progress_cb=None,
) -> dict:
    annotation_path = Path(annotation_path)
    output_path = Path(output_path)

    def _emit(stage: str, idx: int, pct: int, msg: str) -> None:
        if progress_cb is not None:
            try:
                progress_cb(stage, idx, 4, pct, msg)
            except Exception:
                pass

    _emit("load_inputs", 1, 5, "Loading annotation + liquify controls")
    img = nib.load(str(annotation_path))
    ann = np.asarray(img.dataobj, dtype=np.int32)
    source_pts, target_pts = landmarks_to_point_arrays(pairs)

    if spacing is None:
        zooms = img.header.get_zooms()[:3]
        spacing_val = tuple(float(z) for z in zooms) if all(z > 0 for z in zooms) else None
    else:
        spacing_val = spacing

    _emit(
        "solve",
        2,
        15,
        f"Solving 3D Laplacian on {ann.shape} with {len(pairs)} control pair(s)",
    )
    field = compute_3d_displacement(
        vol_shape=tuple(ann.shape),
        source_pts=source_pts,
        target_pts=target_pts,
        spacing=spacing_val,
        rtol=rtol,
        maxiter=maxiter,
    )

    _emit("apply_warp", 3, 80, "Warping annotation with displacement field")
    warped = apply_3d_warp_to_annotation(ann, field)

    _emit("save", 4, 95, "Writing refined annotation NIfTI")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(warped, img.affine, img.header), str(output_path))
    _emit("done", 4, 100, f"Refined annotation saved to {output_path.name}")

    return {
        "pair_count": len(pairs),
        "displacement_max_voxels": float(np.abs(field).max()) if field.size else 0.0,
        "output_path": str(output_path),
    }
```

Then make `refine_annotation_with_landmarks()` call this helper after reading CSV pairs:

```python
pairs = LandmarkStore(landmarks_csv).list_pairs()
return refine_annotation_with_pairs(
    annotation_path=annotation_path,
    pairs=pairs,
    output_path=output_path,
    spacing=spacing,
    rtol=rtol,
    maxiter=maxiter,
    progress_cb=progress_cb,
)
```

- [ ] **Step 5: Use combined controls in `/apply`**

In `project/frontend/blueprints/api_liquify_3d.py`, import `refine_annotation_with_pairs`.

In `liquify_3d_apply()`, replace:

```python
pairs = store.list_pairs()
if not pairs:
```

with:

```python
explicit_pairs = store.list_pairs()
strokes = _stroke_store_for(job_id).list_strokes()
annotation_shape = None
annotation_path = _resolve_annotation_path(job_id)
if annotation_path is not None:
    import nibabel as nib
    annotation_shape = tuple(nib.load(str(annotation_path)).shape)
stroke_pairs = strokes_to_landmark_pairs(strokes, annotation_shape=annotation_shape)
pairs = [*explicit_pairs, *stroke_pairs]
if not pairs:
```

In sync mode and async runner, call:

```python
refine_annotation_with_pairs(
    annotation_path=annotation_path,
    pairs=pairs,
    output_path=out_path,
    rtol=rtol,
    maxiter=maxiter,
    progress_cb=_progress_cb_for_job(job_id),
)
```

- [ ] **Step 6: Run focused apply tests**

Run:

```powershell
python -m pytest project/tests/unit/test_liquify_3d.py::test_landmarks_to_point_arrays_accepts_stroke_derived_pairs project/tests/unit/test_api_liquify_3d.py::test_apply_accepts_stroke_only_controls -q
```

Expected: PASS.

- [ ] **Step 7: Run existing liquify API tests**

Run:

```powershell
python -m pytest project/tests/unit/test_api_liquify_3d.py project/tests/unit/test_liquify_3d.py -q
```

Expected: PASS.

- [ ] **Step 8: Commit**

```powershell
git add project/scripts/liquify_3d.py project/frontend/blueprints/api_liquify_3d.py project/tests/unit/test_api_liquify_3d.py project/tests/unit/test_liquify_3d.py
git commit -m "feat(liquify): apply brush strokes in 3d warp"
```

---

## Task 5: 3D Brush UI

**Files:**
- Modify: `project/frontend/index.html`
- Modify: `project/frontend/styles.css`
- Modify: `project/frontend/app.js`
- Test: `project/tests/unit/test_frontend_regressions.py`

This task changes the 3D tab's default UX from click-pair placement to brush dragging.

- [ ] **Step 1: Write static UI tests**

Add to `project/tests/unit/test_frontend_regressions.py`:

```python
def test_liquify3d_has_brush_mode_controls():
    html = (_FRONTEND_DIR / "index.html").read_text(encoding="utf-8", errors="replace")
    js = (_FRONTEND_DIR / "app.js").read_text(encoding="utf-8", errors="replace")
    css = (_FRONTEND_DIR / "styles.css").read_text(encoding="utf-8", errors="replace")

    assert 'name="liq3dToolMode"' in html
    assert 'value="brush"' in html
    assert 'value="landmark"' in html
    assert 'id="liq3dBrushRadius"' in html
    assert 'id="liq3dBrushStrength"' in html
    assert 'id="liq3dStrokeCount"' in html
    assert 'id="liq3dStrokesBody"' in html

    assert "liq3dPointerDown" in js
    assert "liq3dPointerMove" in js
    assert "liq3dPointerUp" in js
    assert "postLiquify3dStroke" in js
    assert "renderStrokeHistory" in js
    assert "/api/liquify-3d/stroke" in js

    assert "#liq3dCanvas" in css
    assert "touch-action: none" in css
    assert "max-width: 100%" in css
```

- [ ] **Step 2: Run focused failing test**

Run:

```powershell
python -m pytest project/tests/unit/test_frontend_regressions.py::test_liquify3d_has_brush_mode_controls -q
```

Expected: FAIL because controls and handlers are not present.

- [ ] **Step 3: Add mode controls to HTML**

In `project/frontend/index.html`, replace the "Next click" row with:

```html
<div class="form-row liq3d-tool-row">
  <span data-i18n="liquify3d.toolLabel">Tool:</span>
  <label class="seg-option">
    <input type="radio" name="liq3dToolMode" value="brush" checked />
    <span data-i18n="liquify3d.toolBrush">Brush</span>
  </label>
  <label class="seg-option">
    <input type="radio" name="liq3dToolMode" value="landmark" />
    <span data-i18n="liquify3d.toolLandmark">Landmark</span>
  </label>
  <label class="liq3d-brush-control">
    <span data-i18n="liquify3d.brushRadius">Radius</span>
    <input id="liq3dBrushRadius" type="range" min="8" max="260" value="80" />
    <input id="liq3dBrushRadiusNum" type="number" min="8" max="260" value="80" />
  </label>
  <label class="liq3d-brush-control">
    <span data-i18n="liquify3d.brushStrength">Strength</span>
    <input id="liq3dBrushStrength" type="range" min="5" max="150" value="72" />
    <span id="liq3dBrushStrengthNum" class="hint-text">0.72</span>
  </label>
  <span id="liq3dPendingStatus" class="hint-text"></span>
</div>

<div id="liq3dLandmarkRow" class="form-row" style="gap:12px; align-items:center; margin-top:10px; display:none;">
  <span data-i18n="liquify3d.modeLabel">Next click:</span>
  <label><input type="radio" name="liq3dClickMode" value="atlas" checked />
    <span data-i18n="liquify3d.modeAtlas">Atlas (where region currently is)</span></label>
  <label><input type="radio" name="liq3dClickMode" value="real" />
    <span data-i18n="liquify3d.modeReal">Real (where it should be)</span></label>
</div>
```

In the history header, change the count line to:

```html
<span data-i18n="liquify3d.pairsTitle">Landmark pairs</span>
(<span id="liq3dPairCount">0</span>)
 · <span data-i18n="liquify3d.strokesTitle">Brush strokes</span>
(<span id="liq3dStrokeCount">0</span>)
```

Add a compact stroke history table below the existing landmark table:

```html
<div style="margin-top:10px;">
  <h3 style="font-size:14px;margin:0 0 6px 0;">
    <span data-i18n="liquify3d.strokesTitle">Brush strokes</span>
    (<span id="liq3dStrokeCount">0</span>)
  </h3>
  <div id="liq3dStrokesWrap" style="max-height:180px; overflow:auto; border:1px solid var(--border, #333); border-radius:4px;">
    <table style="width:100%; border-collapse:collapse; font-size:12px;">
      <thead style="background:var(--bg-alt, #222);">
        <tr>
          <th style="padding:4px;">#</th>
          <th style="padding:4px;">z</th>
          <th style="padding:4px;" data-i18n="liquify3d.strokePoints">Points</th>
          <th style="padding:4px;" data-i18n="liquify3d.brushRadius">Radius</th>
          <th style="padding:4px;" data-i18n="liquify3d.brushStrength">Strength</th>
          <th style="padding:4px;"></th>
        </tr>
      </thead>
      <tbody id="liq3dStrokesBody"></tbody>
    </table>
  </div>
</div>
```

- [ ] **Step 4: Add responsive canvas CSS**

In `project/frontend/styles.css`, add:

```css
#liq3dCanvasWrap {
  position: relative;
  max-width: 100%;
  overflow: auto;
}

#liq3dCanvas {
  max-width: 100%;
  height: auto;
  touch-action: none;
}

.liq3d-tool-row {
  gap: 12px;
  align-items: center;
  margin-top: 10px;
  flex-wrap: wrap;
}

.liq3d-brush-control {
  display: inline-flex;
  align-items: center;
  gap: 6px;
}

.liq3d-brush-control input[type="range"] {
  width: 120px;
}

.liq3d-brush-control input[type="number"] {
  width: 64px;
}
```

- [ ] **Step 5: Add 3D brush state**

In the 3D Liquify IIFE state in `project/frontend/app.js`, add:

```javascript
toolMode: 'brush',
strokes: [],
brushStrokePoints: [],
isBrushDragging: false,
brushPreviewPoint: null,
```

Add element handles:

```javascript
const strokeCountEl = el('liq3dStrokeCount');
const brushRadius = el('liq3dBrushRadius');
const brushRadiusNum = el('liq3dBrushRadiusNum');
const brushStrength = el('liq3dBrushStrength');
const brushStrengthNum = el('liq3dBrushStrengthNum');
const landmarkRow = el('liq3dLandmarkRow');
const strokesBody = el('liq3dStrokesBody');
```

- [ ] **Step 6: Add brush value helpers**

In the IIFE:

```javascript
function get3dBrushRadius() {
  const v = Number(brushRadius?.value || brushRadiusNum?.value || 80);
  return Math.max(8, Math.min(260, Number.isFinite(v) ? v : 80));
}

function get3dBrushStrength() {
  const pct = Number(brushStrength?.value || 72);
  return Math.max(0.05, Math.min(1.5, Number.isFinite(pct) ? pct / 100 : 0.72));
}

function sample3dBrushPoint(x, y) {
  const last = state.brushStrokePoints[state.brushStrokePoints.length - 1];
  if (last && Math.hypot(x - last.x, y - last.y) < 3) return;
  state.brushStrokePoints.push({ x: Number(x), y: Number(y) });
}
```

- [ ] **Step 7: Render brush strokes**

Extend `redraw()`:

```javascript
state.strokes.forEach((s, i) => {
  if (s.z !== state.currentZ || !Array.isArray(s.points) || s.points.length < 2) return;
  ctx.strokeStyle = 'rgba(0,255,255,0.85)';
  ctx.lineWidth = 2;
  ctx.beginPath();
  s.points.forEach((p, idx) => {
    const x = Number(p.x);
    const y = Number(p.y);
    if (idx === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();
  const first = s.points[0];
  ctx.fillStyle = '#00ffff';
  ctx.font = 'bold 12px sans-serif';
  ctx.fillText(`S${i + 1}`, Number(first.x), Number(first.y) - 8);
});

if (state.toolMode === 'brush' && state.brushPreviewPoint) {
  const p = state.brushPreviewPoint;
  const r = get3dBrushRadius();
  ctx.strokeStyle = 'rgba(255,255,255,0.9)';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([5, 4]);
  ctx.beginPath();
  ctx.arc(p.x, p.y, r, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);
}
```

- [ ] **Step 8: Add pointer handlers**

Replace direct `canvas.addEventListener('click', ...)` with mode-aware handlers. Keep the landmark click logic inside `handleLandmarkClick(e)`.

```javascript
async function postLiquify3dStroke(points) {
  if (!Array.isArray(points) || points.length < 2) return;
  pendingStatus.textContent = `Saving brush stroke…`;
  try {
    const resp = await fetch('/api/liquify-3d/stroke', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        jobId: currentJobId(),
        z: state.currentZ,
        points,
        radius: get3dBrushRadius(),
        strength: get3dBrushStrength(),
        image_dims_yx: [state.imageNaturalSize.h, state.imageNaturalSize.w],
      }),
    });
    const data = await resp.json();
    if (!data.ok) throw new Error(data.error || 'stroke failed');
    pendingStatus.textContent = `Brush stroke added (${data.stroke_count} total).`;
    await refreshState();
    redraw();
  } catch (err) {
    pendingStatus.textContent = 'Stroke failed: ' + err.message;
  }
}

function liq3dPointerDown(e) {
  if (!state.sliceFiles.length || state.toolMode !== 'brush') return;
  e.preventDefault();
  const { x, y } = canvasToImageCoords(e);
  state.isBrushDragging = true;
  state.brushStrokePoints = [];
  state.brushPreviewPoint = { x, y };
  sample3dBrushPoint(x, y);
  canvas.setPointerCapture?.(e.pointerId);
  redraw();
}

function liq3dPointerMove(e) {
  if (!state.sliceFiles.length || state.toolMode !== 'brush') return;
  const { x, y } = canvasToImageCoords(e);
  state.brushPreviewPoint = { x, y };
  if (state.isBrushDragging) sample3dBrushPoint(x, y);
  redraw();
}

function liq3dPointerUp(e) {
  if (!state.isBrushDragging || state.toolMode !== 'brush') return;
  e.preventDefault();
  const { x, y } = canvasToImageCoords(e);
  sample3dBrushPoint(x, y);
  canvas.releasePointerCapture?.(e.pointerId);
  state.isBrushDragging = false;
  const points = state.brushStrokePoints.slice();
  state.brushStrokePoints = [];
  postLiquify3dStroke(points);
}

canvas.addEventListener('pointerdown', liq3dPointerDown);
canvas.addEventListener('pointermove', liq3dPointerMove);
canvas.addEventListener('pointerup', liq3dPointerUp);
canvas.addEventListener('click', (e) => {
  if (state.toolMode === 'landmark') handleLandmarkClick(e);
});
```

- [ ] **Step 9: Update state refresh and history rendering**

In `refreshState()`:

```javascript
state.strokes = data.strokes || [];
renderStrokeHistory();
```

Add:

```javascript
function renderStrokeHistory() {
  if (strokeCountEl) strokeCountEl.textContent = String(state.strokes.length);
  if (!strokesBody) return;
  strokesBody.innerHTML = '';
  state.strokes.forEach((s, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="padding:4px;">${i + 1}</td>
      <td style="padding:4px;">${s.z}</td>
      <td style="padding:4px;">${s.point_count || (s.points || []).length}</td>
      <td style="padding:4px;">${Number(s.radius || 0).toFixed(0)}</td>
      <td style="padding:4px;">${Number(s.strength || 0).toFixed(2)}</td>
      <td style="padding:4px;">
        <button type="button" data-idx="${i}" class="liq3d-remove-stroke-btn"
          style="background:transparent;color:var(--danger,#f55);border:none;cursor:pointer;">&times;</button>
      </td>`;
    strokesBody.appendChild(tr);
  });
  strokesBody.querySelectorAll('.liq3d-remove-stroke-btn').forEach((btn) => {
    btn.addEventListener('click', async () => {
      const idx = parseInt(btn.getAttribute('data-idx'), 10);
      await fetch(`/api/liquify-3d/stroke/${idx}?job=${encodeURIComponent(currentJobId())}`, {
        method: 'DELETE',
      });
      await refreshState();
      redraw();
    });
  });
}
```

- [ ] **Step 10: Wire mode and brush controls**

Add:

```javascript
document.querySelectorAll('input[name="liq3dToolMode"]').forEach((r) => {
  r.addEventListener('change', (e) => {
    state.toolMode = e.target.value === 'landmark' ? 'landmark' : 'brush';
    if (landmarkRow) landmarkRow.style.display = state.toolMode === 'landmark' ? '' : 'none';
    canvas.style.cursor = state.toolMode === 'brush' ? 'none' : 'crosshair';
    _saveTabState();
    redraw();
  });
});

brushRadius?.addEventListener('input', () => {
  if (brushRadiusNum) brushRadiusNum.value = brushRadius.value;
  redraw();
});
brushRadiusNum?.addEventListener('change', () => {
  if (brushRadius) brushRadius.value = String(get3dBrushRadius());
  redraw();
});
brushStrength?.addEventListener('input', () => {
  if (brushStrengthNum) brushStrengthNum.textContent = get3dBrushStrength().toFixed(2);
});
```

- [ ] **Step 11: Run frontend regression test**

Run:

```powershell
python -m pytest project/tests/unit/test_frontend_regressions.py::test_liquify3d_has_brush_mode_controls -q
```

Expected: PASS.

- [ ] **Step 12: Commit**

```powershell
git add project/frontend/index.html project/frontend/styles.css project/frontend/app.js project/tests/unit/test_frontend_regressions.py
git commit -m "feat(liquify): add ps-style 3d brush UI"
```

---

## Task 6: Stroke Undo And Clear Semantics

**Files:**
- Modify: `project/frontend/app.js`
- Modify: `project/frontend/blueprints/api_liquify_3d.py`
- Test: `project/tests/unit/test_api_liquify_3d.py`
- Test: `project/tests/unit/test_frontend_regressions.py`

This task makes Ctrl+Z intuitive in brush mode.

- [ ] **Step 1: Add frontend regression test**

Add to `project/tests/unit/test_frontend_regressions.py`:

```python
def test_liquify3d_undo_is_mode_aware():
    js = (_FRONTEND_DIR / "app.js").read_text(encoding="utf-8", errors="replace")

    assert "undoLastStroke" in js
    assert "/api/liquify-3d/stroke/${lastIdx}" in js or "/api/liquify-3d/stroke/" in js
    assert "state.toolMode === 'brush'" in js
```

- [ ] **Step 2: Add API test for stroke undo endpoint if not already covered**

If Task 3's `test_remove_stroke` exists, no new API test is needed. If it was skipped, add it now.

- [ ] **Step 3: Add `undoLastStroke()`**

In `project/frontend/app.js`, inside the 3D Liquify IIFE:

```javascript
async function undoLastStroke() {
  if (!state.strokes.length) {
    pendingStatus.textContent = 'No brush stroke to undo.';
    return;
  }
  const lastIdx = state.strokes.length - 1;
  try {
    const resp = await fetch(
      `/api/liquify-3d/stroke/${lastIdx}?job=${encodeURIComponent(currentJobId())}`,
      { method: 'DELETE' },
    );
    const data = await resp.json();
    if (!data.ok) throw new Error(data.error || 'stroke undo failed');
    pendingStatus.textContent = `Undid brush stroke #${lastIdx + 1}.`;
    await refreshState();
    redraw();
  } catch (err) {
    pendingStatus.textContent = 'Undo failed: ' + err.message;
  }
}
```

- [ ] **Step 4: Make `undoLastPair()` mode-aware**

Replace undo button handler with:

```javascript
async function undoLiquify3d() {
  if (state.toolMode === 'brush') {
    await undoLastStroke();
  } else {
    await undoLastPair();
  }
}

el('liq3dUndoBtn')?.addEventListener('click', undoLiquify3d);
```

In the Ctrl+Z handler, replace `undoLastPair();` with:

```javascript
undoLiquify3d();
```

- [ ] **Step 5: Confirm clear removes both**

The clear handler should call `/api/liquify-3d/clear` and then:

```javascript
await refreshState();
redraw();
```

Because Task 3 clears both pairs and strokes, no separate frontend endpoint is needed.

- [ ] **Step 6: Run tests**

Run:

```powershell
python -m pytest project/tests/unit/test_frontend_regressions.py::test_liquify3d_undo_is_mode_aware project/tests/unit/test_api_liquify_3d.py::test_remove_stroke project/tests/unit/test_api_liquify_3d.py::test_clear_empties_pairs_and_strokes -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```powershell
git add project/frontend/app.js project/frontend/blueprints/api_liquify_3d.py project/tests/unit/test_api_liquify_3d.py project/tests/unit/test_frontend_regressions.py
git commit -m "feat(liquify): make brush undo mode-aware"
```

---

## Task 7: Class Prior Compatibility For Brush Jobs

**Files:**
- Modify: `project/frontend/blueprints/api_liquify_3d.py`
- Modify: `project/scripts/class_prior.py`
- Test: `project/tests/unit/test_api_liquify_3d.py`

This task keeps the existing class-prior system useful for brush-corrected jobs without designing a new learned vector-field format yet.

- [ ] **Step 1: Write API test**

Add to `project/tests/unit/test_api_liquify_3d.py`:

```python
def test_class_prior_save_accepts_stroke_derived_pairs(client, tmp_path, monkeypatch):
    monkeypatch.setattr(ctx, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(ctx, "OUTPUT_DIR", tmp_path / "outputs")

    job_id = "stroke_prior_source"
    client.post(
        "/api/liquify-3d/stroke",
        data=json.dumps(
            {
                "jobId": job_id,
                "z": 10,
                "points": [{"x": 10, "y": 20}, {"x": 20, "y": 20}, {"x": 25, "y": 25}],
                "radius": 40,
                "strength": 0.8,
            }
        ),
        content_type="application/json",
    )

    save_resp = client.post(
        "/api/liquify-3d/class-prior/save",
        data=json.dumps({"jobId": job_id, "class": "ChATe27"}),
        content_type="application/json",
    )

    assert save_resp.status_code == 200, save_resp.get_json()
    data = save_resp.get_json()
    assert data["merged_pair_count"] >= 2
    assert data["source_control_types"] == ["stroke"]
```

- [ ] **Step 2: Run focused failing test**

Run:

```powershell
python -m pytest project/tests/unit/test_api_liquify_3d.py::test_class_prior_save_accepts_stroke_derived_pairs -q
```

Expected: FAIL because class-prior save currently reads only landmark pairs.

- [ ] **Step 3: Update `class_prior_save()`**

In `project/frontend/blueprints/api_liquify_3d.py`, replace:

```python
pairs = _landmark_store_for(job_id).list_pairs()
```

with:

```python
explicit_pairs = _landmark_store_for(job_id).list_pairs()
strokes = _stroke_store_for(job_id).list_strokes()
annotation_shape = None
ann_path = _resolve_annotation_path(job_id)
if ann_path is not None:
    import nibabel as nib
    annotation_shape = tuple(nib.load(str(ann_path)).shape)
stroke_pairs = strokes_to_landmark_pairs(strokes, annotation_shape=annotation_shape)
pairs = [*explicit_pairs, *stroke_pairs]
source_control_types = []
if explicit_pairs:
    source_control_types.append("landmark")
if stroke_pairs:
    source_control_types.append("stroke")
```

Add to response:

```python
"source_control_types": source_control_types,
```

- [ ] **Step 4: Add metadata to class-prior sample log**

In `project/scripts/class_prior.py`, update `ClassPriorStore.merge_sample()` signature to accept optional `source_control_types`:

```python
def merge_sample(
    self,
    *,
    sample_id: str,
    pairs,
    metrics: dict | None = None,
    source_control_types: list[str] | None = None,
) -> dict:
```

When writing the sample log row, include:

```python
"source_control_types": list(source_control_types or ["landmark"]),
```

Update call sites to pass the new keyword only from `class_prior_save()`.

- [ ] **Step 5: Run prior tests**

Run:

```powershell
python -m pytest project/tests/unit/test_api_liquify_3d.py::test_class_prior_save_accepts_stroke_derived_pairs project/tests/unit/test_class_prior.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
git add project/frontend/blueprints/api_liquify_3d.py project/scripts/class_prior.py project/tests/unit/test_api_liquify_3d.py
git commit -m "feat(liquify): include brush controls in class prior"
```

---

## Task 8: Copy, Localization, And Visual Polish

**Files:**
- Modify: `project/frontend/app.js`
- Modify: `project/frontend/index.html`
- Modify: `project/frontend/styles.css`
- Test: `project/tests/unit/test_frontend_regressions.py`

This task makes the UI clear enough for a biologist user and removes the misleading "Next click" default language from the brush path.

- [ ] **Step 1: Add i18n keys**

In `project/frontend/app.js`, add English keys:

```javascript
'liquify3d.toolLabel': 'Tool:',
'liquify3d.toolBrush': 'Brush',
'liquify3d.toolLandmark': 'Landmark',
'liquify3d.brushRadius': 'Radius',
'liquify3d.brushStrength': 'Strength',
'liquify3d.strokesTitle': 'Brush strokes',
'liquify3d.strokePoints': 'Points',
'liquify3d.hint': 'Brush mode: drag the atlas boundary toward the real anatomy. Landmark mode remains available for precise point-pair correction.',
```

Add Chinese keys:

```javascript
'liquify3d.toolLabel': '工具：',
'liquify3d.toolBrush': '画笔',
'liquify3d.toolLandmark': '地标点',
'liquify3d.brushRadius': '半径',
'liquify3d.brushStrength': '力度',
'liquify3d.strokesTitle': '画笔笔画',
'liquify3d.strokePoints': '点数',
'liquify3d.hint': '画笔模式：按住并拖动，把图谱边界推到真实解剖位置。地标点模式仍可用于精确点对校正。',
```

- [ ] **Step 2: Add frontend test for new copy**

Add to `project/tests/unit/test_frontend_regressions.py`:

```python
def test_liquify3d_brush_copy_is_localized():
    js = (_FRONTEND_DIR / "app.js").read_text(encoding="utf-8", errors="replace")

    for key in (
        "liquify3d.toolBrush",
        "liquify3d.toolLandmark",
        "liquify3d.brushRadius",
        "liquify3d.brushStrength",
        "liquify3d.strokesTitle",
    ):
        assert js.count(key) >= 2
```

- [ ] **Step 3: Run focused test**

Run:

```powershell
python -m pytest project/tests/unit/test_frontend_regressions.py::test_liquify3d_brush_copy_is_localized -q
```

Expected: PASS.

- [ ] **Step 4: Commit**

```powershell
git add project/frontend/app.js project/frontend/index.html project/frontend/styles.css project/tests/unit/test_frontend_regressions.py
git commit -m "feat(liquify): polish brush liquify copy"
```

---

## Task 9: Verification Pass

**Files:**
- No source edits unless verification exposes a defect.

- [ ] **Step 1: Run focused unit suites**

Run:

```powershell
python -m pytest project/tests/unit/test_liquify_3d.py project/tests/unit/test_api_liquify_3d.py project/tests/unit/test_frontend_regressions.py project/tests/unit/test_class_prior.py -q
```

Expected: PASS.

- [ ] **Step 2: Run service tests touched by 2D Liquify**

Run:

```powershell
python -m pytest project/tests/unit/test_services.py -q
```

Expected: PASS.

- [ ] **Step 3: Run a smoke server boot**

Run:

```powershell
python -m pytest project/tests/integration/test_server_boot_e2e.py -q
```

Expected: PASS.

- [ ] **Step 4: Manual browser smoke**

Start Brainfast:

```powershell
python -m project.frontend.server
```

Open:

```text
http://127.0.0.1:8787
```

Manual acceptance:

- Load a completed job in `3D Liquify`.
- Click `Reload slice list`.
- Confirm `Brush` is selected by default.
- Drag on the overlay and confirm a cyan stroke appears immediately.
- Confirm stroke count increments.
- Press `Ctrl+Z` and confirm the stroke disappears.
- Switch to `Landmark` mode and confirm the old atlas/real click-pair flow still works.
- Click `Clear all pairs` and confirm both pair count and stroke count become zero.
- Add one brush stroke and click `Apply 3D warp`.
- Confirm `/api/liquify-3d/progress` reaches `done`.
- Click `Finalize & re-export cell counts`.

- [ ] **Step 5: Check git diff**

Run:

```powershell
git diff --stat
git diff -- project/frontend/app.js project/frontend/index.html project/frontend/styles.css project/frontend/blueprints/api_liquify_3d.py project/scripts/liquify_3d.py project/scripts/class_prior.py
```

Expected:

- Only Liquify-related files changed.
- No unrelated release/test report files staged.
- No generated runtime outputs staged.

- [ ] **Step 6: Final commit**

If previous tasks were committed individually, skip this step. If execution was done as one batch, commit:

```powershell
git add project/frontend/app.js project/frontend/index.html project/frontend/styles.css project/frontend/blueprints/api_liquify_3d.py project/scripts/liquify_3d.py project/scripts/class_prior.py project/tests/unit/test_liquify_3d.py project/tests/unit/test_api_liquify_3d.py project/tests/unit/test_frontend_regressions.py project/tests/unit/test_class_prior.py
git commit -m "feat(liquify): add ps-style brush workflow"
```

---

## Rollout Notes

The MVP intentionally stores brush strokes but still feeds the existing Laplacian solver through derived landmark pairs. This keeps the scientific output path stable:

- `annotation_refined_liquify3d.nii.gz` remains the refined annotation artifact.
- `Finalize` still produces `cell_counts_hierarchy_liquify3d.csv`.
- Existing jobs with only `landmarks_3d.csv` remain valid.
- Existing class priors remain valid.

The next iteration after this plan should introduce a true stroke-prior format:

- Store averaged vector-field hints per class.
- Show prior hints visually as faint arrows on the canvas.
- Let the user accept, adjust, or delete prior-suggested strokes before applying.
