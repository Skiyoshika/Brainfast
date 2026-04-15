# Cellpose Full GUI Integration Design

**Date:** 2026-04-13
**Status:** Approved
**Purpose:** Embed Cellpose's annotation, training, and model management capabilities
directly into Brainfast's single-page application, enabling a one-stop workflow from
cell detection through mask correction to custom model training.

---

## 1. Problem Statement

Brainfast currently uses Cellpose as a pure inference engine — it calls `model.eval()`
with hardcoded config parameters and discards everything except centroids. Users cannot:

- Adjust detection parameters (diameter, thresholds) without editing JSON
- Correct detection errors (missed cells, false positives, bad boundaries)
- Accumulate corrected masks as training data
- Fine-tune Cellpose models on their own tissue samples
- Use custom-trained models in the pipeline

The built-in `cpsam` model fails systematically on cleared-tissue fluorescence samples
(tensor dimension mismatch), forcing fallback to the LoG detector. The only path to
better detection is training a custom model, which currently requires leaving Brainfast
entirely and using Cellpose's standalone GUI.

## 2. User Workflow

```
Select slice → Run inference with adjustable params → Edit masks (correct errors)
→ Save corrected masks to training set → Repeat until enough data accumulated
→ One-click train → New model auto-applies to future runs
```

The user's goal is to accumulate enough high-quality training data over normal usage
to train one universal model that works across all their sample types (ChATe27, PVe3,
different markers). Once the model is good enough, training stops permanently.

## 3. Subsystem Decomposition

Four subsystems, delivered in order. Each is independently useful upon delivery.

| # | Subsystem | Solves | Depends On |
|---|-----------|--------|------------|
| 1 | Parameter Panel | No more JSON editing for detection params | Independent |
| 2 | Mask Annotation Canvas | Correct detection errors, produce training data | Subsystem 1 |
| 3 | Training Set Management + Training | Accumulate data, train custom model | Subsystem 2 |
| 4 | Model Auto-Apply | Trained model seamlessly replaces default | Subsystem 3 |

---

## 4. Subsystem 1: Parameter Panel

### 4.1 What It Does

Adds interactive controls to the detection preview area so users can adjust Cellpose
parameters and see results immediately, without editing `run_config*.json`.

### 4.2 Frontend Components

Location: Below the existing "Detect Cells" button in Step 2 (Atlas Preview).

| Control | Type | Range | Default | Maps To |
|---------|------|-------|---------|---------|
| Model | Dropdown | Available models from API | `cpsam` | `detection.primary_model` |
| Diameter (µm) | Number input + slider | 1–100 | 12.0 | `detection.cellpose_diameter_um` |
| Flow Threshold | Slider | 0.0–1.0 | 0.4 | `detection.cellpose_flow_threshold` |
| Cell Probability | Slider | -6.0–6.0 | 0.0 | `detection.cellpose_cellprob_threshold` |
| Min Size (px) | Number input | 1–500 | 8 | `detection.cellpose_min_size_px` |
| GPU | Toggle | on/off | on | `detection.cellpose_gpu` |

### 4.3 Backend Changes

**New endpoint:** `GET /api/cellpose/models`
- Returns list of available models: built-in (`cpsam`, `cyto3`, `cyto2`, `nuclei`)
  plus user-trained models from `cellpose.models.get_user_models()`
- Response: `{"models": [{"name": "cpsam", "type": "builtin"}, {"name": "brainfast_v1_20260413", "type": "custom"}]}`

**Modified endpoint:** `POST /api/detect/preview`
- Accept optional parameter overrides in request body:
  ```json
  {
    "slicePath": "...",
    "params": {
      "model": "cyto3",
      "diameter_um": 15.0,
      "flow_threshold": 0.5,
      "cellprob_threshold": -1.0,
      "min_size_px": 10
    }
  }
  ```
- When `params` is present, merge with config (params take precedence)
- Return full masks array (not just centroids) when `returnMasks=true`

**Modified:** `detect.py`
- `detect_cells_cellpose()`: Capture and optionally return the full masks array
  alongside the centroid DataFrame
- `_load_cellpose_model()`: Support loading custom model paths via
  `CellposeModel(pretrained_model="path/to/model")` when model name is not
  in the built-in list

### 4.4 i18n

All new labels need entries in both `LANGS.en` and `LANGS.zh` in `app.js`.

---

## 5. Subsystem 2: Mask Annotation Canvas

### 5.1 What It Does

A dedicated mask editing interface where users correct Cellpose inference results.
The corrected masks become training data for fine-tuning.

### 5.2 Layout: Side Panel (Photoshop Style)

Enters via "Edit Masks" button after detection preview completes.

```
┌─────────────────────────────────────────────────────────────┐
│ [Toolbar]  🖌 🧹 ➕ 🗑 🔗 ✂️  |  ✋ 🔍  │ [Close] [Save] │
├────┬───────────────────────────────────────────┬────────────┤
│    │                                           │ Brush Size │
│ T  │                                           │ ═══●═════  │
│ o  │                                           │            │
│ o  │         Canvas                             │ Opacity    │
│ l  │    (original image + colored mask overlay) │ ═══════●═  │
│ s  │                                           │            │
│    │                                           │ ────────── │
│    │                                           │ Cell List  │
│    │                                           │ ● Cell 1   │
│    │                                           │ ● Cell 2 ▸ │
│    │                                           │ ● Cell 3   │
├────┴───────────────────────────────────────────┴────────────┤
│ Brush: 10px | Cell #2 selected | Zoom: 100% | 42 cells     │
└─────────────────────────────────────────────────────────────┘
```

### 5.3 Canvas Architecture

**Two-layer canvas stack** (both HTML5 Canvas 2D contexts):

1. **Background layer** — Original microscope image (read-only)
2. **Mask layer** — Instance segmentation mask with per-cell colors, semi-transparent

The mask layer is a pixel array where each pixel value is a cell ID (0 = background,
1..N = cell instances). Rendering uses a color LUT to map IDs to distinct colors.

**Data format in memory:**
```javascript
{
  imageData: Uint16Array,      // original image pixels (grayscale)
  maskData: Int32Array,        // instance mask (0=bg, 1..N=cell IDs)
  nextCellId: N+1,             // next ID to assign
  colorLUT: Map<int, string>,  // cell ID → CSS color
  selectedCell: int|null,      // currently selected cell ID
  brushSize: int,              // brush radius in pixels
  currentTool: string,         // 'brush'|'eraser'|'add'|'delete'|'merge'|'split'|'pan'|'zoom'
}
```

### 5.4 Tools

| Tool | Interaction | Effect on maskData |
|------|-------------|-------------------|
| **Brush** 🖌 | Paint on canvas | Set pixels to `selectedCell` ID within brush radius |
| **Eraser** 🧹 | Paint on canvas | Set pixels to 0 (background) within brush radius |
| **Add Cell** ➕ | Click, then paint | Assign `nextCellId++`, paint new cell region |
| **Delete Cell** 🗑 | Click on a cell | Set all pixels with that cell ID to 0 |
| **Merge** 🔗 | Click cell A, then cell B | Replace all B pixels with A's ID |
| **Split** ✂️ | Draw a line through a cell | Pixels on each side become separate cell IDs (connected components) |
| **Pan** ✋ | Click-drag | Translate viewport |
| **Zoom** 🔍 | Scroll wheel or pinch | Scale viewport |

**Keyboard shortcuts** (matching Cellpose GUI conventions):
- `B` → Brush, `E` → Eraser, `N` → New cell (Add), `D` → Delete
- `[` / `]` → Decrease/increase brush size
- `Z` → Undo, `Shift+Z` → Redo
- `Space+drag` → Pan (temporary override)
- Scroll wheel → Zoom

### 5.5 Undo/Redo

Maintain a stack of mask snapshots (or diffs for memory efficiency).
Maximum undo depth: 20 steps.

For memory efficiency with large masks (e.g., 1500x1500 = 9MB per Int32 snapshot):
- Store diffs as sparse arrays: `{pixels: [{idx, oldVal, newVal}, ...]}` per action
- Only store full snapshots every 5th action as checkpoints

### 5.6 Save to Training Set

"Save to Training Set" button:

1. Backend receives: original image path + corrected mask (Int32 array)
2. Saves to `project/cellpose_training/`:
   - `{slice_name}.tif` — original image (copy or symlink)
   - `{slice_name}_masks.tif` — corrected instance mask as 16-bit TIFF
3. This follows Cellpose's native `_masks` suffix convention, so the training
   directory can be passed directly to `io.load_train_test_data()`
4. Returns confirmation with updated training set statistics

**New endpoint:** `POST /api/cellpose/save-training-sample`
```json
{
  "imagePath": "data/35_C0_demo/z0050.tif",
  "mask": [0,0,0,1,1,1,2,2,...],  // flattened Int32 array
  "width": 1500,
  "height": 1500
}
```

Response:
```json
{
  "ok": true,
  "savedAs": "cellpose_training/z0050.tif",
  "trainingSetStats": {
    "totalImages": 12,
    "totalCells": 487,
    "avgCellsPerImage": 40.6
  }
}
```

### 5.7 Large Image Handling

Microscope images can be 1500x1500 or larger. Strategies:

- **Server-side downscale** for canvas display (max 1024px on longest edge)
- **Mask editing at display resolution**, then upscale mask back to original resolution
  on save using nearest-neighbor interpolation
- **Tile-based loading** deferred to future enhancement if needed

---

## 6. Subsystem 3: Training Set Management + Training

### 6.1 Training Set Panel

Accessible from a new "Training" tab or section in the sidebar.

**View:** Thumbnail grid of all images in `project/cellpose_training/`.
Each thumbnail shows:
- The original image with mask overlay
- Cell count badge
- Delete button (remove from training set)

**Statistics bar:**
- Total images: N
- Total annotated cells: M
- Average cells/image: M/N
- Readiness indicator: "Ready to train" (if N >= 5) or "Need N more images"

### 6.2 Training Configuration

Minimal — the user should not need to adjust hyperparameters.

| Parameter | Default | Exposed? |
|-----------|---------|----------|
| Base model | Current active model (e.g., `cyto3`) | Yes — dropdown |
| Epochs | 100 | No (hardcoded) |
| Learning rate | 1e-5 | No (hardcoded) |
| Weight decay | 0.1 | No (hardcoded) |
| Batch size | 1 | No (hardcoded) |
| Model name | `brainfast_{timestamp}` | Yes — text input |

### 6.3 Training Execution

**New endpoint:** `POST /api/cellpose/train`
```json
{
  "baseModel": "cyto3",
  "modelName": "brainfast_v1",
  "trainingDir": "cellpose_training"
}
```

**Backend flow:**
1. Validate training directory has enough data (`min_train_masks=5`)
2. Load base model: `CellposeModel(model_type=baseModel, gpu=True)`
3. Load training data: `io.load_train_test_data(train_dir)` with 80/20 train/test split
4. Run in background thread:
   ```python
   model_path, train_losses, test_losses = train_seg(
       net=model.net,
       train_data=train_data,
       train_labels=train_labels,
       test_data=test_data,
       test_labels=test_labels,
       n_epochs=100,
       learning_rate=1e-5,
       weight_decay=0.1,
       save_path=str(PROJECT_ROOT / "cellpose_training" / "models"),
       model_name=model_name,
   )
   io.add_model(model_path)  # Register in ~/.cellpose/models/
   ```
5. During training, write progress to shared state (epoch, train_loss, test_loss)
6. On completion, store result summary

**Progress endpoint:** `GET /api/cellpose/train-status`
```json
{
  "state": "training",
  "epoch": 45,
  "totalEpochs": 100,
  "trainLoss": 0.234,
  "testLoss": 0.289,
  "lossHistory": [[0, 1.2, 1.3], [1, 0.9, 1.0], ...],
  "estimatedTimeRemaining": "3m 20s"
}
```

### 6.4 Training Progress UI

- Progress bar: epoch N / 100
- Real-time loss chart (train loss + test loss lines, updated via polling every 2s)
- "Cancel Training" button
- On completion: summary card with final loss, improvement %, model file path
- "Apply Model" button (→ Subsystem 4)

---

## 7. Subsystem 4: Model Auto-Apply

### 7.1 What It Does

After training completes, the new model becomes the default for all future
detection runs without manual config editing.

### 7.2 Mechanism

1. Training completes → model saved as `brainfast_{name}_{timestamp}` in
   `~/.cellpose/models/`
2. Backend updates `run_config.template.json` (or active config):
   `detection.primary_model` → new model name
3. Model cache in `detect.py` is invalidated (clear `_CELLPOSE_MODEL_CACHE`)
4. Next detection preview or pipeline run automatically uses the new model

### 7.3 Comparison View (Optional)

After training, automatically run inference on one annotated image with both
old and new model. Show side-by-side:
- Old model: N cells detected, overlay image
- New model: M cells detected, overlay image
- Ground truth: K cells annotated

This gives immediate visual feedback on whether training helped.

**New endpoint:** `POST /api/cellpose/compare-models`
```json
{
  "imagePath": "cellpose_training/z0050.tif",
  "modelA": "cyto3",
  "modelB": "brainfast_v1_20260413"
}
```

---

## 8. Data Flow Summary

```
                    ┌─────────────┐
                    │ Microscope  │
                    │   Image     │
                    └──────┬──────┘
                           │
                    ┌──────▼──────┐
                    │ Subsystem 1 │
                    │  Parameter  │
                    │   Panel     │
                    │ (adjust &   │
                    │  preview)   │
                    └──────┬──────┘
                           │ masks + centroids
                    ┌──────▼──────┐
                    │ Subsystem 2 │
                    │   Mask      │
                    │ Annotation  │◄──── User corrects
                    │  Canvas     │
                    └──────┬──────┘
                           │ image.tif + image_masks.tif
                    ┌──────▼──────┐
                    │ Subsystem 3 │
                    │  Training   │
                    │   Set +     │
                    │  train_seg  │
                    └──────┬──────┘
                           │ custom_model.npy
                    ┌──────▼──────┐
                    │ Subsystem 4 │
                    │ Auto-Apply  │──► Pipeline uses new model
                    └─────────────┘
```

## 9. File Layout

```
project/
  cellpose_training/           # NEW — training data directory
    z0050.tif                  # original image
    z0050_masks.tif            # corrected instance mask
    z0075.tif
    z0075_masks.tif
    ...
    models/                    # trained model checkpoints
      brainfast_v1_20260413
  frontend/
    blueprints/
      api_cellpose.py          # NEW — all Cellpose API endpoints
    app.js                     # MODIFIED — add parameter panel, annotation canvas,
                               #            training UI components
    index.html                 # MODIFIED — add HTML structure for new panels
    styles.css                 # MODIFIED — annotation canvas styles
  scripts/
    detect.py                  # MODIFIED — return full masks, support custom models
    cellpose_trainer.py        # NEW — training wrapper (background thread)
  configs/
    run_config.template.json   # MODIFIED — primary_model updated after training
```

## 10. Technology Choices

| Concern | Choice | Rationale |
|---------|--------|-----------|
| Canvas rendering | HTML5 Canvas 2D | Already used for liquify/annotations in Brainfast |
| Mask transfer (server↔client) | Base64-encoded PNG (8-bit) or raw Int32 binary | PNG for display, binary for lossless mask editing |
| Training progress | HTTP polling (2s interval) | Simpler than WebSocket; training is minutes-long |
| Model storage | `~/.cellpose/models/` via `io.add_model()` | Native Cellpose convention; models available to CLI too |
| Training data format | `{name}.tif` + `{name}_masks.tif` | Native Cellpose convention; compatible with `io.load_train_test_data()` |

## 11. Scope Exclusions

The following are explicitly out of scope for this design:

- **3D annotation** — Only 2D slice-by-slice annotation. 3D volumetric annotation is
  a separate project.
- **Multi-user training** — Single-user, single-model workflow. No concurrent training
  sessions or model versioning beyond timestamp naming.
- **Flow field visualization** — Cellpose's gradient flow outputs are not displayed.
  Could be added as a future enhancement.
- **Cellpose CLI integration** — No command-line training interface. All training
  happens through the web UI.
- **Model architecture changes** — Only fine-tuning existing Cellpose architectures
  (CellposeModel). No custom network architectures.
- **Advanced augmentation config** — Uses Cellpose's built-in `random_rotate_and_resize`
  augmentation. No user-configurable augmentation pipeline.

## 12. Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Large mask arrays crash browser | High | Server-side downscale to max 1024px; edit at display resolution |
| Training fails silently | Medium | Wrap `train_seg()` in try/except; surface errors in UI |
| cpsam base model incompatible with fine-tuning | Medium | Default to `cyto3` as base model; test cpsam fine-tuning separately |
| Mask transfer latency | Low | Use binary encoding for large masks; compress with zlib |
| Undo stack memory | Low | Sparse diff storage; cap at 20 steps with periodic checkpoints |

## 13. Cellpose API Reference (v4.1.1, verified)

Verified against the installed `cellpose==4.1.1` package on 2026-04-13.

### Training
```python
from cellpose.train import train_seg
model_path, train_losses, test_losses = train_seg(
    net,                          # CellposeModel.net (the torch network)
    train_data=list[np.ndarray],  # images
    train_labels=list[np.ndarray],# instance masks (0=bg, 1..N=cells)
    test_data=list[np.ndarray],
    test_labels=list[np.ndarray],
    n_epochs=100,
    learning_rate=1e-5,
    weight_decay=0.1,
    batch_size=1,
    save_path="path/to/save",
    model_name="my_model",
    min_train_masks=5,            # skip images with fewer masks
)
```

### Data Loading
```python
from cellpose import io
# Convention: image.tif + image_masks.tif in same directory
images, labels, names = io.load_images_labels(
    tdir="training_dir/",
    mask_filter="_masks",         # suffix for mask files
)
```

### Model Registration
```python
from cellpose import io
io.add_model("path/to/trained_model")  # copies to ~/.cellpose/models/ + updates MODEL_LIST_PATH
```

### User Model Discovery
```python
from cellpose.models import get_user_models
custom_models = get_user_models()  # returns list of registered custom model names
```

### Model Loading (Custom)
```python
from cellpose.models import CellposeModel
model = CellposeModel(pretrained_model="path/to/model_or_name", gpu=True)
masks, flows, styles = model.eval(image, diameter=diameter)
```
