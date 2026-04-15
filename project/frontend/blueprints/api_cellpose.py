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

    return jsonify(
        {
            "ok": True,
            "savedAs": f"cellpose_training/{stem}.tif",
            "trainingSetStats": stats,
        }
    )


@bp.post("/train")
def start_training():
    """Start Cellpose model training in a background thread.

    Request JSON:
        baseModel: str — base model to fine-tune (default: "cyto3")
        modelName: str — name for the trained model (default: auto-generated)
        epochs: int — number of training epochs (default: 100)
        gpu: bool — use GPU (default: true)
    """
    from project.scripts.cellpose_trainer import TrainingError, get_trainer

    payload = request.get_json(force=True)
    base_model = payload.get("baseModel", "cyto3")
    model_name = payload.get("modelName", "")
    epochs = int(payload.get("epochs", 100))
    use_gpu = bool(payload.get("gpu", True))

    if not model_name:
        import time as _time

        model_name = f"brainfast_{int(_time.time())}"

    td = _training_dir()
    stats = _training_set_stats(td)

    if stats["totalImages"] < 2:
        return jsonify(
            {
                "ok": False,
                "error": f"Need at least 2 training images, have {stats['totalImages']}",
            }
        ), 400

    try:
        trainer = get_trainer()
        trainer.start(
            training_dir=td,
            base_model=base_model,
            model_name=model_name,
            n_epochs=epochs,
            use_gpu=use_gpu,
        )
    except TrainingError as exc:
        return jsonify({"ok": False, "error": str(exc)}), 409

    return jsonify(
        {
            "ok": True,
            "modelName": model_name,
            "epochs": epochs,
            "trainingImages": stats["totalImages"],
        }
    )


@bp.get("/train-status")
def train_status():
    """Return current training status."""
    from project.scripts.cellpose_trainer import get_trainer

    trainer = get_trainer()
    return jsonify({"ok": True, **trainer.get_status()})


@bp.post("/train-cancel")
def train_cancel():
    """Cancel the current training run."""
    from project.scripts.cellpose_trainer import get_trainer

    trainer = get_trainer()
    trainer.cancel()
    return jsonify({"ok": True, "status": "cancelled"})


@bp.get("/training-set")
def training_set_info():
    """Return info about the current training set.

    Returns list of training samples with stats.
    """
    td = _training_dir()
    stats = _training_set_stats(td)

    samples = []
    mask_files = sorted(td.glob("*_masks.tif"))
    for mf in mask_files:
        stem = mf.name.replace("_masks.tif", "")
        img_path = td / f"{stem}.tif"
        try:
            m = imread(str(mf))
            cell_count = int(m.max())
        except Exception:
            cell_count = 0
        samples.append(
            {
                "name": stem,
                "imageExists": img_path.exists(),
                "cellCount": cell_count,
            }
        )

    return jsonify(
        {
            "ok": True,
            "stats": stats,
            "samples": samples,
            "ready": stats["totalImages"] >= 2,
        }
    )


@bp.post("/apply-model")
def apply_model():
    """Set a model as the active detection model.

    Request JSON:
        modelName: str — name of the model to apply

    Updates the active config and clears the model cache.
    """
    from project.scripts.cellpose_trainer import _clear_model_cache, _update_config_primary_model

    payload = request.get_json(force=True)
    model_name = payload.get("modelName", "")
    if not model_name:
        return jsonify({"ok": False, "error": "modelName is required"}), 400

    _update_config_primary_model(model_name)
    _clear_model_cache()

    return jsonify({"ok": True, "appliedModel": model_name})


@bp.delete("/training-set/<name>")
def delete_training_sample(name: str):
    """Delete a training sample by name."""
    td = _training_dir()
    img_path = td / f"{name}.tif"
    mask_path = td / f"{name}_masks.tif"

    deleted = False
    if img_path.exists():
        img_path.unlink()
        deleted = True
    if mask_path.exists():
        mask_path.unlink()
        deleted = True

    if not deleted:
        return jsonify({"ok": False, "error": f"Sample '{name}' not found"}), 404

    stats = _training_set_stats(td)
    return jsonify({"ok": True, "trainingSetStats": stats})
