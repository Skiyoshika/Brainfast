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
