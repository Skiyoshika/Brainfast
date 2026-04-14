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
