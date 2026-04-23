"""Sidecar metadata for `annotation_registered.nii.gz` et al.

The 3D whole-brain registration writes a sidecar JSON next to the registered
annotation so that downstream consumers (liquify finalize, re-quantify, etc.)
can recover the upstream ``annotation_sampling_mode`` without threading a
new parameter through every API.

Schema:

.. code-block:: json

    {
        "schema": "brainfast.annotation.sidecar/v1",
        "annotation_sampling_mode": "3d_reslice" | "per_slice_native"
    }

Missing sidecar → assume ``3d_reslice`` (backwards compatible with annotations
written before this sidecar was introduced).
"""

from __future__ import annotations

import json
from pathlib import Path

_SCHEMA_VERSION = "brainfast.annotation.sidecar/v1"
_DEFAULT_MODE = "3d_reslice"
_VALID_MODES = ("3d_reslice", "per_slice_native")


def sidecar_path_for(annotation_path: Path | str) -> Path:
    """Return the sidecar JSON path paired with an annotation NIfTI."""
    p = Path(annotation_path)
    return p.with_suffix(".meta.json") if p.suffix != ".json" else p


def _sidecar_path(annotation_path: Path) -> Path:
    # annotation_registered.nii.gz → annotation_registered.meta.json
    # annotation_registered.nii    → annotation_registered.meta.json
    name = annotation_path.name
    if name.endswith(".nii.gz"):
        stem = name[: -len(".nii.gz")]
    elif name.endswith(".nii"):
        stem = name[: -len(".nii")]
    else:
        stem = annotation_path.stem
    return annotation_path.with_name(f"{stem}.meta.json")


def write_annotation_sidecar(
    annotation_path: Path | str,
    annotation_sampling_mode: str,
) -> Path:
    """Write ``annotation_registered.meta.json`` next to the NIfTI."""
    mode = str(annotation_sampling_mode or _DEFAULT_MODE).strip().lower()
    if mode not in _VALID_MODES:
        mode = _DEFAULT_MODE
    out = _sidecar_path(Path(annotation_path))
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"schema": _SCHEMA_VERSION, "annotation_sampling_mode": mode}
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def read_annotation_sampling_mode(
    annotation_path: Path | str,
) -> str:
    """Return the sampling mode recorded alongside the annotation, or the default."""
    p = _sidecar_path(Path(annotation_path))
    if not p.exists():
        return _DEFAULT_MODE
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _DEFAULT_MODE
    mode = str(data.get("annotation_sampling_mode", _DEFAULT_MODE)).strip().lower()
    return mode if mode in _VALID_MODES else _DEFAULT_MODE


def annotation_prewarped_for_mode(mode: str) -> bool:
    """Translate sampling mode → ``render_overlay(prewarped_label=...)`` flag.

    * ``3d_reslice`` → label is already in sample space → ``prewarped=True``
    * ``per_slice_native`` → label is at CCF native Y×X → ``prewarped=False``
      so the per-slice tissue-guided 2D warp in ``overlay_render`` aligns it.
    """
    return str(mode or _DEFAULT_MODE).strip().lower() != "per_slice_native"
