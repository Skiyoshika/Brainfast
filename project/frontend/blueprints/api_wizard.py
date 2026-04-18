"""Onboarding wizard endpoints.

Two endpoints — ``/api/wizard/inspect-source`` and ``/api/wizard/launch`` —
let a brand-new user go from "I have raw TIFFs" to "the pipeline is
running" entirely from the UI, without hand-editing config JSON or
shelling out to ``python scripts/main.py``.

Inspection:
    Reads ImageJ metadata + page count from a single multi-page TIFF, or
    counts files + reads sample shape from a directory of slice TIFFs, then
    returns sane suggested defaults for ``pixel_size_um_xy``,
    ``slice_spacing_um`` and ``sample_id``.

Launch:
    Composes a payload understood by the existing ``/api/run`` endpoint and
    invokes the same code path the rest of the UI uses, so wizard-launched
    runs share status reporting, progress tracking, and error handling
    with hand-built configs. The wizard does not invent a parallel pipeline.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

from flask import Blueprint, jsonify, request

import project.frontend.server_context as ctx
from project.frontend.api_errors import ERR_INVALID_INPUT, ERR_NOT_FOUND

bp = Blueprint("api_wizard", __name__, url_prefix="/api")


# Allen-CCFv3 friendly defaults — match what the demo configs use so a new
# user without prior knowledge gets a working starting point. Override
# happens at the form level, never silently here.
_DEFAULT_PIXEL_UM_XY = 5.0
_DEFAULT_Z_SPACING_UM = 25.0
_DEFAULT_HEMISPHERE = "right_flipped"
_DEFAULT_TARGET_UM = 25.0


def _read_imagej_spacing(path: Path) -> tuple[float | None, float | None]:
    """Return ``(pixel_um_xy, z_spacing_um)`` parsed from a TIFF's ImageJ
    metadata, or ``(None, None)`` if either is unavailable.
    """
    from tifffile import TiffFile

    try:
        with TiffFile(str(path)) as tf:
            page = tf.pages[0]
            x_res = page.tags.get("XResolution")
            ij = getattr(tf, "imagej_metadata", None) or {}
            spacing = ij.get("spacing")
            unit = (ij.get("unit") or "").lower()

            pixel_um = None
            if x_res is not None:
                num, den = x_res.value
                if num and den:
                    # ImageJ stores resolution as pixels-per-unit; pixel size
                    # in that unit is den / num.
                    pixel_um = float(den) / float(num)

            z_um = None
            if spacing is not None:
                z_um = float(spacing)
                if unit and unit not in ("micron", "um", "µm"):
                    # We don't bother with full unit conversion here; the
                    # frontend lets the user adjust if the unit was odd.
                    z_um = float(spacing)

            return pixel_um, z_um
    except Exception:  # noqa: BLE001 — metadata is best-effort
        return None, None


@bp.post("/wizard/inspect-source")
def wizard_inspect_source():
    """Look at a TIFF file or directory and report what kind of input it
    is, plus suggested config defaults the wizard form will pre-fill.
    """
    payload = request.get_json(silent=True) or {}
    raw_path = (payload.get("sourcePath") or "").strip()
    if not raw_path:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "missing 'sourcePath'",
                    "error_code": ERR_INVALID_INPUT,
                }
            ),
            400,
        )

    src = Path(raw_path).expanduser()
    if not src.exists():
        return (
            jsonify(
                {
                    "ok": False,
                    "error": f"source path not found: {src}",
                    "error_code": ERR_NOT_FOUND,
                }
            ),
            404,
        )

    suggested_sample_id = src.stem if src.is_file() else src.name

    if src.is_file():
        # Treat as multi-page TIFF — read pages and metadata
        from tifffile import TiffFile

        try:
            with TiffFile(str(src)) as tf:
                n_pages = len(tf.pages)
                first = tf.pages[0]
                shape = list(first.shape)
                dtype = str(first.dtype)
        except Exception as exc:  # noqa: BLE001
            return (
                jsonify(
                    {
                        "ok": False,
                        "error": f"failed to read TIFF: {exc}",
                        "error_code": ERR_INVALID_INPUT,
                    }
                ),
                400,
            )
        ij_pixel_um, ij_z_um = _read_imagej_spacing(src)
        return jsonify(
            {
                "ok": True,
                "kind": "multipage_tiff",
                "n_pages": n_pages,
                "sample_shape": shape,
                "dtype": dtype,
                "needs_extraction": True,
                "suggested_sample_id": suggested_sample_id,
                "suggested_pixel_um_xy": ij_pixel_um or _DEFAULT_PIXEL_UM_XY,
                "suggested_z_spacing_um": ij_z_um or _DEFAULT_Z_SPACING_UM,
            }
        )

    # Directory of slice TIFFs
    if not src.is_dir():
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "source path is neither a file nor a directory",
                    "error_code": ERR_INVALID_INPUT,
                }
            ),
            400,
        )

    glob_pattern = str(payload.get("sliceGlob") or "*.tif")
    files = sorted(src.glob(glob_pattern))
    if not files:
        # Try a more permissive glob before giving up
        files = sorted([p for p in src.glob("*.tif*") if p.is_file()])

    sample_shape: list[int] | None = None
    sample_dtype: str | None = None
    if files:
        from tifffile import TiffFile

        try:
            with TiffFile(str(files[0])) as tf:
                first = tf.pages[0]
                sample_shape = list(first.shape)
                sample_dtype = str(first.dtype)
        except Exception:  # noqa: BLE001
            pass

    return jsonify(
        {
            "ok": True,
            "kind": "directory",
            "n_files": len(files),
            "sample_shape": sample_shape,
            "dtype": sample_dtype,
            "needs_extraction": False,
            "suggested_sample_id": suggested_sample_id,
            "suggested_pixel_um_xy": _DEFAULT_PIXEL_UM_XY,
            "suggested_z_spacing_um": _DEFAULT_Z_SPACING_UM,
            "suggested_slice_glob": glob_pattern,
        }
    )


def _build_run_config(payload: dict) -> dict:
    """Compose a Brainfast run-config dict from wizard form fields."""
    sample_id = str(payload.get("sampleId") or "sample").strip() or "sample"
    pixel_um = float(payload["pixelSizeUm"])
    z_um = float(payload["zSpacingUm"])
    channels_in = payload.get("channels") or ["red"]
    if isinstance(channels_in, str):
        channels_in = [channels_in]
    active_channel = str(channels_in[0]) if channels_in else "red"
    hemisphere = str(payload.get("atlasHemisphere", _DEFAULT_HEMISPHERE))
    slice_glob = str(payload.get("sliceGlob") or "z*.tif")

    return {
        "project": {"name": sample_id, "version": "wizard-1.0"},
        "input": {
            "slice_dir": str(payload["inputDir"]),
            "slice_glob": slice_glob,
            "sampling_mode": "single",
            "slice_interval_n": 1,
            "pixel_size_um_xy": pixel_um,
            "slice_spacing_um": z_um,
            "bit_depth": 16,
            "channel_map": {"red": 0, "green": 1, "farred": 2},
            "active_channel": active_channel,
        },
        "target": {"marker": "neurons", "signal_type": "cyto"},
        "compute": {"device": "auto", "vram_gb": 8},
        "registration": {
            "mode": "2d_slice_with_3d_smoothness",
            "scope": "whole",
            "whole_brain_backend": "miki_3d",
            "truth_source": "3d_registered_volume",
            "template_path": "configs/allen_ref_cache/average_template_25.nii.gz",
            "annotation_path": "annotation_25.nii.gz",
            "skip_laplacian_refinement": False,
            "ants_transform": "SyN",
            "random_seed": 42,
            "atlas_hemisphere": hemisphere,
            "atlas_z_from_filename": False,
            "atlas_z_z_scale": 1.0,
            "atlas_z_offset": 0,
            "atlas_z_range": [0, 528],
            "axis_alignment_enabled": False,
            "intensity_adapt": {
                "mode": "hist_match+clahe",
                "clahe_kernel_size": 32,
                "clahe_clip_limit": 0.01,
            },
            "ml_flip": False,
        },
        "detection": {
            "mode": "cellpose",
            "primary_model": "cpsam",
            "cellpose_gpu": True,
            "cellpose_diameter_um": 12.0,
            "cellpose_channels": [0, 0],
            "allow_fallback": False,
        },
        "dedup": {
            "enabled": True,
            "method": "kdtree",
            "neighbor_slices": 1,
            "r_xy_um": 8.0,
            "r_z_rule": "slice_spacing_um*0.5",
        },
        "outputs": {
            "leaf_csv": f"outputs/{sample_id}_leaf.csv",
            "hierarchy_csv": f"outputs/{sample_id}_hierarchy.csv",
            "qc_dir": f"outputs/{sample_id}_qc",
        },
    }


@bp.post("/wizard/launch")
def wizard_launch():
    """Generate a config file from wizard form fields and start the
    pipeline using the same threaded runner the existing ``/api/run``
    endpoint uses. Returns the assigned ``jobId`` so the frontend can
    redirect to the standard pipeline progress display.
    """
    payload = request.get_json(silent=True) or {}

    required = ("sampleId", "inputDir", "pixelSizeUm", "zSpacingUm")
    missing = [k for k in required if k not in payload or payload[k] in (None, "")]
    if missing:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": f"missing required field(s): {', '.join(missing)}",
                    "error_code": ERR_INVALID_INPUT,
                }
            ),
            400,
        )

    input_dir = Path(str(payload["inputDir"])).expanduser()
    if not input_dir.exists():
        return (
            jsonify(
                {
                    "ok": False,
                    "error": f"inputDir not found: {input_dir}",
                    "error_code": ERR_NOT_FOUND,
                }
            ),
            404,
        )

    cfg = _build_run_config(payload)
    sample_id = cfg["project"]["name"]
    job_id = sample_id

    # Persist the wizard-built config under the job's outputs dir so it is
    # discoverable next to the pipeline outputs (mirrors what the existing
    # /api/run endpoint does for hand-built configs).
    job_dir = ctx.OUTPUT_DIR / job_id
    runtime_dir = job_dir / "runtime_configs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_path = runtime_dir / f"run_config_{stamp}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    channels = payload.get("channels") or ["red"]
    if isinstance(channels, str):
        channels = [channels]

    # Reuse server_context._runner (same code path as /api/run) so progress
    # tracking and status endpoints continue to work without additional UI.
    t = threading.Thread(
        target=ctx._runner,
        args=(str(cfg_path), str(input_dir), list(channels), {}),
        kwargs={"job_id": job_id},
        daemon=True,
    )
    t.start()

    return jsonify(
        {
            "ok": True,
            "jobId": job_id,
            "config_path": str(cfg_path),
            "input_dir": str(input_dir),
            "channels": list(channels),
            "outputs_dir": str(job_dir),
        }
    )
