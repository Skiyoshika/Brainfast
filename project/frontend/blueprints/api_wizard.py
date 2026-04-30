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


@bp.post("/wizard/extract-multipage-tiff")
def wizard_extract_multipage_tiff():
    """Split a multi-page TIFF into per-slice TIFFs so the wizard can launch.

    Body: ``{"src": "<path/to/multipage.tif>", "outDir": "<dst dir>",
              "channel": 0, "everyN": 1}``.

    The wizard's existing logic only enables the launch button when the
    source path resolves to a directory of single-page TIFFs.  Until this
    endpoint existed, users with a multi-page TIFF input had to drop to a
    terminal and run ``python scripts/extract_zstack.py`` by hand.  Now the
    UI can call this directly and re-Inspect the output dir.

    Synchronous on purpose — typical extracts are 600-1000 pages and run in
    seconds; if a future input is large enough to need progress streaming we
    can promote to a job-tracked async run.
    """
    payload = request.get_json(silent=True) or {}
    raw_src = (payload.get("src") or "").strip()
    raw_out = (payload.get("outDir") or "").strip()
    if not raw_src or not raw_out:
        return jsonify(
            {"ok": False, "error": "missing src/outDir", "error_code": ERR_INVALID_INPUT}
        ), 400

    src = Path(raw_src).expanduser()
    out = Path(raw_out).expanduser()
    if not src.exists() or not src.is_file():
        return jsonify(
            {
                "ok": False,
                "error": f"src not found or not a file: {src}",
                "error_code": ERR_NOT_FOUND,
            }
        ), 404

    channel = int(payload.get("channel", 0) or 0)
    every_n = int(payload.get("everyN", 1) or 1)

    try:
        from project.scripts.extract_zstack import extract_zstack

        written = extract_zstack(
            src=src,
            out_dir=out,
            channel=channel,
            every_n=every_n,
            z_min=0,
            z_max=-1,
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": f"extract failed: {exc}"}), 500

    return jsonify(
        {
            "ok": True,
            "src": str(src),
            "outDir": str(out),
            "channel": channel,
            "everyN": every_n,
            # extract_zstack returns the list of written paths; we surface
            # just the count for the UI.  Tolerate both ints and lists in case
            # the underlying contract changes.
            "writtenCount": (
                len(written) if hasattr(written, "__len__") else int(written or 0)
            ),
        }
    )


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

    # Xu Lab parity options (advanced wizard fields).
    # All default to "Brainfast historical" behaviour so existing wizard
    # callers with no advanced fields still work unchanged.
    ants_transform = str(payload.get("antsTransform") or "SyNRA")
    axis_alignment_enabled = bool(payload.get("axisAlignmentEnabled", False))
    use_cell_to_ccf_mapping = bool(payload.get("useCellToCcfMapping", False))
    fixed_max_dim_raw = payload.get("fixedMaxDim")
    try:
        fixed_max_dim = (
            int(fixed_max_dim_raw) if fixed_max_dim_raw not in (None, "", 0, "0") else None
        )
    except (TypeError, ValueError):
        fixed_max_dim = None

    reg_block: dict[str, object] = {
        "mode": "2d_slice_with_3d_smoothness",
        "scope": "whole",
        "whole_brain_backend": "miki_3d",
        "truth_source": "3d_registered_volume",
        "template_path": "configs/allen_ref_cache/average_template_25.nii.gz",
        "annotation_path": "annotation_25.nii.gz",
        "skip_laplacian_refinement": False,
        "ants_transform": ants_transform,
        "random_seed": 42,
        "atlas_hemisphere": hemisphere,
        "atlas_z_from_filename": False,
        "atlas_z_z_scale": 1.0,
        "atlas_z_offset": 0,
        "atlas_z_range": [0, 528],
        "axis_alignment_enabled": axis_alignment_enabled,
        "use_cell_to_ccf_mapping": use_cell_to_ccf_mapping,
        "intensity_adapt": {
            "mode": "hist_match+clahe",
            "clahe_kernel_size": 32,
            "clahe_clip_limit": 0.01,
        },
        "ml_flip": False,
    }
    if fixed_max_dim is not None:
        reg_block["fixed_max_dim"] = fixed_max_dim

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
        "registration": reg_block,
        "detection": {
            "mode": "cellpose",
            "primary_model": "cpsam",
            "cellpose_gpu": True,
            "cellpose_diameter_um": 12.0,
            "cellpose_channels": [0, 0],
            # Fallback ON by default so new users aren't dead-ended by
            # Cellpose OOM on large section sizes (cpsam tiles a 1636×1359
            # slice into 952 × 3 × 730 × 730 float32 = 5.67 GiB per slice
            # at batch size 1). The LoG fallback is much lighter and
            # produces usable cell counts even on low-VRAM machines.
            # Strict pipelines should override this to False explicitly.
            "allow_fallback": True,
            "fallback_model": "log",
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

    # Single-channel callers still send inputDir (str). Dual-channel callers
    # send inputDirs (dict {channel_name: dir}). If only inputDirs is given
    # we synthesize inputDir for the legacy required-field validation.
    input_dirs_payload = payload.get("inputDirs")
    if isinstance(input_dirs_payload, dict) and input_dirs_payload and not payload.get("inputDir"):
        # Pick the dir for the first declared channel as the primary
        primary_ch = (payload.get("channels") or [next(iter(input_dirs_payload))])[0]
        payload["inputDir"] = input_dirs_payload.get(primary_ch)

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

    # Validate every dir when multi-channel. Fail loudly before spawning work
    # so the user gets a fast, actionable error instead of a mid-pipeline crash.
    per_channel_dirs: dict[str, str] = {}
    if isinstance(input_dirs_payload, dict) and input_dirs_payload:
        for ch_name, ch_dir_raw in input_dirs_payload.items():
            ch_dir = Path(str(ch_dir_raw)).expanduser()
            if not ch_dir.exists():
                return (
                    jsonify(
                        {
                            "ok": False,
                            "error": f"inputDirs[{ch_name}] not found: {ch_dir}",
                            "error_code": ERR_NOT_FOUND,
                        }
                    ),
                    404,
                )
            per_channel_dirs[str(ch_name)] = str(ch_dir)

    cfg = _build_run_config(payload)
    sample_id = cfg["project"]["name"]
    job_id = sample_id

    # Persist the wizard-built config under the SAME outputs dir where
    # ctx._runner will write the pipeline artifacts. Previously we wrote
    # to outputs/<id>/ while the runner wrote to outputs/jobs/<id>/ —
    # an inconsistency that made the wizard's returned outputs_dir a
    # dead pointer. Use ctx._job_output_dir here so the config, progress
    # file, and pipeline artifacts all live together.
    job_dir = ctx._job_output_dir(job_id)
    runtime_dir = job_dir / "runtime_configs"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    cfg_path = runtime_dir / f"run_config_{stamp}.json"
    cfg_path.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")

    channels = payload.get("channels") or ["red"]
    if isinstance(channels, str):
        channels = [channels]

    # Runner accepts either a single str or a dict keyed by channel name.
    # Use dict form when we have per-channel dirs (covers dual-channel runs);
    # otherwise the single inputDir keeps backwards compatibility with every
    # existing wizard test + the single-channel happy path.
    runner_input: object = str(input_dir)
    if per_channel_dirs and len(channels) > 1:
        runner_input = per_channel_dirs

    # Reuse server_context._runner (same code path as /api/run) so progress
    # tracking and status endpoints continue to work without additional UI.
    t = threading.Thread(
        target=ctx._runner,
        args=(str(cfg_path), runner_input, list(channels), {}),
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
            "input_dirs": per_channel_dirs or {channels[0]: str(input_dir)},
            "channels": list(channels),
            "outputs_dir": str(job_dir),
        }
    )
