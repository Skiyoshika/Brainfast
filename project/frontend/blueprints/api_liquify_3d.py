"""api_liquify_3d.py — Phase β 3D landmark liquify REST endpoints.

The frontend picks (real voxel, atlas voxel) pairs in a 3D slice viewer and
POSTs them here. The backend stores them per-job, and on demand solves a
sparse-landmark Laplacian warp that refines the registered annotation.

All endpoints are scoped under ``/api/liquify-3d/...`` and take the usual
``jobId`` query param or JSON field. See
``docs/superpowers/plans/2026-04-16-internal-alignment-closed-loop-plan.md``
(Phase β) for the architectural context.
"""

from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, request

import project.frontend.server_context as ctx
from project.frontend.api_errors import ERR_INVALID_INPUT, ERR_NOT_FOUND

try:
    from project.scripts.class_prior import (
        MIN_SAMPLES_FOR_APPLY,
        ClassPriorStore,
        load_prior,
    )
    from project.scripts.class_registry import (
        detect_class_for_sample,
        list_known_classes,
    )
    from project.scripts.liquify_3d import (
        LandmarkStore,
        refine_annotation_with_landmarks,
    )
    from project.scripts.liquify_3d_finalize import finalize_liquify_to_cell_counts
    from project.scripts.liquify_progress import (
        clear_liquify_progress,
        read_liquify_progress,
        write_liquify_progress,
    )
except ImportError:
    from class_prior import MIN_SAMPLES_FOR_APPLY, ClassPriorStore, load_prior
    from class_registry import detect_class_for_sample, list_known_classes
    from liquify_progress import (
        clear_liquify_progress,
        read_liquify_progress,
        write_liquify_progress,
    )
    from scripts.liquify_3d import LandmarkStore, refine_annotation_with_landmarks
    from scripts.liquify_3d_finalize import finalize_liquify_to_cell_counts


def _class_registry_path() -> Path:
    return ctx.PROJECT_ROOT / "configs" / "sample_class_registry.json"


def _progress_cb_for_job(job_id: str):
    """Build a progress_cb closure that writes to the job's liquify progress."""
    job_dir = ctx._job_output_dir(job_id)

    def _cb(stage, idx, total, percent, message):
        write_liquify_progress(
            job_dir=job_dir,
            stage=str(stage),
            stage_index=int(idx),
            stage_count=int(total),
            percent=int(max(0, min(100, percent))),
            message=str(message),
        )
    return _cb


def _class_priors_root() -> Path:
    """On-disk location for per-class aggregated priors.

    Lives beside the per-sample ``train_data_set`` tree so backups/exports
    catch both. Call-sites should not import PROJECT_ROOT directly; this
    helper keeps the path resolvable at import time without reading ctx
    before server.py has populated it.
    """
    return ctx.PROJECT_ROOT / "train_data_set" / "class_priors"

bp = Blueprint("api_liquify_3d", __name__, url_prefix="/api")

_LANDMARKS_FILENAME = "landmarks_3d.csv"
_REFINED_ANNOTATION_FILENAME = "annotation_refined_liquify3d.nii.gz"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _landmark_store_for(job_id: str) -> LandmarkStore:
    return LandmarkStore(ctx._job_file(job_id, _LANDMARKS_FILENAME))


def _resolve_annotation_path(job_id: str) -> Path | None:
    """Locate the moving-space registered annotation to refine.

    Only the **job-scoped** ants_registration output is accepted. Falling
    back to ``active_output_dir()`` would silently pull in an unrelated
    pipeline's huge annotation and OOM the solver, so we require callers to
    copy/link the annotation into their job directory first.
    """
    candidate = ctx._job_output_dir(job_id) / "ants_registration" / "annotation_registered.nii.gz"
    return candidate if candidate.exists() else None


def _pair_to_dict(pair) -> dict:
    return {
        "z": int(pair.z),
        "real_y": float(pair.real[0]),
        "real_x": float(pair.real[1]),
        "atlas_y": float(pair.atlas[0]),
        "atlas_x": float(pair.atlas[1]),
    }


# ---------------------------------------------------------------------------
# GET /api/liquify-3d/state  — list current landmarks + refinement status
# ---------------------------------------------------------------------------


@bp.get("/liquify-3d/state")
def liquify_3d_state():
    job_id = ctx._query_job_id()
    store = _landmark_store_for(job_id)
    pairs = store.list_pairs()
    refined_path = ctx._job_file(job_id, _REFINED_ANNOTATION_FILENAME)
    annotation_path = _resolve_annotation_path(job_id)
    source_available = annotation_path is not None

    # Expose the annotation grid shape so the frontend can show what its
    # canvas-pixel → annotation-voxel rescale factor will be (#12 — coord
    # conversion was previously silent).
    annotation_shape = None
    if source_available:
        try:
            import nibabel as _nib
            annotation_shape = list(_nib.load(str(annotation_path)).shape)
        except Exception:  # noqa: BLE001 — surface only when we can read header
            annotation_shape = None

    # Empty-state hint: if no source annotation, tell the frontend what the
    # user needs to do BEFORE liquify can run. Prevents a silent empty UI.
    if not source_available:
        guidance = (
            "No ants_registration/annotation_registered.nii.gz found for this job. "
            "Run the ANTs registration pipeline first, then return to this tab."
        )
    else:
        guidance = None
    return jsonify(
        {
            "ok": True,
            "jobId": job_id,
            "pair_count": len(pairs),
            "pairs": [_pair_to_dict(p) for p in pairs],
            "refined_annotation_exists": refined_path.exists(),
            "source_annotation_available": source_available,
            "annotation_shape": annotation_shape,
            "guidance": guidance,
        }
    )


@bp.get("/liquify-3d/class-prior/coverage")
def class_prior_coverage():
    """Return the prior's landmark coverage as a list of ``{z, count}`` rows
    (one row per z-grid entry). Powers the frontend's coverage heatmap so
    the user can see at a glance which AP positions still need corrections.
    """
    class_name = (request.args.get("class") or "").strip()
    if not class_name:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "missing 'class' query param",
                    "error_code": ERR_INVALID_INPUT,
                }
            ),
            400,
        )
    store = ClassPriorStore(class_name=class_name, priors_root=_class_priors_root())
    bins = [{"z": int(e.z), "count": int(e.n)} for e in store.entries()]
    bins.sort(key=lambda b: b["z"])
    return jsonify(
        {
            "ok": True,
            "class": class_name,
            "bins": bins,
            "sample_count": store.sample_count(),
        }
    )


@bp.get("/liquify-3d/class-registry/list")
def class_registry_list():
    """Return all class names known to the system: those in
    ``sample_class_registry.json`` plus those that have on-disk priors.
    Powers the frontend dropdown so the user does not have to type an
    existing class name and risk a typo.
    """
    classes = list_known_classes(
        registry_path=_class_registry_path(),
        priors_root=_class_priors_root(),
    )
    return jsonify({"ok": True, "classes": classes})


@bp.get("/liquify-3d/class-registry/detect")
def class_registry_detect():
    """Suggest a class for a given ``sampleId`` by consulting the
    auto-detect registry. Returns ``{class: null}`` when no pattern matches.
    """
    sample_id = (request.args.get("sampleId") or request.args.get("jobId") or "").strip()
    cls = detect_class_for_sample(sample_id, registry_path=_class_registry_path())
    return jsonify({"ok": True, "sampleId": sample_id, "class": cls})


_QC_DONE_FILENAME = "qc_done.json"


@bp.post("/liquify-3d/qc-done")
def liquify_3d_qc_done():
    """Mark the current job as user-approved and persist the metrics that
    were achieved at sign-off so future class-prior loads can show
    "previously approved at NCC=X" badges.

    Body: ``{jobId, metrics?, note?, className?}``

    When ``className`` is provided we also append a metrics-bearing record
    to that class's ``sample_log.jsonl`` so γ history captures the quality
    each contributing sample reached at sign-off.
    """
    import json as _json
    import time as _time

    payload = request.get_json(silent=True) or {}
    job_id = ctx._payload_job_id(payload)
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else {}
    note = str(payload.get("note") or "")
    class_name = str(payload.get("className") or "").strip()

    marker_path = ctx._job_file(job_id, _QC_DONE_FILENAME)
    record = {
        "jobId": job_id,
        "timestamp": _time.time(),
        "metrics": metrics,
        "note": note,
        "className": class_name or None,
    }
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    marker_path.write_text(_json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")

    # Optional: append metrics into the class prior's sample_log so γ
    # history captures sign-off quality across same-class runs.
    if class_name:
        store = ClassPriorStore(class_name=class_name, priors_root=_class_priors_root())
        log_path = store.class_dir / "sample_log.jsonl"
        if log_path.parent.exists() or True:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8") as fh:
                fh.write(
                    _json.dumps(
                        {
                            "sample_id": job_id,
                            "pair_count": len(_landmark_store_for(job_id).list_pairs()),
                            "metrics": metrics,
                            "qc_note": note,
                            "qc_done_ts": record["timestamp"],
                        },
                        ensure_ascii=False,
                    ) + "\n"
                )

    return jsonify({"ok": True, "jobId": job_id, "marker_path": str(marker_path)})


@bp.get("/liquify-3d/qc-status")
def liquify_3d_qc_status():
    """Return whether the job has been QC-signed-off and the metrics from
    that sign-off, so the frontend can show a "Done ✓" badge instead of
    asking the user to re-confirm.
    """
    import json as _json

    job_id = ctx._query_job_id()
    marker_path = ctx._job_file(job_id, _QC_DONE_FILENAME)
    if not marker_path.exists():
        return jsonify({"ok": True, "jobId": job_id, "done": False})
    try:
        record = _json.loads(marker_path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        record = {}
    return jsonify(
        {
            "ok": True,
            "jobId": job_id,
            "done": True,
            "timestamp": record.get("timestamp"),
            "metrics": record.get("metrics") or {},
            "note": record.get("note") or "",
            "className": record.get("className"),
        }
    )


@bp.get("/liquify-3d/progress")
def liquify_3d_progress():
    """Return the latest progress snapshot for the given job.

    The frontend polls this while an ``apply`` or ``finalize`` is in flight
    so the UI can surface stage + percent instead of an idle spinner.
    ``{}`` indicates no operation in progress (or one that hasn't reported
    anything yet).
    """
    job_id = ctx._query_job_id()
    data = read_liquify_progress(ctx._job_output_dir(job_id))
    return jsonify({"ok": True, "jobId": job_id, **data})


# ---------------------------------------------------------------------------
# POST /api/liquify-3d/add-pair
# Body: {jobId, z, real: [y, x], atlas: [y, x]}
# ---------------------------------------------------------------------------


@bp.post("/liquify-3d/add-pair")
def liquify_3d_add_pair():
    payload = request.get_json(silent=True) or {}
    job_id = ctx._payload_job_id(payload)

    try:
        z = int(payload["z"])
        real_y, real_x = float(payload["real"][0]), float(payload["real"][1])
        atlas_y, atlas_x = float(payload["atlas"][0]), float(payload["atlas"][1])
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": f"invalid payload: {exc}",
                    "error_code": ERR_INVALID_INPUT,
                    "required": "{jobId, z:int, real:[y,x], atlas:[y,x]}",
                }
            ),
            400,
        )

    # Optional: frontend posts click-pixel coords in the overlay PNG's native
    # size; we rescale to the annotation-grid voxel space so the Laplacian
    # solver's vol_shape is consistent with the stored landmarks.
    image_dims = payload.get("image_dims_yx")
    if image_dims and len(image_dims) == 2:
        try:
            img_h, img_w = float(image_dims[0]), float(image_dims[1])
        except (TypeError, ValueError):
            img_h = img_w = 0.0
        ann_path = _resolve_annotation_path(job_id)
        if ann_path is not None and img_h > 0 and img_w > 0:
            import nibabel as nib  # local import — only needed on this path
            ann_img = nib.load(str(ann_path))
            _d, ann_h, ann_w = ann_img.shape
            sy = ann_h / img_h
            sx = ann_w / img_w
            real_y, real_x = real_y * sy, real_x * sx
            atlas_y, atlas_x = atlas_y * sy, atlas_x * sx

    store = _landmark_store_for(job_id)
    pair = store.add_pair(z=z, real=(real_y, real_x), atlas=(atlas_y, atlas_x))
    return jsonify(
        {
            "ok": True,
            "jobId": job_id,
            "pair": _pair_to_dict(pair),
            "pair_count": len(store.list_pairs()),
        }
    )


# ---------------------------------------------------------------------------
# DELETE /api/liquify-3d/pair/<index>?job=<id>
# ---------------------------------------------------------------------------


@bp.delete("/liquify-3d/pair/<int:index>")
def liquify_3d_remove_pair(index: int):
    job_id = ctx._query_job_id()
    store = _landmark_store_for(job_id)
    try:
        removed = store.remove_pair(index)
    except IndexError as exc:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": str(exc),
                    "error_code": ERR_INVALID_INPUT,
                }
            ),
            400,
        )
    return jsonify(
        {
            "ok": True,
            "jobId": job_id,
            "removed": _pair_to_dict(removed),
            "pair_count": len(store.list_pairs()),
        }
    )


# ---------------------------------------------------------------------------
# POST /api/liquify-3d/clear
# ---------------------------------------------------------------------------


@bp.post("/liquify-3d/clear")
def liquify_3d_clear():
    payload = request.get_json(silent=True) or {}
    job_id = ctx._payload_job_id(payload)
    store = _landmark_store_for(job_id)
    store.clear()
    return jsonify({"ok": True, "jobId": job_id, "pair_count": 0})


# ---------------------------------------------------------------------------
# POST /api/liquify-3d/apply
# Body: {jobId, rtol?, maxiter?}
# Runs the Laplacian solve and writes annotation_refined_liquify3d.nii.gz
# ---------------------------------------------------------------------------


@bp.post("/liquify-3d/apply")
def liquify_3d_apply():
    payload = request.get_json(silent=True) or {}
    job_id = ctx._payload_job_id(payload)
    store = _landmark_store_for(job_id)
    pairs = store.list_pairs()
    if not pairs:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "no landmarks to apply",
                    "error_code": ERR_INVALID_INPUT,
                }
            ),
            400,
        )

    annotation_path = _resolve_annotation_path(job_id)
    if annotation_path is None:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "source annotation (ants_registration/annotation_registered.nii.gz) not found",
                    "error_code": ERR_NOT_FOUND,
                }
            ),
            404,
        )

    out_path = ctx._job_file(job_id, _REFINED_ANNOTATION_FILENAME)
    # Reset any stale progress from a prior run so the poller starts clean.
    clear_liquify_progress(ctx._job_output_dir(job_id))
    meta = refine_annotation_with_landmarks(
        annotation_path=annotation_path,
        landmarks_csv=store.path,
        output_path=out_path,
        rtol=float(payload.get("rtol", 1e-2)),
        maxiter=int(payload.get("maxiter", 500)),
        progress_cb=_progress_cb_for_job(job_id),
    )
    return jsonify(
        {
            "ok": True,
            "jobId": job_id,
            "source_annotation_path": str(annotation_path),
            **meta,
        }
    )


# ===========================================================================
# Phase γ — class-prior closed-loop endpoints
# ===========================================================================


@bp.get("/liquify-3d/class-prior/status")
def class_prior_status():
    """Return how many samples have contributed to a class's prior and
    whether it currently qualifies for warm-start application.

    Query: ``class=<name>``
    """
    class_name = (request.args.get("class") or "").strip()
    if not class_name:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "missing 'class' query param",
                    "error_code": ERR_INVALID_INPUT,
                }
            ),
            400,
        )
    store = ClassPriorStore(class_name=class_name, priors_root=_class_priors_root())
    return jsonify(
        {
            "ok": True,
            "class": class_name,
            "sample_count": store.sample_count(),
            "entry_count": len(store.entries()),
            "min_samples_for_apply": MIN_SAMPLES_FOR_APPLY,
            "ready_for_warm_start": store.sample_count() >= MIN_SAMPLES_FOR_APPLY,
        }
    )


@bp.post("/liquify-3d/class-prior/save")
def class_prior_save():
    """Merge the current job's landmark pairs into the named class prior.

    Body: ``{jobId, class, sampleId?, metrics?}``
    """
    payload = request.get_json(silent=True) or {}
    job_id = ctx._payload_job_id(payload)
    class_name = str(payload.get("class", "")).strip()
    if not class_name:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "missing 'class' field",
                    "error_code": ERR_INVALID_INPUT,
                }
            ),
            400,
        )
    sample_id = str(payload.get("sampleId") or job_id)
    metrics = payload.get("metrics") if isinstance(payload.get("metrics"), dict) else None

    pairs = _landmark_store_for(job_id).list_pairs()
    if not pairs:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "job has no landmark pairs to contribute",
                    "error_code": ERR_INVALID_INPUT,
                }
            ),
            400,
        )
    store = ClassPriorStore(class_name=class_name, priors_root=_class_priors_root())
    store.update(sample_id=sample_id, pairs=pairs, metrics=metrics)
    return jsonify(
        {
            "ok": True,
            "class": class_name,
            "sampleId": sample_id,
            "merged_pair_count": len(pairs),
            "sample_count": store.sample_count(),
            "entry_count": len(store.entries()),
            "ready_for_warm_start": store.sample_count() >= MIN_SAMPLES_FOR_APPLY,
        }
    )


@bp.post("/liquify-3d/class-prior/apply-warm-start")
def class_prior_apply_warm_start():
    """Populate the target job's landmark store from the class prior.

    Body: ``{jobId, class, force?}``
    """
    payload = request.get_json(silent=True) or {}
    job_id = ctx._payload_job_id(payload)
    class_name = str(payload.get("class", "")).strip()
    force = bool(payload.get("force", False))
    if not class_name:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "missing 'class' field",
                    "error_code": ERR_INVALID_INPUT,
                }
            ),
            400,
        )
    prior = load_prior(class_name, priors_root=_class_priors_root())
    if prior is None:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": (
                        f"class '{class_name}' has fewer than "
                        f"{MIN_SAMPLES_FOR_APPLY} contributing samples"
                    ),
                    "error_code": ERR_NOT_FOUND,
                }
            ),
            404,
        )

    target_csv = ctx._job_file(job_id, _LANDMARKS_FILENAME)
    try:
        written = prior.apply_as_warm_start(target_csv, force=force)
    except ValueError as exc:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": str(exc),
                    "error_code": ERR_INVALID_INPUT,
                    "hint": "re-POST with force=true to overwrite existing pairs",
                }
            ),
            409,
        )
    return jsonify(
        {
            "ok": True,
            "jobId": job_id,
            "class": class_name,
            "pair_count": written,
        }
    )


# ===========================================================================
# Close-the-loop: finalize refined annotation into downstream cell counts
# ===========================================================================


def _resolve_cells_csv(job_id: str) -> Path | None:
    """Pick the best deduped-cells CSV to re-map against the refined labels.

    Preference order (most to least ideal):
      1. ``cells_dedup.csv`` — the pre-mapping deduped output from the
         pipeline, has no stale region columns.
      2. ``cells_mapped.csv`` — already mapped; finalize drops the stale
         region columns before re-mapping.
    """
    job_dir = ctx._job_output_dir(job_id)
    for name in ("cells_dedup.csv", "cells_mapped.csv"):
        p = job_dir / name
        if p.exists():
            return p
    # Also look in the globally active run dir as a fallback for legacy
    # pipelines that didn't scope outputs by job id.
    for name in ("cells_dedup.csv", "cells_mapped.csv"):
        p = ctx.active_output_dir() / name
        if p.exists():
            return p
    return None


def _resolve_real_slice_paths(job_id: str) -> list[Path]:
    """Collect the moving-volume slice paths recorded by the original
    truth export so we can re-run it against the refined annotation.
    """
    job_dir = ctx._job_output_dir(job_id)
    # Option A: read slice_registration_qc.csv and lift the 'slice_path'
    # column — most faithful to the original run's slice ordering.
    qc_csv = job_dir / "slice_registration_qc.csv"
    if qc_csv.exists():
        try:
            import pandas as _pd

            df = _pd.read_csv(qc_csv)
            if "slice_path" in df.columns:
                paths = [Path(str(p)) for p in df["slice_path"].tolist()]
                paths = [p for p in paths if p.exists()]
                if paths:
                    return paths
        except Exception:
            pass
    # Option B: glob tmp_merged for merged_*.tif in ascending order.
    merged_dir = job_dir / "tmp_merged"
    if merged_dir.exists():
        return sorted(merged_dir.glob("merged_*.tif"))
    return []


@bp.post("/liquify-3d/finalize")
def liquify_3d_finalize():
    """Re-export truth slices from the refined annotation and re-run the
    cell→region mapping + aggregation so downstream bar charts reflect the
    user's corrections.

    Body: ``{jobId, pixelSizeUm?, slicingPlane?, atlasHemisphere?, structureCsv?}``
    """
    payload = request.get_json(silent=True) or {}
    job_id = ctx._payload_job_id(payload)

    job_dir = ctx._job_output_dir(job_id)
    refined_path = job_dir / "annotation_refined_liquify3d.nii.gz"
    if not refined_path.exists():
        return (
            jsonify(
                {
                    "ok": False,
                    "error": (
                        "annotation_refined_liquify3d.nii.gz not found — run /apply first"
                    ),
                    "error_code": ERR_NOT_FOUND,
                }
            ),
            404,
        )

    cells_csv = _resolve_cells_csv(job_id)
    if cells_csv is None:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": "no cells CSV (cells_dedup.csv or cells_mapped.csv) found for this job",
                    "error_code": ERR_NOT_FOUND,
                }
            ),
            404,
        )

    real_slice_paths = _resolve_real_slice_paths(job_id)
    if not real_slice_paths:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": (
                        "no real-slice paths found (slice_registration_qc.csv missing and "
                        "tmp_merged/ empty)"
                    ),
                    "error_code": ERR_NOT_FOUND,
                }
            ),
            404,
        )

    pixel_size_um = float(payload.get("pixelSizeUm", 5.0))
    slicing_plane = str(payload.get("slicingPlane", "coronal"))
    atlas_hemisphere = str(payload.get("atlasHemisphere", ""))
    structure_csv = payload.get("structureCsv")

    clear_liquify_progress(job_dir)
    try:
        meta = finalize_liquify_to_cell_counts(
            outputs_dir=job_dir,
            real_slice_paths=real_slice_paths,
            cells_csv=cells_csv,
            pixel_size_um=pixel_size_um,
            slicing_plane=slicing_plane,
            atlas_hemisphere=atlas_hemisphere,
            structure_csv=structure_csv,
            progress_cb=_progress_cb_for_job(job_id),
        )
    except FileNotFoundError as exc:
        return (
            jsonify(
                {
                    "ok": False,
                    "error": str(exc),
                    "error_code": ERR_NOT_FOUND,
                }
            ),
            404,
        )

    return jsonify(
        {
            "ok": True,
            "jobId": job_id,
            "source_cells_csv": str(cells_csv),
            **meta,
        }
    )
