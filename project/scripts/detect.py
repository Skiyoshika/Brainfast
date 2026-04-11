from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from skimage import measure
from skimage.feature import blob_log, peak_local_max
from tifffile import imread

try:
    from scripts.exceptions import CellposeRuntimeError
except ImportError:
    try:
        from exceptions import CellposeRuntimeError
    except ImportError:  # pragma: no cover

        class CellposeRuntimeError(RuntimeError):  # type: ignore[no-redef]
            pass


try:
    from scipy.spatial import cKDTree
except ImportError:  # pragma: no cover - optional dependency fallback
    cKDTree = None

_log = logging.getLogger(__name__)


_CELLPOSE_MODEL_CACHE: dict[tuple[str, bool], Any] = {}


def _read_gray(slice_path: Path) -> np.ndarray:
    img = imread(str(slice_path))
    if img.ndim == 3:
        img = img[..., 0]
    return img.astype(np.float32, copy=False)


def _norm_for_cellpose(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32, copy=False)
    p1, p99 = np.percentile(x, [1.0, 99.0])
    if float(p99) <= float(p1):
        p1, p99 = float(np.min(x)), float(np.max(x) + 1e-6)
    x = np.clip((x - p1) / (p99 - p1 + 1e-6), 0.0, 1.0)
    return x.astype(np.float32, copy=False)


def _resolve_model_type(name: str) -> str:
    """Resolve model name to Cellpose model type.

    Cellpose v4+ (Cellpose-SAM) uses 'cpsam' as the unified model;
    legacy names like cyto2/cyto3/nuclei are accepted but ignored by v4+.
    """
    s = str(name or "").strip().lower()
    if "cpsam" in s or "sam" in s:
        return "cpsam"
    if "nuclei" in s:
        return "nuclei"
    if "cyto3" in s:
        return "cyto3"
    if "cyto2" in s:
        return "cyto2"
    if "cyto" in s:
        return "cyto"
    # Default to cpsam (Cellpose-SAM) for v4+
    return "cpsam"


def _is_cellpose_model(name: str) -> bool:
    """Return True if the model name refers to any Cellpose/Cellpose-SAM model."""
    s = str(name or "").strip().lower()
    return s.startswith("cellpose") or s in ("cpsam", "sam", "cyto", "cyto2", "cyto3", "nuclei")


def _pixel_size_um_from_cfg(cfg: dict[str, Any]) -> float | None:
    raw = cfg.get("input", {}).get("pixel_size_um_xy", None)
    if raw in (None, "", "TODO"):
        return None
    try:
        v = float(raw)
        return v if v > 0 else None
    except Exception:
        return None


def _diameter_px(det_cfg: dict[str, Any], cfg: dict[str, Any]) -> float | None:
    raw_px = det_cfg.get("cellpose_diameter_px", None)
    if raw_px not in (None, "", 0):
        try:
            v = float(raw_px)
            if v > 0:
                return v
        except Exception:
            pass

    raw_um = det_cfg.get("cellpose_diameter_um", None)
    if raw_um in (None, "", 0):
        return None
    try:
        d_um = float(raw_um)
    except Exception:
        return None
    px_um = _pixel_size_um_from_cfg(cfg)
    if px_um is None or px_um <= 0:
        return None
    d_px = d_um / px_um
    return float(d_px) if d_px > 0 else None


def _use_gpu(cfg: dict[str, Any], det_cfg: dict[str, Any]) -> bool:
    """Auto-detect GPU: use CUDA if available, unless explicitly disabled."""
    forced = det_cfg.get("cellpose_gpu", None)
    if forced is not None:
        return bool(forced)
    dev = str(cfg.get("compute", {}).get("device", "auto")).lower()
    if dev == "cpu":
        return False
    # "auto", "cuda", "gpu" → try to use GPU if torch+CUDA is available
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


def _load_cellpose_model(model_type: str, use_gpu: bool):
    key = (str(model_type), bool(use_gpu))
    if key in _CELLPOSE_MODEL_CACHE:
        return _CELLPOSE_MODEL_CACHE[key]
    from cellpose import models

    # Cellpose v4+ (Cellpose-SAM): use CellposeModel with pretrained_model.
    # Falls back to legacy models.Cellpose for older versions.
    if hasattr(models, "CellposeModel"):
        # v4+: model_type is ignored, pretrained_model selects the model
        model = models.CellposeModel(gpu=bool(use_gpu), pretrained_model=str(model_type))
        _log.info("Loaded Cellpose-SAM model (v4+): pretrained=%s, gpu=%s", model_type, use_gpu)
    elif hasattr(models, "Cellpose"):
        # Legacy v2/v3
        model = models.Cellpose(gpu=bool(use_gpu), model_type=str(model_type))
        _log.info("Loaded legacy Cellpose model: type=%s, gpu=%s", model_type, use_gpu)
    else:
        raise CellposeRuntimeError(
            "Cannot find Cellpose model class. Please upgrade cellpose: pip install --upgrade cellpose"
        )
    _CELLPOSE_MODEL_CACHE[key] = model
    return model


def _masks_to_centroids(masks: np.ndarray, detector: str) -> pd.DataFrame:
    if masks is None or masks.size == 0 or int(np.max(masks)) <= 0:
        return pd.DataFrame(columns=["cell_id", "x", "y", "score", "detector", "area_px"])

    props = measure.regionprops_table(
        masks.astype(np.int32, copy=False),
        properties=("label", "centroid", "area"),
    )
    if not props or len(props.get("label", [])) == 0:
        return pd.DataFrame(columns=["cell_id", "x", "y", "score", "detector", "area_px"])

    df = pd.DataFrame(
        {
            "x": np.asarray(props["centroid-1"], dtype=np.float32),
            "y": np.asarray(props["centroid-0"], dtype=np.float32),
            "area_px": np.asarray(props["area"], dtype=np.float32),
        }
    )
    df["score"] = np.clip(np.sqrt(df["area_px"].astype(np.float32)), 0.0, None)
    df["cell_id"] = np.arange(1, len(df) + 1, dtype=np.int32)
    df["detector"] = str(detector)
    return df[["cell_id", "x", "y", "score", "detector", "area_px"]]


def _dedup_xy(df: pd.DataFrame, radius_px: float = 4.0) -> pd.DataFrame:
    if len(df) <= 1 or radius_px <= 0:
        return df
    if cKDTree is None:
        return df

    arr = df[["x", "y"]].to_numpy(dtype=np.float32, copy=False)
    scores = df["score"].to_numpy(dtype=np.float32, copy=False)
    order = np.argsort(-scores)
    tree = cKDTree(arr)
    keep = np.ones(len(df), dtype=bool)
    for idx in order:
        if not keep[idx]:
            continue
        neigh = tree.query_ball_point(arr[idx], r=float(radius_px))
        for j in neigh:
            if j == idx:
                continue
            keep[j] = False
    out = df.loc[keep].copy().reset_index(drop=True)
    out["cell_id"] = np.arange(1, len(out) + 1, dtype=np.int32)
    return out


def detect_cells_reporter_positive(
    slice_path: Path,
    intensity_threshold_pct: float = 95.0,
    min_area_px: float = 8.0,
    max_area_px: float = 2000.0,
) -> pd.DataFrame:
    """Detect fluorescent reporter-positive cells by intensity thresholding + connected components.

    Designed for dTom / GFP-style sparse labelling where positive cells are
    distinctly brighter than background.  Replaces Cellpose for study-specific
    reporter quantification (e.g. AAV-toolbox enhancer validation).
    """
    img = _read_gray(slice_path)
    x = _norm_for_cellpose(img)  # robust [0,1] normalisation
    thr = float(np.percentile(x, float(intensity_threshold_pct)))
    mask = (x >= thr).astype(np.uint8)
    labeled = measure.label(mask)
    props = measure.regionprops_table(
        labeled,
        intensity_image=img,
        properties=("label", "centroid", "area", "mean_intensity"),
    )
    if not props or len(props.get("label", [])) == 0:
        return pd.DataFrame(columns=["cell_id", "x", "y", "score", "detector", "area_px"])

    df = pd.DataFrame(
        {
            "x": np.asarray(props["centroid-1"], dtype=np.float32),
            "y": np.asarray(props["centroid-0"], dtype=np.float32),
            "area_px": np.asarray(props["area"], dtype=np.float32),
            "score": np.asarray(props["mean_intensity"], dtype=np.float32),
        }
    )
    df = df[(df["area_px"] >= float(min_area_px)) & (df["area_px"] <= float(max_area_px))]
    df = df.reset_index(drop=True)
    df["cell_id"] = np.arange(1, len(df) + 1, dtype=np.int32)
    df["detector"] = "reporter_positive"
    return df[["cell_id", "x", "y", "score", "detector", "area_px"]]


def detect_cells_fallback(
    slice_path: Path,
    min_distance: int = 8,
    threshold_abs: float = 200.0,
) -> pd.DataFrame:
    img = _read_gray(slice_path)
    coords = peak_local_max(
        img,
        min_distance=max(1, int(min_distance)),
        threshold_abs=float(threshold_abs),
    )
    rows = []
    for i, (y, x) in enumerate(coords, 1):
        rows.append(
            {
                "cell_id": i,
                "x": float(x),
                "y": float(y),
                "score": float(img[y, x]),
                "detector": "fallback_peak",
                "area_px": 1.0,
            }
        )
    return pd.DataFrame(rows, columns=["cell_id", "x", "y", "score", "detector", "area_px"])


def detect_cells_log_fallback(
    slice_path: Path,
    min_sigma: float = 1.2,
    max_sigma: float = 5.0,
    num_sigma: int = 8,
    threshold_rel: float = 0.03,
) -> pd.DataFrame:
    img = _read_gray(slice_path)
    x = _norm_for_cellpose(img)
    blobs = blob_log(
        x,
        min_sigma=float(min_sigma),
        max_sigma=float(max_sigma),
        num_sigma=max(1, int(num_sigma)),
        threshold=float(threshold_rel),
    )
    if blobs is None or len(blobs) == 0:
        return pd.DataFrame(columns=["cell_id", "x", "y", "score", "detector", "area_px"])

    rows = []
    for i, b in enumerate(blobs, 1):
        y, x0, sigma = float(b[0]), float(b[1]), float(b[2])
        r = np.sqrt(2.0) * sigma
        y0 = int(np.clip(round(y), 0, img.shape[0] - 1))
        x1 = int(np.clip(round(x0), 0, img.shape[1] - 1))
        rows.append(
            {
                "cell_id": i,
                "x": x0,
                "y": y,
                "score": float(img[y0, x1]),
                "detector": "fallback_log",
                "area_px": float(np.pi * r * r),
            }
        )
    return pd.DataFrame(rows, columns=["cell_id", "x", "y", "score", "detector", "area_px"])


def _safe_tile_size(
    img_shape: tuple[int, ...], diameter_px: float | None, vram_gb: float = 8.0
) -> int | None:
    """Compute a safe tile size (bsize) to avoid OOM with Cellpose-SAM.

    Cellpose-SAM upscales to its internal resolution (~30px cell diameter).
    If user cells are small (e.g. 2.4px), this creates a 12.5x upscale that
    exceeds GPU memory.  We use tiling (bsize) to keep peak memory within
    VRAM budget.

    Returns bsize (tile edge length in pixels) or None if tiling not needed.
    """
    if diameter_px is None or diameter_px <= 0:
        return None

    # Cellpose-SAM internal target is ~30px; upscale factor = 30 / diameter
    cellpose_target_diam = 30.0
    upscale = cellpose_target_diam / max(diameter_px, 1.0)

    if upscale <= 2.5:
        # Modest upscale — no tiling needed
        return None

    # Estimate memory: upscaled tile needs ~4 bytes/px * 6 buffers (model internals)
    # Max tile pixels = vram_gb * 1e9 / (4 * 6) / upscale^2
    # With safety margin of 2x
    max_upscaled_pixels = vram_gb * 1e9 / (4.0 * 6.0 * 2.0)
    max_input_pixels = max_upscaled_pixels / (upscale * upscale)
    bsize = int(np.sqrt(max_input_pixels))
    # Clamp to reasonable range
    bsize = max(128, min(bsize, max(img_shape[:2])))
    _log.info(
        "Cellpose tile mode: diameter=%.1fpx, upscale=%.1fx, bsize=%d",
        diameter_px,
        upscale,
        bsize,
    )
    return bsize


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
) -> pd.DataFrame:
    try:
        model = _load_cellpose_model(model_type=model_type, use_gpu=use_gpu)
    except Exception as exc:
        raise CellposeRuntimeError(f"Failed to load Cellpose model '{model_type}': {exc}") from exc

    img = _read_gray(slice_path)
    imgf = _norm_for_cellpose(img)

    # Build eval kwargs — compatible with Cellpose v2/v3/v4.
    # v4 (Cellpose-SAM): `channels` is deprecated, eval returns 3 values.
    # v2/v3: eval returns 4 values (masks, flows, styles, diams).
    kwargs: dict[str, Any] = dict(
        diameter=diameter_px,
        flow_threshold=float(flow_threshold),
        cellprob_threshold=float(cellprob_threshold),
        min_size=max(0, int(min_size)),
    )

    # Tile-based inference to prevent OOM with small cells / large upscale
    bsize = _safe_tile_size(imgf.shape, diameter_px, vram_gb=8.0)
    if bsize is not None:
        kwargs["bsize"] = bsize

    # Only pass channels for legacy versions (v2/v3)
    from cellpose import models as _cp_models

    if not hasattr(_cp_models, "CellposeModel") or hasattr(_cp_models, "Cellpose"):
        ch = channels if isinstance(channels, list) and len(channels) == 2 else [0, 0]
        kwargs["channels"] = ch

    try:
        result = model.eval(imgf, **kwargs)
        masks = result[0]  # works for both 3-tuple (v4) and 4-tuple (v2/v3)
    except (TypeError, RuntimeError) as exc:
        # OOM or API incompatibility — retry with conservative settings
        _log.warning("Cellpose eval failed (%s), retrying with tile mode...", exc)
        kwargs2: dict[str, Any] = dict(diameter=diameter_px)
        # Force tiling for retry
        retry_bsize = _safe_tile_size(imgf.shape, diameter_px, vram_gb=4.0)
        if retry_bsize is not None:
            kwargs2["bsize"] = retry_bsize
        try:
            result = model.eval(imgf, **kwargs2)
            masks = result[0]
        except Exception as exc2:
            raise CellposeRuntimeError(
                f"Cellpose inference failed for '{slice_path.name}' (model={model_type}): {exc2}"
            ) from exc2
    except Exception as exc:
        raise CellposeRuntimeError(
            f"Cellpose inference failed for '{slice_path.name}' (model={model_type}): {exc}"
        ) from exc

    return _masks_to_centroids(masks, detector=f"cellpose_{model_type}")


def _run_cellpose_by_name(slice_path: Path, model_name: str, cfg: dict[str, Any]) -> pd.DataFrame:
    det_cfg = cfg.get("detection", {})
    model_type = _resolve_model_type(model_name)
    d_px = _diameter_px(det_cfg, cfg)
    use_gpu = _use_gpu(cfg, det_cfg)
    channels = det_cfg.get("cellpose_channels", [0, 0])
    flow_thr = float(det_cfg.get("cellpose_flow_threshold", 0.4))
    prob_thr = float(det_cfg.get("cellpose_cellprob_threshold", 0.0))
    min_sz = int(det_cfg.get("cellpose_min_size_px", 8))

    return detect_cells_cellpose(
        slice_path=slice_path,
        model_type=model_type,
        diameter_px=d_px,
        use_gpu=use_gpu,
        channels=channels if isinstance(channels, list) else [0, 0],
        flow_threshold=flow_thr,
        cellprob_threshold=prob_thr,
        min_size=min_sz,
    )


def detect_cells(slice_path: Path, cfg: dict[str, Any]) -> pd.DataFrame:
    det_cfg = cfg.get("detection", {})

    # ── reporter-positive mode (study-specific fluorescent cell counting) ──
    if str(det_cfg.get("mode", "")).lower() == "reporter_positive":
        df = detect_cells_reporter_positive(
            slice_path,
            intensity_threshold_pct=float(det_cfg.get("reporter_intensity_threshold_pct", 95.0)),
            min_area_px=float(det_cfg.get("reporter_min_area_px", 8.0)),
            max_area_px=float(det_cfg.get("reporter_max_area_px", 2000.0)),
        )
        within_slice_dedup_px = float(det_cfg.get("within_slice_dedup_px", 4.0))
        df = _dedup_xy(df, radius_px=within_slice_dedup_px)
        df["cell_id"] = np.arange(1, len(df) + 1, dtype=np.int32)
        return df

    primary = str(det_cfg.get("primary_model", "cellpose_cyto2"))
    secondary = str(det_cfg.get("secondary_model", "cellpose_nuclei"))
    merge_secondary = bool(det_cfg.get("merge_primary_secondary", False))
    within_slice_dedup_px = float(det_cfg.get("within_slice_dedup_px", 4.0))
    auto_switch = bool(det_cfg.get("auto_switch_on_distortion", True))

    # Track whether a Cellpose model was requested so we can distinguish
    # "Cellpose ran and found 0 cells" from "Cellpose crashed".
    cellpose_requested = _is_cellpose_model(primary) or _is_cellpose_model(secondary)

    primary_df = pd.DataFrame()
    if _is_cellpose_model(primary):
        try:
            primary_df = _run_cellpose_by_name(slice_path, primary, cfg)
        except CellposeRuntimeError:
            if not auto_switch:
                raise
            _log.warning(
                "Cellpose primary model '%s' failed for %s; falling back to non-Cellpose detector",
                primary,
                slice_path.name,
            )
            cellpose_requested = False  # allow fallback
        if not primary_df.empty and not merge_secondary:
            out = _dedup_xy(primary_df, radius_px=within_slice_dedup_px)
            out["cell_id"] = np.arange(1, len(out) + 1, dtype=np.int32)
            return out

    if _is_cellpose_model(secondary):
        try:
            secondary_df = _run_cellpose_by_name(slice_path, secondary, cfg)
        except CellposeRuntimeError:
            if not auto_switch:
                raise
            _log.warning(
                "Cellpose secondary model '%s' failed for %s; falling back to non-Cellpose detector",
                secondary,
                slice_path.name,
            )
            secondary_df = pd.DataFrame()
            cellpose_requested = False  # allow fallback
        else:
            if not secondary_df.empty:
                if primary_df.empty:
                    out = _dedup_xy(secondary_df, radius_px=within_slice_dedup_px)
                    out["cell_id"] = np.arange(1, len(out) + 1, dtype=np.int32)
                    return out
                if merge_secondary:
                    combined = pd.concat([primary_df, secondary_df], ignore_index=True)
                    out = _dedup_xy(combined, radius_px=within_slice_dedup_px)
                    out["cell_id"] = np.arange(1, len(out) + 1, dtype=np.int32)
                    return out

    # If Cellpose was configured and we still haven't returned, both models
    # either produced empty results or failed.  When auto_switch is False and
    # Cellpose was the explicit choice, refuse to silently fall through to a
    # weaker detector — surface it as an error instead.
    if cellpose_requested and not auto_switch:
        raise CellposeRuntimeError(
            f"Cellpose models ({primary}, {secondary}) returned no cells for "
            f"'{slice_path.name}' and auto_switch_on_distortion is disabled"
        )

    fallback_model = str(det_cfg.get("fallback_model", "log")).lower()
    if "log" in fallback_model:
        log_df = detect_cells_log_fallback(
            slice_path,
            min_sigma=float(det_cfg.get("fallback_log_min_sigma", 1.2)),
            max_sigma=float(det_cfg.get("fallback_log_max_sigma", 5.0)),
            num_sigma=int(det_cfg.get("fallback_log_num_sigma", 8)),
            threshold_rel=float(det_cfg.get("fallback_log_threshold_rel", 0.03)),
        )
        if not log_df.empty:
            log_df["cell_id"] = np.arange(1, len(log_df) + 1, dtype=np.int32)
            return log_df

    thr = float(det_cfg.get("fallback_threshold", 200.0))
    md = int(det_cfg.get("fallback_min_distance", 8))
    out = detect_cells_fallback(slice_path, min_distance=md, threshold_abs=thr)
    out["cell_id"] = np.arange(1, len(out) + 1, dtype=np.int32)
    return out
