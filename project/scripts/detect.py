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


class CellposeDetectionError(RuntimeError):
    """Raised when Cellpose was requested but could not produce a valid run."""


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

    Built-in aliases are normalized (e.g., "sam" → "cpsam").
    Unknown names (user-trained models or paths) pass through unchanged
    so they can be loaded via ``CellposeModel(pretrained_model=name)``.
    """
    s = str(name or "").strip().lower()
    if not s:
        return "cpsam"
    if "cpsam" in s or s == "sam" or "cellpose-sam" in s:
        return "cpsam"
    if "nuclei" in s:
        return "nuclei"
    if "cyto3" in s:
        return "cyto3"
    if "cyto2" in s:
        return "cyto2"
    if "cyto" in s:
        return "cyto"
    # Not a built-in — assume it's a user-trained model name or path.
    # For path-like names, return just the basename to preserve path casing.
    original = str(name or "").strip()
    return Path(original).name if ("/" in original or "\\" in original) else original


_BUILTIN_CELLPOSE_NAMES = {"cpsam", "sam", "cyto", "cyto2", "cyto3", "nuclei"}


def _is_cellpose_model(name: str) -> bool:
    """Return True if the model name refers to any Cellpose model.

    Returns True for built-in names, names starting with 'cellpose',
    and any name that is not a known non-Cellpose detector
    (like 'log', 'peak', 'threshold', 'reporter_positive', 'none').
    """
    s = str(name or "").strip().lower()
    if not s or s in ("none", "disabled", "off", "skip"):
        return False
    if s in _BUILTIN_CELLPOSE_NAMES:
        return True
    if s.startswith("cellpose"):
        return True
    # Non-Cellpose detector names
    if s in ("log", "peak", "threshold", "reporter_positive", "fallback"):
        return False
    # Unknown name — assume it's a user-trained Cellpose model
    return True


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
    """Load and cache a Cellpose model.

    Returns (model, is_legacy) where is_legacy=True for Cellpose v2/v3.
    """
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
        result = (model, False)
    elif hasattr(models, "Cellpose"):
        # Legacy v2/v3
        model = models.Cellpose(gpu=bool(use_gpu), model_type=str(model_type))
        _log.info("Loaded legacy Cellpose model: type=%s, gpu=%s", model_type, use_gpu)
        result = (model, True)
    else:
        raise CellposeRuntimeError(
            "Cannot find Cellpose model class. Please upgrade cellpose: pip install --upgrade cellpose"
        )
    _CELLPOSE_MODEL_CACHE[key] = result
    return result


def _masks_to_centroids(
    masks: np.ndarray,
    detector: str,
    intensity_image: np.ndarray | None = None,
) -> pd.DataFrame:
    _empty_cols = [
        "cell_id",
        "x",
        "y",
        "score",
        "detector",
        "area_px",
        "elongation",
        "mean_intensity",
    ]
    if masks is None or masks.size == 0 or int(np.max(masks)) <= 0:
        return pd.DataFrame(columns=_empty_cols)

    base_props = ["label", "centroid", "area"]
    extra_props: list[str] = []
    if intensity_image is not None:
        extra_props.append("mean_intensity")
    # minor/major axis needs ≥3px objects; safe to request always
    extra_props += ["minor_axis_length", "major_axis_length"]

    props = measure.regionprops_table(
        masks.astype(np.int32, copy=False),
        intensity_image=intensity_image,
        properties=base_props + extra_props,
    )
    if not props or len(props.get("label", [])) == 0:
        return pd.DataFrame(columns=_empty_cols)

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

    # elongation: minor/major axis ratio (1.0 = circle, 0.0 = line)
    minor = np.asarray(props.get("minor_axis_length", []), dtype=np.float32)
    major = np.asarray(props.get("major_axis_length", []), dtype=np.float32)
    if len(minor) == len(df) and len(major) == len(df):
        with np.errstate(invalid="ignore", divide="ignore"):
            elong = np.where(major > 0, minor / major, 1.0)
        df["elongation"] = np.clip(elong, 0.0, 1.0).astype(np.float32)
    else:
        df["elongation"] = np.float32(1.0)

    # mean_intensity from intensity image
    if intensity_image is not None and "mean_intensity" in props:
        mi = np.asarray(props["mean_intensity"], dtype=np.float32)
        df["mean_intensity"] = mi if len(mi) == len(df) else np.float32(0.0)
    else:
        df["mean_intensity"] = np.float32(0.0)

    return df[_empty_cols]


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
    _cols = ["cell_id", "x", "y", "score", "detector", "area_px", "elongation", "mean_intensity"]
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
                "elongation": 1.0,
                "mean_intensity": float(img[y, x]),
            }
        )
    return pd.DataFrame(rows, columns=_cols)


def detect_cells_log_fallback(
    slice_path: Path,
    min_sigma: float = 1.2,
    max_sigma: float = 5.0,
    num_sigma: int = 8,
    threshold_rel: float = 0.03,
) -> pd.DataFrame:
    _cols = ["cell_id", "x", "y", "score", "detector", "area_px", "elongation", "mean_intensity"]
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
        return pd.DataFrame(columns=_cols)

    h, w = img.shape[:2]
    rows = []
    for i, b in enumerate(blobs, 1):
        y, x0, sigma = float(b[0]), float(b[1]), float(b[2])
        r = np.sqrt(2.0) * sigma
        y0 = int(np.clip(round(y), 0, h - 1))
        x1 = int(np.clip(round(x0), 0, w - 1))
        # Disk mean intensity: sample pixels within radius r
        iy0 = int(max(0, y0 - r))
        iy1 = int(min(h, y0 + r + 1))
        ix0 = int(max(0, x1 - r))
        ix1 = int(min(w, x1 + r + 1))
        patch = img[iy0:iy1, ix0:ix1]
        mean_int = float(np.mean(patch)) if patch.size > 0 else float(img[y0, x1])
        rows.append(
            {
                "cell_id": i,
                "x": x0,
                "y": y,
                "score": float(img[y0, x1]),
                "detector": "fallback_log",
                "area_px": float(np.pi * r * r),
                "elongation": 1.0,  # circular blob approximation
                "mean_intensity": mean_int,
            }
        )
    return pd.DataFrame(rows, columns=_cols)


def _tile_starts(length: int, tile_size: int, overlap: int) -> list[int]:
    if tile_size <= 0 or length <= tile_size:
        return [0]
    step = max(1, int(tile_size) - int(overlap))
    starts = list(range(0, max(1, length - tile_size + 1), step))
    last = max(0, int(length) - int(tile_size))
    if not starts or starts[-1] != last:
        starts.append(last)
    return sorted(set(int(v) for v in starts))


def _eval_cellpose_masks(
    model: Any,
    imgf: np.ndarray,
    *,
    slice_label: str,
    model_type: str,
    diameter_px: float | None,
    channels: list[int],
    flow_threshold: float,
    cellprob_threshold: float,
    min_size: int,
    batch_size: int,
    tile_overlap: float,
    resample: bool,
    raise_on_error: bool,
):
    kwargs = dict(
        diameter=diameter_px,
        channels=channels,
        flow_threshold=float(flow_threshold),
        cellprob_threshold=float(cellprob_threshold),
        min_size=max(0, int(min_size)),
        batch_size=max(1, int(batch_size)),
        tile=True,
        tile_overlap=float(tile_overlap),
        resample=bool(resample),
        normalize=False,
    )

    try:
        return model.eval(imgf, **kwargs)
    except TypeError:
        kwargs2 = dict(diameter=diameter_px, channels=channels)
        try:
            return model.eval(imgf, **kwargs2)
        except Exception as exc:
            if raise_on_error:
                raise CellposeDetectionError(
                    f"Cellpose eval failed for model '{model_type}' on {slice_label}: {exc}"
                ) from exc
            return None
    except Exception as exc:
        if raise_on_error:
            raise CellposeDetectionError(
                f"Cellpose eval failed for model '{model_type}' on {slice_label}: {exc}"
            ) from exc
        return None


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
    batch_size: int = 1,
    tile_overlap: float = 0.05,
    resample: bool = False,
    external_tile_size_px: int | None = None,
    external_tile_overlap_px: int = 64,
    raise_on_error: bool = False,
    return_masks: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, np.ndarray]:
    try:
        model, _is_legacy = _load_cellpose_model(model_type=model_type, use_gpu=use_gpu)
    except Exception as exc:
        if raise_on_error:
            raise CellposeDetectionError(
                f"failed to load Cellpose model '{model_type}' (gpu={bool(use_gpu)}): {exc}"
            ) from exc
        empty = pd.DataFrame()
        return (empty, np.empty((0, 0), dtype=np.int32)) if return_masks else empty

    img = _read_gray(slice_path)
    imgf = _norm_for_cellpose(img)

    # Only pass channels for legacy Cellpose (v2/v3).
    # Cellpose-SAM v4+ ignores channels and warns if present; do NOT pass it.
    ch = channels if isinstance(channels, list) and len(channels) == 2 else [0, 0]

    tile_size = int(external_tile_size_px or 0)
    if tile_size <= 0 and not use_gpu and max(imgf.shape[:2]) > 384:
        tile_size = 384
    overlap_px = max(0, int(external_tile_overlap_px))

    # Build eval kwargs — compatible with Cellpose v2/v3/v4.
    eval_kwargs: dict[str, Any] = dict(
        diameter=diameter_px,
        flow_threshold=float(flow_threshold),
        cellprob_threshold=float(cellprob_threshold),
        min_size=max(0, int(min_size)),
        batch_size=max(1, int(batch_size)),
        tile=True,
        tile_overlap=float(tile_overlap),
        resample=bool(resample),
        normalize=False,
    )

    # Tile-based inference to prevent OOM with small cells / large upscale
    bsize = _safe_tile_size(imgf.shape, diameter_px, vram_gb=8.0)
    if bsize is not None:
        eval_kwargs["bsize"] = bsize

    if _is_legacy:
        eval_kwargs["channels"] = ch

    if tile_size > 0 and max(imgf.shape[:2]) > tile_size:
        tile_rows: list[pd.DataFrame] = []
        y_starts = _tile_starts(int(imgf.shape[0]), tile_size, overlap_px)
        x_starts = _tile_starts(int(imgf.shape[1]), tile_size, overlap_px)
        for y0 in y_starts:
            y1 = min(int(imgf.shape[0]), int(y0) + tile_size)
            for x0 in x_starts:
                x1 = min(int(imgf.shape[1]), int(x0) + tile_size)
                tile_img = imgf[y0:y1, x0:x1]
                result = _eval_cellpose_masks(
                    model,
                    tile_img,
                    slice_label=f"{slice_path.name}@y{y0}:{y1},x{x0}:{x1}",
                    model_type=model_type,
                    diameter_px=diameter_px,
                    channels=ch,
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold,
                    min_size=min_size,
                    batch_size=batch_size,
                    tile_overlap=tile_overlap,
                    resample=resample,
                    raise_on_error=raise_on_error,
                )
                if result is None:
                    continue
                masks, flows, styles, *_ = result  # cellpose v4 returns 3 values; v3 returned 4
                tile_intensity = img[y0:y1, x0:x1].astype(np.float32)
                tile_df = _masks_to_centroids(
                    masks,
                    detector=f"cellpose_{model_type}",
                    intensity_image=tile_intensity,
                )
                if tile_df.empty:
                    continue
                tile_df["x"] = tile_df["x"].astype(np.float32) + float(x0)
                tile_df["y"] = tile_df["y"].astype(np.float32) + float(y0)
                tile_rows.append(tile_df)
        _ecols = [
            "cell_id",
            "x",
            "y",
            "score",
            "detector",
            "area_px",
            "elongation",
            "mean_intensity",
        ]
        if not tile_rows:
            empty = pd.DataFrame(columns=_ecols)
            return (empty, np.zeros(imgf.shape[:2], dtype=np.int32)) if return_masks else empty
        out = pd.concat(tile_rows, ignore_index=True)
        out["cell_id"] = np.arange(1, len(out) + 1, dtype=np.int32)
        df = out[_ecols]
        if return_masks:
            # Tiled path: no composite mask available
            return df, np.zeros(imgf.shape[:2], dtype=np.int32)
        return df

    # Single-image (non-tiled) inference
    try:
        result = model.eval(imgf, **eval_kwargs)
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
            if raise_on_error:
                raise CellposeDetectionError(
                    f"Cellpose inference failed for '{slice_path.name}' (model={model_type}): {exc2}"
                ) from exc2
            empty = pd.DataFrame()
            return (empty, np.empty((0, 0), dtype=np.int32)) if return_masks else empty
    except Exception as exc:
        if raise_on_error:
            raise CellposeDetectionError(
                f"Cellpose inference failed for '{slice_path.name}' (model={model_type}): {exc}"
            ) from exc
        empty = pd.DataFrame()
        return (empty, np.empty((0, 0), dtype=np.int32)) if return_masks else empty

    df = _masks_to_centroids(
        masks,
        detector=f"cellpose_{model_type}",
        intensity_image=img.astype(np.float32),
    )
    if return_masks:
        return df, masks
    return df


def _run_cellpose_by_name(
    slice_path: Path,
    model_name: str,
    cfg: dict[str, Any],
    *,
    raise_on_error: bool = False,
) -> pd.DataFrame:
    det_cfg = cfg.get("detection", {})
    model_type = _resolve_model_type(model_name)
    d_px = _diameter_px(det_cfg, cfg)
    use_gpu = _use_gpu(cfg, det_cfg)
    channels = det_cfg.get("cellpose_channels", [0, 0])
    flow_thr = float(det_cfg.get("cellpose_flow_threshold", 0.4))
    prob_thr = float(det_cfg.get("cellpose_cellprob_threshold", 0.0))
    min_sz = int(det_cfg.get("cellpose_min_size_px", 8))
    batch_size = int(det_cfg.get("cellpose_batch_size", 1))
    tile_overlap = float(det_cfg.get("cellpose_tile_overlap", 0.05))
    resample = bool(det_cfg.get("cellpose_resample", False))
    tile_size_px = det_cfg.get("cellpose_external_tile_size_px", None)
    external_tile_size_px = int(tile_size_px) if tile_size_px not in (None, "", 0) else None
    external_tile_overlap_px = int(det_cfg.get("cellpose_external_tile_overlap_px", 64))

    return detect_cells_cellpose(
        slice_path=slice_path,
        model_type=model_type,
        diameter_px=d_px,
        use_gpu=use_gpu,
        channels=channels if isinstance(channels, list) else [0, 0],
        flow_threshold=flow_thr,
        cellprob_threshold=prob_thr,
        min_size=min_sz,
        batch_size=batch_size,
        tile_overlap=tile_overlap,
        resample=resample,
        external_tile_size_px=external_tile_size_px,
        external_tile_overlap_px=external_tile_overlap_px,
        raise_on_error=raise_on_error,
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
    requested_cellpose = _is_cellpose_model(primary) or _is_cellpose_model(secondary)
    allow_fallback_raw = det_cfg.get("allow_fallback", None)
    allow_fallback = bool(allow_fallback_raw) if allow_fallback_raw is not None else auto_switch
    cellpose_errors: list[Exception] = []

    primary_df = pd.DataFrame()
    if _is_cellpose_model(primary):
        try:
            primary_df = _run_cellpose_by_name(
                slice_path,
                primary,
                cfg,
                raise_on_error=not allow_fallback,
            )
        except (CellposeDetectionError, CellposeRuntimeError) as exc:
            cellpose_errors.append(exc)
            if not allow_fallback:
                raise
            _log.warning(
                "Cellpose primary model '%s' failed for %s; falling back to non-Cellpose detector",
                primary,
                slice_path.name,
            )
            primary_df = pd.DataFrame()
        if not primary_df.empty and not merge_secondary:
            out = _dedup_xy(primary_df, radius_px=within_slice_dedup_px)
            out["cell_id"] = np.arange(1, len(out) + 1, dtype=np.int32)
            return out

    secondary_df = pd.DataFrame()
    if _is_cellpose_model(secondary):
        try:
            secondary_df = _run_cellpose_by_name(
                slice_path,
                secondary,
                cfg,
                raise_on_error=not allow_fallback,
            )
        except (CellposeDetectionError, CellposeRuntimeError) as exc:
            cellpose_errors.append(exc)
            if not allow_fallback:
                raise
            _log.warning(
                "Cellpose secondary model '%s' failed for %s; falling back to non-Cellpose detector",
                secondary,
                slice_path.name,
            )
            secondary_df = pd.DataFrame()
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

    if requested_cellpose and not allow_fallback:
        if cellpose_errors:
            raise cellpose_errors[0]
        return pd.DataFrame(columns=["cell_id", "x", "y", "score", "detector", "area_px"])

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
