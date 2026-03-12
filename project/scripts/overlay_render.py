from __future__ import annotations

from pathlib import Path
import numpy as np
from tifffile import imread, imwrite
from skimage.transform import rescale
from skimage import measure
from skimage.transform import rotate
from scripts.allen_colors import load_allen_color_map


def alpha_blend(base_gray: np.ndarray, color_mask: np.ndarray, alpha: float) -> np.ndarray:
    base_rgb = np.stack([base_gray, base_gray, base_gray], axis=-1).astype(np.float32)
    out = (1.0 - alpha) * base_rgb + alpha * color_mask.astype(np.float32)
    return np.clip(out, 0, 255).astype(np.uint8)


def _norm_u8_robust(img: np.ndarray) -> np.ndarray:
    x = img.astype(np.float32)
    p1, p99 = np.percentile(x, 1), np.percentile(x, 99)
    if p99 <= p1:
        p1, p99 = float(np.min(x)), float(np.max(x) + 1e-6)
    x = np.clip((x - p1) / (p99 - p1 + 1e-6), 0, 1)
    return (x * 255.0).astype(np.uint8)


def _align_shape_physical(label_slice: np.ndarray, target_shape: tuple[int, int], atlas_res_um: float = 25.0, real_res_um: float = 0.65, fit_mode: str = "contain"):
    """physical scale -> fit mode -> center crop/pad; returns (canvas, fitted_shape)."""
    scale = atlas_res_um / real_res_um
    scaled_label = rescale(label_slice.astype(np.float32), scale, order=0, preserve_range=True, anti_aliasing=False).astype(np.uint16)

    th, tw = target_shape
    sh, sw = scaled_label.shape

    fh = th / max(sh, 1)
    fw = tw / max(sw, 1)
    mode = str(fit_mode or "contain").lower()
    if mode == "contain":
        fit = min(fh, fw)
    elif mode == "cover":
        fit = max(fh, fw)
    elif mode == "width-lock":
        fit = fw
    elif mode == "height-lock":
        fit = fh
    else:
        fit = min(fh, fw)

    if abs(float(fit) - 1.0) > 1e-6:
        scaled_label = rescale(scaled_label.astype(np.float32), float(fit), order=0, preserve_range=True, anti_aliasing=False).astype(np.uint16)

    fitted_shape = (int(scaled_label.shape[0]), int(scaled_label.shape[1]))
    sh, sw = scaled_label.shape
    out = np.zeros((th, tw), dtype=scaled_label.dtype)

    min_h = min(th, sh)
    min_w = min(tw, sw)
    start_y_out = (th - min_h) // 2
    start_x_out = (tw - min_w) // 2
    start_y_in = (sh - min_h) // 2
    start_x_in = (sw - min_w) // 2

    out[start_y_out : start_y_out + min_h, start_x_out : start_x_out + min_w] = \
        scaled_label[start_y_in : start_y_in + min_h, start_x_in : start_x_in + min_w]

    return out, fitted_shape


def _roi_bbox_from_real(real_img: np.ndarray, pad: int = 4) -> tuple[int, int, int, int]:
    x = real_img.astype(np.float32)
    if x.ndim == 3:
        x = x[..., 0]
    thr = float(np.percentile(x, 88))
    mask = x > thr
    lbl = measure.label(mask, connectivity=2)
    props = measure.regionprops(lbl)
    if not props:
        h, w = x.shape
        return (0, 0, int(w), int(h))
    props = sorted(props, key=lambda r: r.area, reverse=True)
    y0, x0, y1, x1 = props[0].bbox
    h, w = x.shape
    x0 = max(0, int(x0) - pad)
    y0 = max(0, int(y0) - pad)
    x1 = min(w, int(x1) + pad)
    y1 = min(h, int(y1) + pad)
    return (x0, y0, int(x1 - x0), int(y1 - y0))


def draw_contours(label_img: np.ndarray, color=(255, 255, 255)) -> np.ndarray:
    h, w = label_img.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for region_id in np.unique(label_img):
        if region_id == 0:
            continue
        contours = measure.find_contours((label_img == region_id).astype(np.uint8), 0.5)
        for c in contours:
            rr = np.clip(c[:, 0].astype(int), 0, h - 1)
            cc = np.clip(c[:, 1].astype(int), 0, w - 1)
            canvas[rr, cc] = color
    return canvas


def draw_contours_major(label_img: np.ndarray, top_k: int = 12) -> np.ndarray:
    """Draw outer brain boundary + major region boundaries for cleaner diagnostic view."""
    h, w = label_img.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Outer boundary from non-zero mask (cyan)
    mask = (label_img > 0).astype(np.uint8)
    for c in measure.find_contours(mask, 0.5):
        rr = np.clip(c[:, 0].astype(int), 0, h - 1)
        cc = np.clip(c[:, 1].astype(int), 0, w - 1)
        canvas[rr, cc] = np.array([0, 255, 255], dtype=np.uint8)

    # Major region boundaries by area (white)
    ids, counts = np.unique(label_img[label_img > 0], return_counts=True)
    if len(ids) > 0:
        order = np.argsort(counts)[::-1][: max(1, int(top_k))]
        major_ids = ids[order]
        for rid in major_ids:
            for c in measure.find_contours((label_img == rid).astype(np.uint8), 0.5):
                rr = np.clip(c[:, 0].astype(int), 0, h - 1)
                cc = np.clip(c[:, 1].astype(int), 0, w - 1)
                canvas[rr, cc] = np.array([255, 255, 255], dtype=np.uint8)

    return canvas


def render_overlay(real_slice_path: Path, label_slice_path: Path, out_png: Path, alpha: float = 0.45, mode: str = "fill", structure_csv: Path | None = None, min_mean_threshold: float = 8.0, pixel_size_um: float = 0.65, rotate_deg: float = 0.0, flip_mode: str = "none", return_meta: bool = False, major_top_k: int = 12, fit_mode: str = "contain"): 
    real = imread(str(real_slice_path))
    label = imread(str(label_slice_path))

    if real.ndim == 3:
        real = real[..., 0]
    if label.ndim == 3:
        label = label[..., 0]

    label_shape_before = tuple(int(x) for x in label.shape)
    real_shape = tuple(int(x) for x in real.shape)
    scale = float(25.0 / float(pixel_size_um))

    # Pre-transform: rotate/flip before align
    if rotate_deg != 0.0:
        label = rotate(label.astype(np.float32), rotate_deg, order=0, preserve_range=True, resize=True).astype(np.uint16)
    if flip_mode == "horizontal":
        label = np.fliplr(label)
    elif flip_mode == "vertical":
        label = np.flipud(label)

    # Geometrically align label to real slice (physical scale + fit mode + center crop/pad)
    label_shape_fitted = tuple(int(x) for x in label.shape)
    if label.shape != real.shape:
        label, label_shape_fitted = _align_shape_physical(label, real.shape, atlas_res_um=25.0, real_res_um=pixel_size_um, fit_mode=fit_mode)

    label_shape_after = tuple(int(x) for x in label.shape)
    roi_bbox = _roi_bbox_from_real(real)

    # ROI coordinate roundtrip sanity (full -> local -> full)
    rx, ry, rw, rh = roi_bbox
    cx_full = float(rx + rw / 2.0)
    cy_full = float(ry + rh / 2.0)
    cx_local = float(cx_full - rx)
    cy_local = float(cy_full - ry)
    cx_full_back = float(cx_local + rx)
    cy_full_back = float(cy_local + ry)
    roundtrip_err = float(abs(cx_full_back - cx_full) + abs(cy_full_back - cy_full))

    real_u8 = _norm_u8_robust(real)

    # Allen color map if available, else pseudo-color LUT
    colored = None
    if structure_csv is not None:
        cmap = load_allen_color_map(structure_csv)
        if cmap:
            colored = np.zeros((*label.shape, 3), dtype=np.uint8)
            for rid in np.unique(label.astype(np.int32)):
                if rid == 0:
                    continue
                color = cmap.get(int(rid))
                if color is None:
                    continue
                colored[label == rid] = np.array(color, dtype=np.uint8)

    if colored is None:
        lut = np.array([
            [20, 20, 20],
            [0, 200, 255],
            [0, 255, 120],
            [255, 120, 180],
            [255, 70, 70],
            [220, 220, 80],
            [160, 120, 255],
            [255, 180, 70],
        ], dtype=np.uint8)
        colored = lut[(label.astype(np.int32) % len(lut))]

    if mode == "contour":
        colored = draw_contours(label)
    elif mode == "contour-major":
        colored = draw_contours_major(label, top_k=major_top_k)
        
    overlay = alpha_blend(real_u8, colored, alpha)

    # quality guard: avoid near-black invalid outputs
    if float(np.mean(overlay)) < float(min_mean_threshold):
        raise ValueError(f'overlay quality check failed: near-black output (mean={float(np.mean(overlay)):.2f}, threshold={float(min_mean_threshold):.2f})')

    out_png.parent.mkdir(parents=True, exist_ok=True)
    imwrite(str(out_png), overlay)

    if return_meta:
        return out_png, {
            "real_shape": real_shape,
            "label_shape_before": label_shape_before,
            "label_shape_after": label_shape_after,
            "label_shape_fitted": [int(label_shape_fitted[0]), int(label_shape_fitted[1])],
            "scale": scale,
            "pixelSizeUm": float(pixel_size_um),
            "rotateAtlas": float(rotate_deg),
            "flipAtlas": str(flip_mode),
            "fitMode": str(fit_mode),
            "real_aspect": float(real_shape[1] / max(real_shape[0], 1)),
            "atlas_aspect_before": float(label_shape_before[1] / max(label_shape_before[0], 1)),
            "atlas_aspect": float(label_shape_before[1] / max(label_shape_before[0], 1)),
            "roi_bbox": [int(roi_bbox[0]), int(roi_bbox[1]), int(roi_bbox[2]), int(roi_bbox[3])],
            "roi_center_full": [cx_full, cy_full],
            "roi_roundtrip_error": roundtrip_err,
        }

    return out_png
