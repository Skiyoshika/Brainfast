from __future__ import annotations

from pathlib import Path
import numpy as np
from tifffile import imread, imwrite
from skimage.filters import sobel
from skimage.metrics import structural_similarity as ssim
from skimage.segmentation import find_boundaries
from skimage import morphology
from scripts.slice_select import select_real_slice_2d


def _roi_bbox_from_real(real_img: np.ndarray, pad: int = 4) -> tuple[int, int, int, int]:
    x = real_img.astype(np.float32)
    if x.ndim == 3:
        x = x[..., 0]
    thr = float(np.percentile(x, 88))
    mask = x > thr
    from skimage import measure
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


def _center_pad_or_crop(img: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    th, tw = target_shape
    sh, sw = img.shape
    out = np.zeros((th, tw), dtype=np.float32)
    min_h = min(th, sh)
    min_w = min(tw, sw)
    oy = (th - min_h) // 2
    ox = (tw - min_w) // 2
    iy = (sh - min_h) // 2
    ix = (sw - min_w) // 2
    out[oy:oy + min_h, ox:ox + min_w] = img[iy:iy + min_h, ix:ix + min_w]
    return out


def _prep_real_features(real_img: np.ndarray, target_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    if real_img.ndim == 3:
        real_img = real_img[..., 0]
    x = _center_pad_or_crop(real_img.astype(np.float32), target_shape)
    edges = sobel(x)

    # Build a robust real tissue mask in atlas-space for shape-aware scoring
    p80 = float(np.percentile(x, 80))
    p65 = float(np.percentile(x, 65))
    mask = x > p80
    if float(np.mean(mask)) < 0.01:
        mask = x > p65
    return edges.astype(np.float32), mask.astype(bool)


def _centroid(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(mask)
    if len(ys) == 0:
        return None
    return float(np.mean(ys)), float(np.mean(xs))


def _shift_with_zero(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    h, w = img.shape
    out = np.zeros_like(img)

    sy0 = max(0, -dy)
    sy1 = min(h, h - dy)  # exclusive
    sx0 = max(0, -dx)
    sx1 = min(w, w - dx)  # exclusive
    if sy1 <= sy0 or sx1 <= sx0:
        return out

    ty0 = sy0 + dy
    ty1 = sy1 + dy
    tx0 = sx0 + dx
    tx1 = sx1 + dx
    out[ty0:ty1, tx0:tx1] = img[sy0:sy1, sx0:sx1]
    return out


def _safe_dice(a: np.ndarray, b: np.ndarray) -> float:
    aa = (a > 0)
    bb = (b > 0)
    den = float(np.sum(aa) + np.sum(bb)) + 1e-6
    inter = float(np.sum(aa & bb))
    return 2.0 * inter / den


def _norm_corr(a: np.ndarray, b: np.ndarray) -> float:
    af = a.astype(np.float32).ravel()
    bf = b.astype(np.float32).ravel()
    den = float(np.linalg.norm(af) * np.linalg.norm(bf)) + 1e-6
    return float(np.dot(af, bf) / den)


def _label_edge_score(real_edges: np.ndarray, real_mask: np.ndarray, label_slice: np.ndarray) -> float:
    # Skip nearly-empty atlas slices
    coverage = float(np.mean(label_slice > 0))
    if coverage < 0.015:
        return -1.0

    atlas_mask = (label_slice > 0).astype(np.float32)
    outer = sobel(atlas_mask)
    inner = find_boundaries(label_slice.astype(np.int32), mode="inner", connectivity=2).astype(np.float32)
    atlas_feat = 0.55 * outer + 1.00 * inner

    # Coarse centroid alignment to reduce translation sensitivity during scoring
    c_real = _centroid(real_mask)
    c_atlas = _centroid(atlas_mask > 0)
    if c_real is not None and c_atlas is not None:
        dy = int(round(c_real[0] - c_atlas[0]))
        dx = int(round(c_real[1] - c_atlas[1]))
        atlas_feat = _shift_with_zero(atlas_feat, dy=dy, dx=dx)
        atlas_mask = _shift_with_zero(atlas_mask, dy=dy, dx=dx)
        inner = _shift_with_zero(inner, dy=dy, dx=dx)

    real_signal = float(np.mean(real_edges > (np.percentile(real_edges, 70))))
    if real_signal < 0.004:
        return 0.6 * coverage + 0.4 * _safe_dice(real_mask, atlas_mask > 0)

    dr = float(max(np.ptp(real_edges), np.ptp(atlas_feat), 1e-6))
    ssim_score = float(ssim(real_edges, atlas_feat, data_range=dr))
    shape_dice = _safe_dice(real_mask, atlas_mask > 0)
    inner_corr = _norm_corr(real_edges, inner)
    return float(0.70 * ssim_score + 0.20 * shape_dice + 0.10 * inner_corr + 0.03 * coverage)


def _candidate_zs_from_coarse(
    coarse_scores: list[tuple[int, float]],
    z_dim: int,
    base_step: int,
    max_seeds: int = 10,
) -> list[int]:
    if not coarse_scores:
        return [z_dim // 2]
    sorted_scores = sorted(coarse_scores, key=lambda x: x[1], reverse=True)
    seeds = [int(z) for z, _ in sorted_scores[:max(1, int(max_seeds))]]
    radius = max(1, int(base_step))
    cand = set()
    for z in seeds:
        for dz in range(-radius, radius + 1):
            zz = z + dz
            if 0 <= zz < z_dim:
                cand.add(int(zz))
    return sorted(cand)


def autopick_best_z(
    real_path: Path,
    annotation_nii: Path,
    out_label_tif: Path,
    z_step: int = 1,
    pixel_size_um: float = 0.65,
    slicing_plane: str = "coronal",
    roi_mode: str = "auto",
    real_z_index: int | None = None,
) -> dict:
    import nibabel as nib
    from skimage.transform import rescale

    real_raw = imread(str(real_path))
    real, real_slice_meta = select_real_slice_2d(
        real_raw,
        z_index=real_z_index,
        source_path=real_path,
    )
    nii = nib.load(str(annotation_nii))
    vol = np.asarray(nii.get_fdata(), dtype=np.int32)

    plane = str(slicing_plane or "coronal").lower()

    def _get_slice(v: np.ndarray, z: int, p: str) -> np.ndarray:
        if p == "coronal":
            return v[z, :, :]
        if p == "sagittal":
            return v[:, :, z]
        if p in ("horizontal", "axial"):
            return v[:, z, :]
        raise ValueError(f"unsupported slicing_plane: {p}")

    if plane == "coronal":
        z_dim = vol.shape[0]
    elif plane == "sagittal":
        z_dim = vol.shape[2]
    elif plane in ("horizontal", "axial"):
        z_dim = vol.shape[1]
    else:
        raise ValueError(f"unsupported slicing_plane: {plane}")

    sample = _get_slice(vol, 0, plane)
    target_shape = sample.shape

    roi_bbox = (0, 0, int(real.shape[1]), int(real.shape[0]))
    roi = real
    if str(roi_mode or "off").lower() in ("auto", "on", "true"):
        x0, y0, rw, rh = _roi_bbox_from_real(real)
        roi_bbox = (int(x0), int(y0), int(rw), int(rh))
        roi = real[y0:y0 + rh, x0:x0 + rw]

    # Pre-scale real ROI to atlas space
    scale = pixel_size_um / 25.0
    real_f = roi.astype(np.float32)
    real_scaled = rescale(real_f, scale, order=1, preserve_range=True, anti_aliasing=True)
    real_edges, real_mask = _prep_real_features(real_scaled, target_shape)

    best_z, best_score = 0, -1.0
    coarse_scores: list[tuple[int, float]] = []
    step = max(1, int(z_step))
    for z in range(0, z_dim, step):
        atlas_slice = _get_slice(vol, z, plane)
        s = _label_edge_score(real_edges, real_mask, atlas_slice)
        coarse_scores.append((int(z), float(s)))
        if s > best_score:
            best_score = s
            best_z = z

    # Stage-2 refined z evaluation: run fast tissue-guided warp scoring on top candidates.
    refined_best_z = int(best_z)
    refined_best_score = float(best_score)
    refined_scores: list[tuple[int, float]] = []
    try:
        from scripts.overlay_render import _tissue_guided_warp, _norm_u8_robust, _alignment_quality

        real_u8 = _norm_u8_robust(real)
        candidates = _candidate_zs_from_coarse(
            coarse_scores,
            z_dim=z_dim,
            base_step=step,
            max_seeds=10,
        )
        for z in candidates:
            atlas_slice = _get_slice(vol, int(z), plane).astype(np.int32)
            warped, meta = _tissue_guided_warp(
                real,
                atlas_slice,
                atlas_res_um=25.0,
                real_res_um=float(pixel_size_um),
                fit_mode="contain",
                enable_nonlinear=False,
                opt_maxiter=70,
            )
            tissue_mask = meta.get("tissue_mask")
            if tissue_mask is not None:
                clip = morphology.dilation(tissue_mask.astype(np.uint8), morphology.disk(3)).astype(bool)
                warped = np.where(clip, warped, 0).astype(np.int32)
            q = float(_alignment_quality(real_u8, warped, tissue_mask=tissue_mask))
            refined_scores.append((int(z), q))
            if q > refined_best_score:
                refined_best_score = q
                refined_best_z = int(z)
    except Exception:
        refined_scores = []

    if refined_scores:
        refined_best_z, refined_best_score = max(refined_scores, key=lambda x: x[1])
    best_z = int(refined_best_z)
    best_score = float(refined_best_score)

    best_slice = _get_slice(vol, best_z, plane).astype(np.int32)
    out_label_tif.parent.mkdir(parents=True, exist_ok=True)
    imwrite(str(out_label_tif), best_slice)

    return {
        "best_z": int(best_z),
        "best_score": float(best_score),
        "best_score_type": "refined_warp_quality" if refined_scores else "coarse_edge_score",
        "label_slice_tif": str(out_label_tif),
        "shape": [int(x) for x in vol.shape],
        "slicing_plane": plane,
        "slice_shape": [int(x) for x in best_slice.shape],
        "roi_mode": str(roi_mode),
        "roi_bbox": [int(roi_bbox[0]), int(roi_bbox[1]), int(roi_bbox[2]), int(roi_bbox[3])],
        "real_slice": real_slice_meta,
        "coarse_top": [[int(z), float(s)] for z, s in sorted(coarse_scores, key=lambda x: x[1], reverse=True)[:8]],
        "refined_top": [[int(z), float(s)] for z, s in sorted(refined_scores, key=lambda x: x[1], reverse=True)[:8]],
    }
