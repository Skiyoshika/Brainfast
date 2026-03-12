from __future__ import annotations

from pathlib import Path
import json
import numpy as np
from tifffile import imread, imwrite
from skimage.transform import rescale, rotate, resize, SimilarityTransform, PiecewiseAffineTransform, warp as skwarp
from skimage import measure, morphology
from skimage.filters import sobel
from skimage.metrics import structural_similarity as ssim
from skimage.registration import optical_flow_tvl1
from skimage.segmentation import find_boundaries
from scipy.ndimage import gaussian_filter, map_coordinates, distance_transform_edt
from scripts.allen_colors import load_allen_color_map
from scripts.slice_select import select_real_slice_2d, select_label_slice_2d

# ── Allen structure tree (region names + official colors) ───────────────────
_STRUCTURE_TREE: dict | None = None

def _load_structure_tree() -> dict:
    global _STRUCTURE_TREE
    if _STRUCTURE_TREE is not None:
        return _STRUCTURE_TREE
    # Try project configs, then same dir as this file
    candidates = [
        Path(__file__).resolve().parent.parent / "configs" / "allen_structure_tree.json",
        Path(__file__).resolve().parent / "allen_structure_tree.json",
    ]
    for p in candidates:
        if p.exists():
            with open(p, encoding="utf-8") as f:
                _STRUCTURE_TREE = json.load(f)
            return _STRUCTURE_TREE
    _STRUCTURE_TREE = {}
    return _STRUCTURE_TREE


# ── Utilities ──────────────────────────────────────────────────────────────

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


def _detect_tissue(real: np.ndarray) -> dict:
    """Find the dominant tissue region in a real-space image."""
    rf = real.astype(np.float32)
    if rf.ndim == 3:
        rf = rf[..., 0]
    rh, rw = rf.shape

    p2, p98 = np.percentile(rf, 2), np.percentile(rf, 98)
    if p98 - p2 < 1:
        return {"ok": False}

    thr = p2 + (p98 - p2) * 0.12
    raw_mask = rf > thr

    try:
        closed = morphology.closing(raw_mask, morphology.disk(max(3, rw // 300)))
        lbl_tmp = measure.label(closed)
        min_area = max(500, rh * rw // 2000)
        areas = {r.label: r.area for r in measure.regionprops(lbl_tmp)}
        cleaned = np.isin(lbl_tmp, [k for k, v in areas.items() if v >= min_area])
    except Exception:
        cleaned = raw_mask

    labeled = measure.label(cleaned, connectivity=2)
    props = sorted(measure.regionprops(labeled), key=lambda r: r.area, reverse=True)
    if not props:
        return {"ok": False}

    tissue = props[0]
    ty0, tx0, ty1, tx1 = tissue.bbox
    t_cy, t_cx = tissue.centroid
    t_h, t_w = ty1 - ty0, tx1 - tx0

    # Build tight tissue mask (largest region only)
    tight_mask = labeled == tissue.label
    major_axis = getattr(tissue, "axis_major_length", None)
    if major_axis is None:
        major_axis = getattr(tissue, "major_axis_length", 1.0)
    minor_axis = getattr(tissue, "axis_minor_length", None)
    if minor_axis is None:
        minor_axis = getattr(tissue, "minor_axis_length", 1.0)

    return {
        "ok": True,
        "bbox": (int(ty0), int(tx0), int(ty1), int(tx1)),
        "centroid": (float(t_cy), float(t_cx)),
        "hw": (int(t_h), int(t_w)),
        "orientation": float(tissue.orientation),      # radians, skimage convention
        "major_axis": float(major_axis),
        "minor_axis": float(minor_axis),
        "mask": tight_mask,                            # bool array, real image space
    }


def _atlas_bbox(label: np.ndarray) -> tuple[int, int, int, int]:
    """Return (y0, x0, y1, x1) of the non-zero region of an atlas label."""
    nnz = np.argwhere(label > 0)
    if len(nnz) == 0:
        return (0, 0, label.shape[0], label.shape[1])
    ay0, ax0 = nnz.min(axis=0)
    ay1, ax1 = nnz.max(axis=0)
    return int(ay0), int(ax0), int(ay1), int(ax1)


def _similarity_warp(
    label: np.ndarray,
    a_cy: float, a_cx: float, a_h: float, a_w: float,
    t_cy: float, t_cx: float, t_h: float, t_w: float,
    phys_scale: float,
    fit_mode: str,
    out_shape: tuple[int, int],
) -> np.ndarray:
    """Apply similarity transform to warp atlas label into real image space."""
    # Fit scale: stretch atlas non-zero region to match tissue region
    fit_h = t_h / max(a_h * phys_scale, 1.0)
    fit_w = t_w / max(a_w * phys_scale, 1.0)

    mode = str(fit_mode or "contain").lower()
    if mode == "contain":
        fit_scale = min(fit_h, fit_w)
    elif mode == "cover":
        fit_scale = max(fit_h, fit_w)
    elif mode == "width-lock":
        fit_scale = fit_w
    elif mode == "height-lock":
        fit_scale = fit_h
    else:
        fit_scale = (fit_h * fit_w) ** 0.5

    total_scale = phys_scale * fit_scale

    # Translation: align atlas centroid → tissue centroid
    t_dy = t_cy - a_cy * total_scale
    t_dx = t_cx - a_cx * total_scale

    # skwarp expects inverse_map: output_coords → input_coords
    # forward: atlas → real  (scale up by total_scale)
    # inverse: real  → atlas (scale down by 1/total_scale)
    inv_scale = 1.0 / total_scale
    inverse = SimilarityTransform(
        scale=inv_scale,
        translation=(-t_dx * inv_scale, -t_dy * inv_scale),
    )
    warped = skwarp(
        label.astype(np.float32),
        inverse,
        output_shape=out_shape,
        order=0,
        preserve_range=True,
        mode="constant",
        cval=0.0,
    )
    return warped.astype(np.int32), total_scale, t_dx, t_dy


def _inner_boundaries_fast(label: np.ndarray) -> np.ndarray:
    """Fast inner-boundary approximation for integer label images."""
    li = label.astype(np.int32, copy=False)
    nz = li > 0
    bd = np.zeros(li.shape, dtype=bool)

    d_h = (li[:, 1:] != li[:, :-1]) & nz[:, 1:] & nz[:, :-1]
    bd[:, 1:] |= d_h
    bd[:, :-1] |= d_h

    d_v = (li[1:, :] != li[:-1, :]) & nz[1:, :] & nz[:-1, :]
    bd[1:, :] |= d_v
    bd[:-1, :] |= d_v

    d_d1 = (li[1:, 1:] != li[:-1, :-1]) & nz[1:, 1:] & nz[:-1, :-1]
    bd[1:, 1:] |= d_d1
    bd[:-1, :-1] |= d_d1

    d_d2 = (li[1:, :-1] != li[:-1, 1:]) & nz[1:, :-1] & nz[:-1, 1:]
    bd[1:, :-1] |= d_d2
    bd[:-1, 1:] |= d_d2

    return bd.astype(np.float32)


def _atlas_edge_feature_fast(label: np.ndarray) -> np.ndarray:
    """Combined outer+inner atlas boundary feature."""
    atlas_mask = (label > 0).astype(np.float32)
    outer = sobel(atlas_mask)
    inner = _inner_boundaries_fast(label)
    feat = 0.55 * outer + 1.00 * inner
    mx = float(np.max(feat))
    if mx > 0:
        feat /= mx
    return feat.astype(np.float32)


def _edge_overlap_score(real_u8: np.ndarray, warped_label: np.ndarray) -> float:
    """Quick quality score: edge correlation with outer+inner atlas boundaries."""
    real_e = sobel(real_u8.astype(np.float32) / 255.0)
    atlas_e = _atlas_edge_feature_fast(warped_label.astype(np.int32))
    dr = float(max(np.ptp(real_e), np.ptp(atlas_e), 1e-6))
    # Downsample for speed
    s = 4
    re = real_e[::s, ::s]
    ae = atlas_e[::s, ::s]
    try:
        return float(ssim(re, ae, data_range=dr))
    except Exception:
        return float(np.corrcoef(re.ravel(), ae.ravel())[0, 1])


def _optimize_warp(
    real_u8: np.ndarray,
    atlas_label: np.ndarray,
    init_scale: float,
    init_dx: float,
    init_dy: float,
    out_shape: tuple[int, int],
    ds: int = 4,
    maxiter: int = 150,
    init_angle_deg: float = 0.0,
) -> tuple[float, float, float, float]:
    """
    Refine (scale, angle, dx, dy) by maximizing atlas-boundary ↔ real-edge overlap.
    Works on downsampled images (ds×) for speed.
    Returns (refined_scale, refined_angle_rad, refined_dx, refined_dy).
    """
    from scipy.optimize import minimize

    rh, rw = out_shape
    real_ds = real_u8[::ds, ::ds].astype(np.float32) / 255.0
    atlas_ds = atlas_label[::ds, ::ds].astype(np.float32)  # pre-slice once
    ds_h, ds_w = real_ds.shape
    real_e = sobel(real_ds)
    p = float(np.percentile(real_e, 66))
    real_e = np.clip((real_e - p) / (float(np.max(real_e)) - p + 1e-6), 0.0, 1.0)

    # Build tissue mask from real image (downsampled)
    tissue_mask = real_ds > 0.05
    # Use tight tissue mask from full-res detection if available; else derive from downsampled
    from skimage.morphology import disk as sk_disk
    # Small dilation (1px at ds=4 ≈ 4px real) to allow for boundary noise
    dilated = morphology.dilation(tissue_mask.astype(np.uint8), sk_disk(1)).astype(bool)

    def _warp_score(params):
        sf, angle_deg, dx_adj, dy_adj = params
        # Clamp to valid ranges (Nelder-Mead ignores bounds otherwise)
        sf = float(np.clip(sf, 0.7, 1.4))
        angle_deg = float(np.clip(angle_deg, -18.0, 18.0))
        dx_adj = float(np.clip(dx_adj, -300.0, 300.0))
        dy_adj = float(np.clip(dy_adj, -300.0, 300.0))
        s = init_scale * sf
        angle_rad = np.deg2rad(angle_deg)
        dx = init_dx + dx_adj
        dy = init_dy + dy_adj
        # Both atlas and real are downsampled by ds, so scale is unchanged;
        # only translation (in real-image pixels) is divided by ds
        fwd = SimilarityTransform(scale=s, rotation=angle_rad,
                                  translation=(dx / ds, dy / ds))
        try:
            w = skwarp(
                atlas_ds,
                fwd.inverse,
                output_shape=(ds_h, ds_w),
                order=0, preserve_range=True, mode="constant", cval=0.0,
            )
        except Exception:
            return 0.0
        wi = np.rint(w).astype(np.int32)
        atlas_mask = (wi > 0).astype(np.float32)
        inner = _inner_boundaries_fast(wi)
        outer = sobel(atlas_mask)
        inner_ov = float(np.sum(inner * real_e))
        outer_ov = float(np.sum(outer * real_e))
        overlap = 1.00 * inner_ov + 0.55 * outer_ov
        # Strong penalty: atlas pixels outside tissue (prevents atlas from growing past tissue edge)
        outside = float(np.sum(atlas_mask * (~dilated).astype(float)))
        atlas_total = float(np.sum(atlas_mask)) + 1e-6
        outside_frac = outside / atlas_total  # fraction of atlas outside tissue [0..1]
        return -(overlap - 2.2 * outside_frac * overlap)

    x0 = [1.0, float(init_angle_deg), 0.0, 0.0]  # sf=1, angle=init, dx_adj=0, dy_adj=0
    result = minimize(
        _warp_score, x0, method="Nelder-Mead",
        options={"maxiter": maxiter, "xatol": 0.3, "fatol": 1e-6, "adaptive": True},
    )
    sf, angle_deg, dx_adj, dy_adj = result.x
    sf = float(np.clip(sf, 0.7, 1.4))
    angle_deg = float(np.clip(angle_deg, -18.0, 18.0))
    dx_adj = float(np.clip(dx_adj, -300.0, 300.0))
    dy_adj = float(np.clip(dy_adj, -300.0, 300.0))
    return (
        float(init_scale * sf),
        float(np.deg2rad(angle_deg)),
        float(init_dx + dx_adj),
        float(init_dy + dy_adj),
    )


def _apply_warp(
    label: np.ndarray,
    scale: float, angle_rad: float, dx: float, dy: float,
    out_shape: tuple[int, int],
) -> np.ndarray:
    """Apply similarity transform (scale, rotation, translation) to atlas label."""
    fwd = SimilarityTransform(scale=scale, rotation=angle_rad, translation=(dx, dy))
    warped = skwarp(
        label.astype(np.float32),
        fwd.inverse,
        output_shape=out_shape,
        order=0, preserve_range=True, mode="constant", cval=0.0,
    )
    return warped.astype(np.int32)


def _atlas_boundary_feature(label: np.ndarray) -> np.ndarray:
    """Build a dense feature map that emphasizes both outer + inner atlas boundaries."""
    outer = find_boundaries(label > 0, mode="outer", connectivity=2).astype(np.float32)
    inner = find_boundaries(label.astype(np.int32), mode="inner", connectivity=2).astype(np.float32)
    feat = 0.55 * outer + 1.00 * inner
    feat = gaussian_filter(feat, sigma=0.9)
    mx = float(np.max(feat))
    if mx > 0:
        feat /= mx
    return feat.astype(np.float32)


def _real_edge_feature(real_u8: np.ndarray, tissue_mask: np.ndarray | None = None) -> np.ndarray:
    """Real-image edge feature map used for non-linear refinement."""
    x = real_u8.astype(np.float32) / 255.0
    e = sobel(x)
    p = float(np.percentile(e, 68))
    e = np.clip((e - p) / (float(np.max(e)) - p + 1e-6), 0.0, 1.0)
    if tissue_mask is not None:
        support = morphology.dilation(tissue_mask.astype(np.uint8), morphology.disk(4)).astype(np.float32)
        e *= support
    return e.astype(np.float32)


def _warp_label_with_flow(label: np.ndarray, flow_v: np.ndarray, flow_u: np.ndarray) -> np.ndarray:
    """Warp label image with dense flow using nearest-neighbor sampling."""
    h, w = label.shape
    rr, cc = np.meshgrid(
        np.arange(h, dtype=np.float32),
        np.arange(w, dtype=np.float32),
        indexing="ij",
    )
    src_r = np.clip(rr + flow_v.astype(np.float32), 0.0, float(h - 1))
    src_c = np.clip(cc + flow_u.astype(np.float32), 0.0, float(w - 1))
    warped = map_coordinates(
        label.astype(np.float32),
        [src_r, src_c],
        order=0,
        mode="constant",
        cval=0.0,
    )
    return warped.astype(np.int32)


def _mask_dice(a: np.ndarray, b: np.ndarray) -> float:
    aa = (a > 0)
    bb = (b > 0)
    den = float(np.sum(aa) + np.sum(bb)) + 1e-6
    inter = float(np.sum(aa & bb))
    return 2.0 * inter / den


def _alignment_quality(real_u8: np.ndarray, label: np.ndarray, tissue_mask: np.ndarray | None = None) -> float:
    """Composite quality score: edge overlap + outer mask agreement."""
    edge_score = _edge_overlap_score(real_u8, label)
    if tissue_mask is None:
        return float(edge_score)
    dice = _mask_dice(label > 0, tissue_mask)
    return float(edge_score + 0.35 * dice)


def _clip_label_to_tissue(label: np.ndarray, tissue_mask: np.ndarray | None, pad: int = 5) -> np.ndarray:
    if tissue_mask is None:
        return label.astype(np.int32)
    clip = morphology.dilation(tissue_mask.astype(np.uint8), morphology.disk(max(1, int(pad)))).astype(bool)
    return np.where(clip, label, 0).astype(np.int32)


def _sample_mask_points(mask: np.ndarray, max_points: int) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    n = len(ys)
    if n == 0:
        return np.zeros((0, 2), dtype=np.int32)
    if n > max_points:
        idx = np.linspace(0, n - 1, int(max_points), dtype=np.int64)
        ys = ys[idx]
        xs = xs[idx]
    return np.stack([ys, xs], axis=1).astype(np.int32)


def _refine_warp_flow_candidate(
    real_u8: np.ndarray,
    label: np.ndarray,
    tissue_mask: np.ndarray | None,
    q_before: float,
) -> tuple[np.ndarray, dict]:
    h, w = label.shape
    if h < 40 or w < 40:
        return label, {"ok": False, "reason": "small_image"}

    if max(h, w) >= 1800:
        ds = 4
    elif max(h, w) >= 1100:
        ds = 3
    else:
        ds = 2

    out_h = max(64, h // ds)
    out_w = max(64, w // ds)
    real_feat = _real_edge_feature(real_u8, tissue_mask=tissue_mask)
    atlas_feat = _atlas_boundary_feature(label)

    real_ds = resize(
        real_feat,
        (out_h, out_w),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)
    atlas_ds = resize(
        atlas_feat,
        (out_h, out_w),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)

    def _build_candidate(
        flow_v_ds: np.ndarray,
        flow_u_ds: np.ndarray,
        sign: float,
        tag: str,
    ) -> tuple[np.ndarray, float, dict]:
        sy = float(h) / float(out_h)
        sx = float(w) / float(out_w)
        flow_v = resize(
            flow_v_ds,
            (h, w),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32) * sy * float(sign)
        flow_u = resize(
            flow_u_ds,
            (h, w),
            order=1,
            preserve_range=True,
            anti_aliasing=True,
        ).astype(np.float32) * sx * float(sign)

        flow_v = gaussian_filter(flow_v, sigma=1.6).astype(np.float32)
        flow_u = gaussian_filter(flow_u, sigma=1.6).astype(np.float32)

        if tissue_mask is not None:
            support = morphology.dilation(tissue_mask.astype(np.uint8), morphology.disk(7)).astype(np.float32)
            flow_v *= support
            flow_u *= support

        max_disp = float(max(10.0, min(h, w) * 0.10))
        mag = np.sqrt(flow_v * flow_v + flow_u * flow_u).astype(np.float32)
        scale = np.minimum(1.0, max_disp / (mag + 1e-6)).astype(np.float32)
        flow_v *= scale
        flow_u *= scale

        refined = _warp_label_with_flow(label, flow_v, flow_u)
        refined = _clip_label_to_tissue(refined, tissue_mask, pad=5)
        q_after = _alignment_quality(real_u8, refined, tissue_mask=tissue_mask)
        return refined, float(q_after), {
            "tag": str(tag),
            "downsample": int(ds),
            "max_displacement_px": float(max_disp),
            "score_after": float(q_after),
        }

    candidates: list[tuple[np.ndarray, float, dict]] = []
    try:
        fv_ra, fu_ra = optical_flow_tvl1(
            real_ds,
            atlas_ds,
            attachment=8.0,
            tightness=0.42,
            num_warp=8,
            num_iter=50,
            tol=1e-4,
            prefilter=True,
            dtype=np.float32,
        )
        candidates.append(_build_candidate(fv_ra, fu_ra, 1.0, "flow_real_atlas"))
    except Exception as e:
        pass

    try:
        fv_ar, fu_ar = optical_flow_tvl1(
            atlas_ds,
            real_ds,
            attachment=8.0,
            tightness=0.42,
            num_warp=8,
            num_iter=50,
            tol=1e-4,
            prefilter=True,
            dtype=np.float32,
        )
        candidates.append(_build_candidate(fv_ar, fu_ar, -1.0, "flow_atlas_real_neg"))
        candidates.append(_build_candidate(fv_ar, fu_ar, 1.0, "flow_atlas_real_pos"))
    except Exception:
        pass

    if not candidates:
        return label, {"ok": False, "reason": "flow_failed"}

    best_label, best_q, best_meta = max(candidates, key=lambda x: x[1])
    best_meta = {
        "ok": True,
        "method": "flow",
        "score_before": float(q_before),
        "score_after": float(best_q),
        "score_delta": float(best_q - q_before),
        **best_meta,
    }
    return best_label, best_meta


def _refine_warp_liquify_candidate(
    real_u8: np.ndarray,
    label: np.ndarray,
    tissue_mask: np.ndarray | None,
    q_before: float,
) -> tuple[np.ndarray, dict]:
    h, w = label.shape
    if h < 60 or w < 60 or float(np.mean(label > 0)) < 0.01:
        return label, {"ok": False, "reason": "insufficient_label_coverage"}

    real_feat = _real_edge_feature(real_u8, tissue_mask=tissue_mask)
    nz = real_feat > 0
    if float(np.mean(nz)) < 0.0005:
        return label, {"ok": False, "reason": "weak_real_edges"}
    thr = float(np.percentile(real_feat[nz], 72)) if np.any(nz) else 0.2
    real_edge = real_feat >= max(0.10, thr)
    if tissue_mask is not None:
        support = morphology.dilation(tissue_mask.astype(np.uint8), morphology.disk(8)).astype(bool)
        real_edge &= support
    if float(np.mean(real_edge)) < 0.001:
        return label, {"ok": False, "reason": "too_few_real_edges"}

    outer = find_boundaries(label > 0, mode="outer", connectivity=2)
    inner = find_boundaries(label.astype(np.int32), mode="inner", connectivity=2)
    src_inner = _sample_mask_points(inner, max_points=2400)
    src_outer = _sample_mask_points(outer, max_points=1200)
    if len(src_inner) == 0 and len(src_outer) == 0:
        return label, {"ok": False, "reason": "no_atlas_boundaries"}
    src_rc = np.vstack([src_inner, src_outer]) if len(src_inner) and len(src_outer) else (
        src_inner if len(src_inner) else src_outer
    )

    dist, inds = distance_transform_edt(~real_edge, return_indices=True)
    yy = src_rc[:, 0]
    xx = src_rc[:, 1]
    ny = inds[0, yy, xx]
    nx = inds[1, yy, xx]
    d = dist[yy, xx]

    max_match_dist = float(max(10.0, min(h, w) * 0.045))
    keep = d <= max_match_dist
    if tissue_mask is not None:
        support2 = morphology.dilation(tissue_mask.astype(np.uint8), morphology.disk(10)).astype(bool)
        keep &= support2[ny, nx]
    if int(np.sum(keep)) < 24:
        return label, {"ok": False, "reason": "too_few_matches"}

    src_xy = np.stack([xx[keep], yy[keep]], axis=1).astype(np.float32)
    dst_xy = np.stack([nx[keep], ny[keep]], axis=1).astype(np.float32)

    max_ctrl = 2600
    if len(src_xy) > max_ctrl:
        idx = np.linspace(0, len(src_xy) - 1, max_ctrl, dtype=np.int64)
        src_xy = src_xy[idx]
        dst_xy = dst_xy[idx]
    pair_key = np.round(np.concatenate([src_xy, dst_xy], axis=1), 1)
    _, uniq_idx = np.unique(pair_key, axis=0, return_index=True)
    uniq_idx = np.sort(uniq_idx)
    src_xy = src_xy[uniq_idx]
    dst_xy = dst_xy[uniq_idx]
    if len(src_xy) < 20:
        return label, {"ok": False, "reason": "too_few_unique_matches"}

    anchors = np.array([
        [0.0, 0.0], [w - 1.0, 0.0], [0.0, h - 1.0], [w - 1.0, h - 1.0],
        [w * 0.5, 0.0], [w * 0.5, h - 1.0], [0.0, h * 0.5], [w - 1.0, h * 0.5],
    ], dtype=np.float32)
    src_all = np.vstack([src_xy, anchors]).astype(np.float32)
    dst_all = np.vstack([dst_xy, anchors]).astype(np.float32)
    refined = None
    method = "liquify_pwa"
    try:
        tform = PiecewiseAffineTransform.from_estimate(src_all, dst_all)
        refined = skwarp(
            label.astype(np.float32),
            tform.inverse,
            output_shape=(h, w),
            order=0,
            preserve_range=True,
            mode="constant",
            cval=0.0,
        ).astype(np.int32)
    except Exception:
        # Fallback: build a smooth dense displacement field from sparse matches.
        method = "liquify_dense_field"
        disp_u = np.zeros((h, w), dtype=np.float32)
        disp_v = np.zeros((h, w), dtype=np.float32)
        wt = np.zeros((h, w), dtype=np.float32)
        sx_i = np.clip(np.rint(src_xy[:, 0]).astype(np.int32), 0, w - 1)
        sy_i = np.clip(np.rint(src_xy[:, 1]).astype(np.int32), 0, h - 1)
        du = (dst_xy[:, 0] - src_xy[:, 0]).astype(np.float32)
        dv = (dst_xy[:, 1] - src_xy[:, 1]).astype(np.float32)
        np.add.at(disp_u, (sy_i, sx_i), du)
        np.add.at(disp_v, (sy_i, sx_i), dv)
        np.add.at(wt, (sy_i, sx_i), 1.0)
        sigma = float(max(6.0, min(h, w) * 0.020))
        den = gaussian_filter(wt, sigma=sigma) + 1e-4
        disp_u = gaussian_filter(disp_u, sigma=sigma) / den
        disp_v = gaussian_filter(disp_v, sigma=sigma) / den
        if tissue_mask is not None:
            support3 = morphology.dilation(tissue_mask.astype(np.uint8), morphology.disk(10)).astype(np.float32)
            disp_u *= support3
            disp_v *= support3
        max_disp2 = float(max(8.0, min(h, w) * 0.06))
        mag2 = np.sqrt(disp_u * disp_u + disp_v * disp_v) + 1e-6
        scale2 = np.minimum(1.0, max_disp2 / mag2).astype(np.float32)
        disp_u *= scale2
        disp_v *= scale2
        refined = _warp_label_with_flow(label, disp_v, disp_u)

    if refined is None:
        return label, {"ok": False, "reason": "liquify_failed"}
    refined = _clip_label_to_tissue(refined, tissue_mask, pad=5)

    q_after = _alignment_quality(real_u8, refined, tissue_mask=tissue_mask)
    return refined, {
        "ok": True,
        "method": method,
        "score_before": float(q_before),
        "score_after": float(q_after),
        "score_delta": float(q_after - q_before),
        "matches": int(len(src_xy)),
        "max_match_dist_px": float(max_match_dist),
    }


def _refine_warp_nonlinear(
    real_u8: np.ndarray,
    label: np.ndarray,
    tissue_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Non-linear refinement: evaluate multiple deformers and keep the best."""
    q_before = _alignment_quality(real_u8, label, tissue_mask=tissue_mask)

    flow_label, flow_meta = _refine_warp_flow_candidate(
        real_u8,
        label,
        tissue_mask=tissue_mask,
        q_before=q_before,
    )
    liquify_label, liquify_meta = _refine_warp_liquify_candidate(
        real_u8,
        label,
        tissue_mask=tissue_mask,
        q_before=q_before,
    )
    hybrid_label = label
    hybrid_meta: dict = {"ok": False, "reason": "flow_or_liquify_unavailable"}
    if flow_meta.get("ok"):
        q_flow = float(flow_meta.get("score_after", q_before))
        h_label, h_meta = _refine_warp_liquify_candidate(
            real_u8,
            flow_label,
            tissue_mask=tissue_mask,
            q_before=q_flow,
        )
        if h_meta.get("ok"):
            hybrid_label = h_label
            hybrid_meta = {
                "ok": True,
                "method": "liquify_after_flow",
                "score_before": float(q_before),
                "score_after": float(h_meta.get("score_after", q_flow)),
                "score_delta": float(h_meta.get("score_after", q_flow) - q_before),
                "inner": h_meta,
            }

    candidates: list[tuple[str, np.ndarray, float]] = [("baseline", label, float(q_before))]
    if flow_meta.get("ok"):
        candidates.append(("flow", flow_label, float(flow_meta.get("score_after", -1e9))))
    if liquify_meta.get("ok"):
        candidates.append(("liquify", liquify_label, float(liquify_meta.get("score_after", -1e9))))
    if hybrid_meta.get("ok"):
        candidates.append(("liquify_after_flow", hybrid_label, float(hybrid_meta.get("score_after", -1e9))))

    best_method, best_label, best_q = max(candidates, key=lambda x: x[2])
    best_q = float(best_q)

    # Prefer true liquify deformation when quality is close to the best candidate.
    liq_like = [c for c in candidates if c[0] in ("liquify", "liquify_after_flow")]
    if liq_like:
        l_method, l_label, l_q = max(liq_like, key=lambda x: x[2])
        l_q = float(l_q)
        if l_q >= best_q - 0.010 and l_q >= q_before + 0.010:
            best_method, best_label, best_q = l_method, l_label, l_q

    if best_method == "baseline":
        return label, {
            "applied": False,
            "reason": "no_quality_gain",
            "score_before": float(q_before),
            "score_after": float(q_before),
            "score_delta": 0.0,
            "flow": flow_meta,
            "liquify": liquify_meta,
            "hybrid": hybrid_meta,
        }

    # Allow slight local tradeoff for internal fit, but avoid visibly worse global alignment.
    if best_q < q_before - 0.006:
        return label, {
            "applied": False,
            "reason": "quality_drop_too_large",
            "score_before": float(q_before),
            "score_after": float(best_q),
            "score_delta": float(best_q - q_before),
            "flow": flow_meta,
            "liquify": liquify_meta,
            "hybrid": hybrid_meta,
        }

    return best_label.astype(np.int32), {
        "applied": True,
        "method": str(best_method),
        "score_before": float(q_before),
        "score_after": float(best_q),
        "score_delta": float(best_q - q_before),
        "flow": flow_meta,
        "liquify": liquify_meta,
        "hybrid": hybrid_meta,
    }


# ── Region label drawing ─────────────────────────────────────────────────────

def draw_region_labels(
    overlay: np.ndarray,
    warped_label: np.ndarray,
    min_area_px: int = 5000,
    max_labels: int = 25,
    font_scale: float = 0.75,
    text_color: tuple = (255, 255, 255),
    shadow_color: tuple = (0, 0, 0),
) -> np.ndarray:
    """Draw Allen region acronyms on the overlay at each region's centroid."""
    try:
        import cv2 as _cv2
        _HAS_CV2 = True
    except ImportError:
        _HAS_CV2 = False

    tree = _load_structure_tree()

    ids, counts = np.unique(warped_label, return_counts=True)
    # Sort by area descending, skip background (0)
    id_count = [(int(i), int(c)) for i, c in zip(ids, counts) if i != 0]
    id_count.sort(key=lambda x: -x[1])
    id_count = id_count[:max_labels]

    props_map: dict[int, measure.RegionProperties] = {}
    for region_id in np.unique(warped_label):
        if region_id == 0:
            continue
        mask = (warped_label == region_id)
        if mask.sum() < min_area_px:
            continue
        region_lbl = measure.label(mask, connectivity=2)
        rprops = measure.regionprops(region_lbl)
        if rprops:
            biggest = max(rprops, key=lambda r: r.area)
            props_map[int(region_id)] = biggest

    out = overlay.copy()

    # Set up PIL once before the loop (avoid per-region conversion overhead)
    pil_draw = None
    pil_img = None
    if not _HAS_CV2:
        from PIL import Image, ImageDraw, ImageFont
        font_size = max(14, int(font_scale * 28))
        fnt = None
        for fname in ["arial.ttf", "Arial.ttf", "DejaVuSans.ttf",
                      "C:/Windows/Fonts/arial.ttf"]:
            try:
                fnt = ImageFont.truetype(fname, font_size)
                break
            except Exception:
                pass
        if fnt is None:
            fnt = ImageFont.load_default(size=font_size) if hasattr(ImageFont, 'load_default') else ImageFont.load_default()
        pil_img = Image.fromarray(out)
        pil_draw = ImageDraw.Draw(pil_img)

    for rid, count in id_count:
        if rid not in props_map:
            continue
        cy, cx = props_map[rid].centroid
        cy, cx = int(cy), int(cx)

        info = tree.get(str(rid), {})
        acronym = info.get("acronym", "")
        if not acronym:
            continue  # skip unknown IDs
        if len(acronym) > 8:
            acronym = acronym[:8]

        if _HAS_CV2:
            font = _cv2.FONT_HERSHEY_SIMPLEX
            lw = max(1, int(font_scale * 1.5))
            _cv2.putText(out, acronym, (cx + 1, cy + 1), font, font_scale,
                         shadow_color, lw + 1, _cv2.LINE_AA)
            _cv2.putText(out, acronym, (cx, cy), font, font_scale,
                         text_color, lw, _cv2.LINE_AA)
        else:
            pil_draw.text((cx + 1, cy + 1), acronym, fill=shadow_color, font=fnt)
            pil_draw.text((cx, cy), acronym, fill=text_color, font=fnt)

    if pil_img is not None:
        out = np.array(pil_img)
    return out


# ── Core registration ───────────────────────────────────────────────────────

def _tissue_guided_warp(
    real: np.ndarray,
    label: np.ndarray,
    atlas_res_um: float = 25.0,
    real_res_um: float = 0.65,
    fit_mode: str = "contain",
    enable_nonlinear: bool = True,
    opt_maxiter: int = 150,
) -> tuple[np.ndarray, dict]:
    """
    Tissue-guided atlas registration:
    1. Detect real tissue bounding box and centroid
    2. Determine whether to use full atlas or one hemisphere:
       - Compare tissue aspect ratio with full vs half atlas aspect ratios
       - If half-atlas fits better, try both L and R and pick by edge overlap
    3. Apply similarity transform (scale + translate) to warp atlas → real space
    """
    rh, rw = real.shape[:2]
    real_u8 = _norm_u8_robust(real)
    phys_scale = float(atlas_res_um) / float(real_res_um)

    # --- Detect tissue ---
    info = _detect_tissue(real)
    if not info["ok"]:
        out, fitted = _align_shape_physical(label, (rh, rw), atlas_res_um, real_res_um, fit_mode)
        return out.astype(np.int32), {"method": "fallback_center"}

    t_cy, t_cx = info["centroid"]
    t_h, t_w = info["hw"]
    tissue_aspect = t_h / max(t_w, 1)
    tissue_orientation = info.get("orientation", 0.0)   # radians, skimage: CCW from col-axis
    tissue_mask = info.get("mask", None)

    # --- Full atlas geometry ---
    ay0, ax0, ay1, ax1 = _atlas_bbox(label)
    a_h = float(ay1 - ay0)
    a_w = float(ax1 - ax0)
    a_cy = (ay0 + ay1) * 0.5
    a_cx = (ax0 + ax1) * 0.5
    a_mid_x = (ax0 + ax1) // 2

    full_aspect = a_h / max(a_w, 1)
    half_aspect = a_h / max(a_w / 2, 1)

    # --- Atlas orientation (from the half-atlas mask) ---
    def _atlas_orientation(lbl: np.ndarray) -> float:
        props = measure.regionprops(measure.label(lbl > 0))
        if props:
            return float(max(props, key=lambda p: p.area).orientation)
        return 0.0

    # --- Decide: full atlas or hemisphere? ---
    diff_full = abs(tissue_aspect - full_aspect)
    diff_half = abs(tissue_aspect - half_aspect)
    use_half = diff_half < diff_full  # half atlas matches tissue shape better

    if not use_half:
        # Use full atlas — initial centroid alignment
        _, total_scale, t_dx, t_dy = _similarity_warp(
            label, a_cy, a_cx, a_h, a_w,
            t_cy, t_cx, t_h, t_w,
            phys_scale, fit_mode, (rh, rw),
        )
        # Refine with edge optimization
        opt_scale, opt_angle, opt_dx, opt_dy = _optimize_warp(
            real_u8, label.astype(np.int32), total_scale, t_dx, t_dy, (rh, rw),
            maxiter=int(opt_maxiter),
        )
        warped = _apply_warp(label, opt_scale, opt_angle, opt_dx, opt_dy, (rh, rw))
        if enable_nonlinear:
            warped, nl_meta = _refine_warp_nonlinear(real_u8, warped, tissue_mask=tissue_mask)
        else:
            nl_meta = {"applied": False, "reason": "disabled"}
        return warped, {
            "method": "tissue_guided_full",
            "is_half_brain": False,
            "total_scale": float(opt_scale),
            "angle_deg": float(np.rad2deg(opt_angle)),
            "translation": [float(opt_dx), float(opt_dy)],
            "nonlinear_refine": nl_meta,
            "tissue_center": [float(t_cx), float(t_cy)],
            "tissue_hw": [int(t_h), int(t_w)],
            "tissue_mask": tissue_mask,
        }

    # --- Half atlas: try left and right, pick by edge overlap ---
    # Left half: x < midline
    left_label = label.copy().astype(np.int32)
    left_label[:, a_mid_x:] = 0
    ly0, lx0, ly1, lx1 = _atlas_bbox(left_label)
    lh, lw = float(ly1 - ly0), float(lx1 - lx0)
    lcy, lcx = (ly0 + ly1) * 0.5, (lx0 + lx1) * 0.5

    # Right half: x >= midline (mirrored)
    right_label = np.fliplr(label.copy()).astype(np.int32)
    right_label[:, a_mid_x:] = 0
    ry0, rx0, ry1, rx1 = _atlas_bbox(right_label)
    rh2, rw2 = float(ry1 - ry0), float(rx1 - rx0)
    rcy, rcx = (ry0 + ry1) * 0.5, (rx0 + rx1) * 0.5

    _, ts_l, dx_l, dy_l = _similarity_warp(
        left_label, lcy, lcx, lh, lw,
        t_cy, t_cx, t_h, t_w, phys_scale, fit_mode, (rh, rw),
    )
    _, ts_r, dx_r, dy_r = _similarity_warp(
        right_label, rcy, rcx, rh2, rw2,
        t_cy, t_cx, t_h, t_w, phys_scale, fit_mode, (rh, rw),
    )

    # Quick hemisphere selection before expensive optimization
    warped_l_q = _apply_warp(left_label, ts_l, 0.0, dx_l, dy_l, (rh, rw))
    warped_r_q = _apply_warp(right_label, ts_r, 0.0, dx_r, dy_r, (rh, rw))
    score_l = _edge_overlap_score(real_u8, warped_l_q)
    score_r = _edge_overlap_score(real_u8, warped_r_q)

    if score_l >= score_r:
        chosen_label, init_s, init_dx, init_dy, chosen = left_label, ts_l, dx_l, dy_l, "left"
        atlas_orient = _atlas_orientation(left_label)
    else:
        chosen_label, init_s, init_dx, init_dy, chosen = right_label, ts_r, dx_r, dy_r, "right_mirrored"
        atlas_orient = _atlas_orientation(right_label)

    # --- Initial rotation from orientation difference ---
    # skimage orientation: angle of major axis from positive col-axis, CCW positive
    # Rotation needed to align atlas major axis → tissue major axis
    init_angle_rad = float(tissue_orientation - atlas_orient)
    # Clamp to ±20°; beyond that, likely a flip/180° ambiguity
    if abs(init_angle_rad) > np.deg2rad(20):
        # Try both the raw angle and the ±π shifted version, keep smaller
        alt = init_angle_rad - np.pi * np.sign(init_angle_rad)
        if abs(alt) < abs(init_angle_rad):
            init_angle_rad = float(alt)
    init_angle_rad = float(np.clip(init_angle_rad, np.deg2rad(-18), np.deg2rad(18)))

    # Refine chosen hemisphere with edge optimization, starting from orientation-derived angle
    opt_scale, opt_angle, opt_dx, opt_dy = _optimize_warp(
        real_u8, chosen_label, init_s, init_dx, init_dy, (rh, rw),
        init_angle_deg=float(np.rad2deg(init_angle_rad)),
        maxiter=int(opt_maxiter),
    )
    warped = _apply_warp(chosen_label, opt_scale, opt_angle, opt_dx, opt_dy, (rh, rw))
    if enable_nonlinear:
        warped, nl_meta = _refine_warp_nonlinear(real_u8, warped, tissue_mask=tissue_mask)
    else:
        nl_meta = {"applied": False, "reason": "disabled"}

    return warped, {
        "method": "tissue_guided_half",
        "is_half_brain": True,
        "hemisphere_chosen": chosen,
        "score_left": float(score_l),
        "score_right": float(score_r),
        "total_scale": float(opt_scale),
        "angle_deg": float(np.rad2deg(opt_angle)),
        "translation": [float(opt_dx), float(opt_dy)],
        "nonlinear_refine": nl_meta,
        "tissue_center": [float(t_cx), float(t_cy)],
        "tissue_hw": [int(t_h), int(t_w)],
        "tissue_aspect": float(tissue_aspect),
        "full_atlas_aspect": float(full_aspect),
        "half_atlas_aspect": float(half_aspect),
        "tissue_mask": tissue_mask,
    }


def _align_shape_physical(label_slice: np.ndarray, target_shape: tuple[int, int],
                           atlas_res_um: float = 25.0, real_res_um: float = 0.65,
                           fit_mode: str = "contain"):
    """Fallback: physical scale + fit mode + center crop/pad."""
    scale = atlas_res_um / real_res_um
    scaled_label = rescale(label_slice.astype(np.float32), scale, order=0,
                           preserve_range=True, anti_aliasing=False).astype(np.int32)
    th, tw = target_shape
    sh, sw = scaled_label.shape
    fh, fw = th / max(sh, 1), tw / max(sw, 1)
    mode = str(fit_mode or "contain").lower()
    fit = min(fh, fw) if mode in ("contain",) else max(fh, fw) if mode == "cover" \
        else fw if mode == "width-lock" else fh if mode == "height-lock" else min(fh, fw)
    if abs(float(fit) - 1.0) > 1e-6:
        scaled_label = rescale(scaled_label.astype(np.float32), float(fit), order=0,
                               preserve_range=True, anti_aliasing=False).astype(np.int32)
    fitted_shape = (int(scaled_label.shape[0]), int(scaled_label.shape[1]))
    sh, sw = scaled_label.shape
    out = np.zeros((th, tw), dtype=scaled_label.dtype)
    min_h, min_w = min(th, sh), min(tw, sw)
    oy, ox = (th - min_h) // 2, (tw - min_w) // 2
    iy, ix = (sh - min_h) // 2, (sw - min_w) // 2
    out[oy:oy + min_h, ox:ox + min_w] = scaled_label[iy:iy + min_h, ix:ix + min_w]
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
    return (max(0, int(x0) - pad), max(0, int(y0) - pad),
            int(min(w, x1 + pad) - max(0, x0 - pad)),
            int(min(h, y1 + pad) - max(0, y0 - pad)))


def _paint_boundaries(canvas: np.ndarray, mask: np.ndarray, color) -> None:
    """Paint region boundaries on canvas in-place using find_boundaries."""
    bd = find_boundaries(mask, mode="outer", connectivity=2)
    canvas[bd] = color


def draw_contours(label_img: np.ndarray, color=(255, 255, 255)) -> np.ndarray:
    h, w = label_img.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    _paint_boundaries(canvas, label_img, color)
    return canvas


def draw_contours_major(
    label_img: np.ndarray,
    top_k: int = 12,
    tissue_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Outer brain boundary (cyan) + major region boundaries (white).

    If tissue_mask is provided it is used for the outer boundary, which
    guarantees the cyan line perfectly traces the real tissue edge.
    """
    h, w = label_img.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    # Outer boundary: prefer real tissue mask so it perfectly follows tissue edge
    outer = tissue_mask if tissue_mask is not None else (label_img > 0).astype(np.uint8)
    _paint_boundaries(canvas, outer, np.array([0, 255, 255], dtype=np.uint8))
    # Major region boundaries (from warped atlas)
    ids, counts = np.unique(label_img[label_img > 0], return_counts=True)
    if len(ids) > 0:
        for rid in ids[np.argsort(counts)[::-1][:max(1, int(top_k))]]:
            _paint_boundaries(canvas, (label_img == rid).astype(np.uint8),
                              np.array([255, 255, 255], dtype=np.uint8))
    return canvas


# ── Public entry point ──────────────────────────────────────────────────────

def render_overlay(
    real_slice_path: Path,
    label_slice_path: Path,
    out_png: Path,
    alpha: float = 0.45,
    mode: str = "fill",
    structure_csv: Path | None = None,
    min_mean_threshold: float = 8.0,
    pixel_size_um: float = 0.65,
    rotate_deg: float = 0.0,
    flip_mode: str = "none",
    return_meta: bool = False,
    major_top_k: int = 20,
    fit_mode: str = "cover",
    warped_label_out: Path | None = None,
    real_z_index: int | None = None,
    label_z_index: int | None = None,
):
    real_raw = imread(str(real_slice_path))
    label_raw = imread(str(label_slice_path))
    real, real_slice_meta = select_real_slice_2d(
        real_raw,
        z_index=real_z_index,
        source_path=real_slice_path,
    )
    label, label_slice_meta = select_label_slice_2d(
        label_raw,
        z_index=label_z_index,
    )

    label_shape_before = tuple(int(x) for x in label.shape)
    real_shape = tuple(int(x) for x in real.shape)

    # User pre-transform: rotate / flip atlas
    if rotate_deg != 0.0:
        label = rotate(label.astype(np.float32), rotate_deg, order=0,
                       preserve_range=True, resize=True).astype(np.int32)
    if flip_mode == "horizontal":
        label = np.fliplr(label)
    elif flip_mode == "vertical":
        label = np.flipud(label)

    # ── Tissue-guided registration ─────────────────────────────────────────
    label, warp_meta = _tissue_guided_warp(
        real, label,
        atlas_res_um=25.0,
        real_res_um=pixel_size_um,
        fit_mode=fit_mode,
    )

    # ── Clip atlas to actual tissue footprint ──────────────────────────────
    # Reuse tissue mask from registration (avoids second _detect_tissue call).
    try:
        tight = warp_meta.get("tissue_mask")
        if tight is not None:
            # 3px dilation: enough to keep atlas at tissue boundary, won't let it far outside
            tissue_clip = morphology.dilation(tight.astype(np.uint8), morphology.disk(3)).astype(bool)
            label = np.where(tissue_clip, label, 0).astype(np.int32)
    except Exception:
        pass
    # ──────────────────────────────────────────────────────────────────────

    label_shape_after = tuple(int(x) for x in label.shape)
    roi_bbox = _roi_bbox_from_real(real)
    rx, ry, rw2, rh2 = roi_bbox
    cx_full = float(rx + rw2 / 2.0)
    cy_full = float(ry + rh2 / 2.0)

    real_u8 = _norm_u8_robust(real)

    # ── Coloring: Allen official colors (hex) → prefer over CSV / LUT ──────
    tree = _load_structure_tree()
    colored = None

    if tree:
        colored = np.zeros((*label.shape, 3), dtype=np.uint8)
        for rid in np.unique(label.astype(np.int32)):
            if rid == 0:
                continue
            info = tree.get(str(rid))
            if info and info.get("color") and len(info["color"]) == 6:
                hx = info["color"]
                r = int(hx[0:2], 16); g = int(hx[2:4], 16); b = int(hx[4:6], 16)
                colored[label == rid] = (r, g, b)
            else:
                # Fallback deterministic color
                rng = (int(rid) * 2654435761) & 0xFFFFFF
                colored[label == rid] = ((rng >> 16) & 0xFF, (rng >> 8) & 0xFF, rng & 0xFF)
    elif structure_csv is not None:
        cmap = load_allen_color_map(structure_csv)
        if cmap:
            colored = np.zeros((*label.shape, 3), dtype=np.uint8)
            for rid in np.unique(label.astype(np.int32)):
                if rid == 0:
                    continue
                color = cmap.get(int(rid))
                if color is not None:
                    colored[label == rid] = np.array(color, dtype=np.uint8)

    if colored is None:
        lut = np.array([
            [20, 20, 20], [0, 200, 255], [0, 255, 120], [255, 120, 180],
            [255, 70, 70], [220, 220, 80], [160, 120, 255], [255, 180, 70],
        ], dtype=np.uint8)
        colored = lut[(label.astype(np.int32) % len(lut))]

    # Retrieve tissue mask for outer boundary (cached from registration)
    _tissue_mask_for_contour = warp_meta.get("tissue_mask")

    if mode == "contour":
        colored = draw_contours(label)
    elif mode == "contour-major":
        colored = draw_contours_major(label, top_k=major_top_k,
                                      tissue_mask=_tissue_mask_for_contour)

    overlay = alpha_blend(real_u8, colored, alpha)

    # Quality guard (lower threshold for contour modes which are mostly black)
    effective_thr = min_mean_threshold if mode == "fill" else min(float(min_mean_threshold) * 0.25, 2.0)
    if float(np.mean(overlay)) < effective_thr:
        raise ValueError(
            f"overlay quality check failed: near-black output "
            f"(mean={float(np.mean(overlay)):.2f}, threshold={effective_thr:.2f})"
        )

    # ── Region labels ────────────────────────────────────────────────────────
    if mode in ("fill",):
        overlay = draw_region_labels(overlay, label)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    imwrite(str(out_png), overlay)

    if warped_label_out is not None:
        warped_label_out.parent.mkdir(parents=True, exist_ok=True)
        imwrite(str(warped_label_out), label.astype(np.int32))

    # Strip non-serializable internals from warp_meta before returning
    warp_meta.pop("tissue_mask", None)

    if return_meta:
        return out_png, {
            "real_shape": real_shape,
            "label_shape_before": label_shape_before,
            "label_shape_after": label_shape_after,
            "real_slice": real_slice_meta,
            "label_slice": label_slice_meta,
            "scale": float(25.0 / float(pixel_size_um)),
            "pixelSizeUm": float(pixel_size_um),
            "rotateAtlas": float(rotate_deg),
            "flipAtlas": str(flip_mode),
            "fitMode": str(fit_mode),
            "real_aspect": float(real_shape[1] / max(real_shape[0], 1)),
            "atlas_aspect_before": float(label_shape_before[1] / max(label_shape_before[0], 1)),
            "atlas_aspect": float(label_shape_after[1] / max(label_shape_after[0], 1)),
            "roi_bbox": [int(roi_bbox[0]), int(roi_bbox[1]), int(roi_bbox[2]), int(roi_bbox[3])],
            "roi_center_full": [cx_full, cy_full],
            "roi_roundtrip_error": 0.0,
            "warped_label_path": str(warped_label_out) if warped_label_out is not None else "",
            "warp": warp_meta,
        }

    return out_png
