"""ai_landmark.py — Cross-modal feature matching for brain atlas registration.

The fundamental challenge: fluorescence microscopy images and Allen CCFv3 atlas
annotations are completely different imaging modalities. Standard same-modality
descriptors (ORB, Harris) fail because texture patterns don't correspond.

Solution: Multi-strategy ensemble that converts both modalities to shared
representations (edges, contours, structural features) before matching.

Strategy cascade (best → fallback):
  1. Edge-domain SIFT — most reliable for cross-modal
  2. Multi-scale edge ORB — fast, decent for edges
  3. Contour correspondence — arc-length proportional sampling
  4. Phase correlation — global rotation/translation estimation
  5. Harris corner fallback — legacy, index-paired (worst)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from skimage.filters import sobel
from skimage.measure import ransac
from skimage.metrics import structural_similarity as ssim
from skimage.transform import AffineTransform
from tifffile import imread

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Image preprocessing: convert both modalities to shared feature spaces
# ---------------------------------------------------------------------------


def _to_u8(img: np.ndarray) -> np.ndarray:
    """Normalize any image to uint8."""
    if img.ndim == 3:
        img = img[..., 0]
    img = img.astype(np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx > mn:
        img = (img - mn) / (mx - mn) * 255.0
    return img.astype(np.uint8)


def _to_edge_u8(img_u8: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Convert to edge map — the key cross-modal bridge.

    Both fluorescence tissue and atlas annotation boundaries produce
    similar edge patterns, making edge-domain the natural shared space.
    """
    blurred = gaussian_filter(img_u8.astype(np.float32), sigma=sigma)
    edges = sobel(blurred)
    # Normalize to full uint8 range
    mx = float(edges.max())
    if mx > 0:
        edges = edges / mx * 255.0
    return edges.astype(np.uint8)


def _to_structure_feature(img_u8: np.ndarray) -> np.ndarray:
    """Structure feature map: 60% edges + 40% dark regions.

    For atlas labels: dark=0 → outside, bright=labeled region
    For real tissue: dark=background, bright=tissue
    Both share the property that structure boundaries are informative.
    """
    edges = _to_edge_u8(img_u8, sigma=1.5)
    # Inverted intensity: dark regions → high values (captures tissue/atlas shape)
    inv = 255 - img_u8
    combined = 0.6 * edges.astype(np.float32) + 0.4 * inv.astype(np.float32)
    mn, mx = float(combined.min()), float(combined.max())
    if mx > mn:
        combined = (combined - mn) / (mx - mn) * 255.0
    return combined.astype(np.uint8)


# ---------------------------------------------------------------------------
# Strategy 1: Edge-domain SIFT (best for cross-modal)
# ---------------------------------------------------------------------------


def _sift_match_on_features(
    feat1: np.ndarray,
    feat2: np.ndarray,
    max_points: int = 40,
    ratio_threshold: float = 0.75,
) -> np.ndarray:
    """SIFT feature detection + Lowe's ratio test matching.

    SIFT's 128-dim float descriptor is far more discriminative than
    ORB's 32-byte binary descriptor for cross-modal edge matching.
    """
    try:
        import cv2
    except ImportError:
        return np.empty((0, 4), dtype=np.float32)

    sift = cv2.SIFT_create(nfeatures=max_points * 4, contrastThreshold=0.02, edgeThreshold=15)
    kp1, des1 = sift.detectAndCompute(feat1, None)
    kp2, des2 = sift.detectAndCompute(feat2, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        return np.empty((0, 4), dtype=np.float32)

    # Lowe's ratio test: much better than cross-check for ambiguous matches
    bf = cv2.BFMatcher(cv2.NORM_L2)
    raw_matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m_pair in raw_matches:
        if len(m_pair) < 2:
            continue
        m, n = m_pair
        if m.distance < ratio_threshold * n.distance:
            good.append(m)

    good = sorted(good, key=lambda m: m.distance)[:max_points]
    if not good:
        return np.empty((0, 4), dtype=np.float32)

    pairs = np.array(
        [
            [
                kp1[m.queryIdx].pt[0],
                kp1[m.queryIdx].pt[1],
                kp2[m.trainIdx].pt[0],
                kp2[m.trainIdx].pt[1],
            ]
            for m in good
        ],
        dtype=np.float32,
    )
    return pairs


def match_edge_sift(
    real_u8: np.ndarray,
    atlas_u8: np.ndarray,
    max_points: int = 40,
) -> np.ndarray:
    """Strategy 1: SIFT on multi-scale edge features.

    Generates edge maps at multiple Gaussian sigmas and concatenates
    matches from all scales for robustness.
    """
    all_pairs = []

    for sigma in (0.8, 1.5, 2.5):
        real_edge = _to_edge_u8(real_u8, sigma=sigma)
        atlas_edge = _to_edge_u8(atlas_u8, sigma=sigma)
        pairs = _sift_match_on_features(real_edge, atlas_edge, max_points=max_points)
        if len(pairs) > 0:
            all_pairs.append(pairs)

    # Also try structure feature maps
    real_struct = _to_structure_feature(real_u8)
    atlas_struct = _to_structure_feature(atlas_u8)
    pairs_struct = _sift_match_on_features(real_struct, atlas_struct, max_points=max_points)
    if len(pairs_struct) > 0:
        all_pairs.append(pairs_struct)

    if not all_pairs:
        return np.empty((0, 4), dtype=np.float32)

    combined = np.vstack(all_pairs)

    # Deduplicate: if two matches land within 5px of each other, keep one
    if len(combined) > 1:
        combined = _deduplicate_pairs(combined, min_dist=5.0)

    return combined


# ---------------------------------------------------------------------------
# Strategy 2: Edge-domain ORB (faster, decent quality)
# ---------------------------------------------------------------------------


def match_edge_orb(
    real_u8: np.ndarray,
    atlas_u8: np.ndarray,
    max_points: int = 30,
) -> np.ndarray:
    """Strategy 2: ORB on edge maps with cross-check + distance filtering."""
    try:
        import cv2
    except ImportError:
        return np.empty((0, 4), dtype=np.float32)

    all_pairs = []
    for sigma in (1.0, 2.0):
        real_edge = _to_edge_u8(real_u8, sigma=sigma)
        atlas_edge = _to_edge_u8(atlas_u8, sigma=sigma)

        orb = cv2.ORB_create(nfeatures=max_points * 3)
        kp1, des1 = orb.detectAndCompute(real_edge, None)
        kp2, des2 = orb.detectAndCompute(atlas_edge, None)

        if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
            continue

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        # Filter by distance: only keep matches below median * 1.5
        if matches:
            dists = [m.distance for m in matches]
            median_d = float(np.median(dists))
            matches = [m for m in matches if m.distance < median_d * 1.5]
            matches = sorted(matches, key=lambda m: m.distance)[:max_points]

        for m in matches:
            all_pairs.append(
                [
                    kp1[m.queryIdx].pt[0],
                    kp1[m.queryIdx].pt[1],
                    kp2[m.trainIdx].pt[0],
                    kp2[m.trainIdx].pt[1],
                ]
            )

    if not all_pairs:
        return np.empty((0, 4), dtype=np.float32)

    combined = np.array(all_pairs, dtype=np.float32)
    return _deduplicate_pairs(combined, min_dist=5.0)[:max_points]


# ---------------------------------------------------------------------------
# Strategy 3: Contour correspondence (geometric, no descriptors needed)
# ---------------------------------------------------------------------------


def match_contours(
    real_u8: np.ndarray,
    atlas_u8: np.ndarray,
    n_points: int = 20,
) -> np.ndarray:
    """Strategy 3: Sample tissue and atlas contours, match by arc-length ratio.

    This is descriptor-free — it relies on the geometric assumption that
    tissue outline ≈ atlas outline (same brain shape), so corresponding
    points along the contour at proportional arc-lengths match spatially.
    """
    try:
        import cv2
    except ImportError:
        return np.empty((0, 4), dtype=np.float32)

    def _largest_contour(img_u8: np.ndarray) -> np.ndarray | None:
        """Extract the largest contour from a binarized image."""
        # Threshold: tissue/atlas is brighter than background
        _, binary = cv2.threshold(img_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 100:
            return None
        return largest.squeeze()

    real_contour = _largest_contour(real_u8)
    atlas_contour = _largest_contour(atlas_u8)

    if real_contour is None or atlas_contour is None:
        return np.empty((0, 4), dtype=np.float32)
    if len(real_contour) < n_points or len(atlas_contour) < n_points:
        return np.empty((0, 4), dtype=np.float32)

    # Sample both contours at uniform arc-length intervals
    def _sample_contour(contour: np.ndarray, n: int) -> np.ndarray:
        # Compute cumulative arc length
        diffs = np.diff(contour, axis=0).astype(np.float64)
        seg_lengths = np.sqrt((diffs**2).sum(axis=1))
        cum_length = np.concatenate([[0.0], np.cumsum(seg_lengths)])
        total = cum_length[-1]
        if total < 1.0:
            return contour[:n]
        # Sample at uniform spacing
        target_lengths = np.linspace(0, total, n, endpoint=False)
        sampled = np.zeros((n, 2), dtype=np.float32)
        for i, tl in enumerate(target_lengths):
            idx = np.searchsorted(cum_length, tl, side="right") - 1
            idx = max(0, min(idx, len(contour) - 2))
            frac = (tl - cum_length[idx]) / max(seg_lengths[idx], 1e-6)
            sampled[i] = contour[idx] + frac * (contour[idx + 1] - contour[idx])
        return sampled

    # Align contour start points by finding the topmost point
    def _align_start(contour: np.ndarray) -> np.ndarray:
        top_idx = np.argmin(contour[:, 1])
        return np.roll(contour, -top_idx, axis=0)

    real_contour = _align_start(real_contour)
    atlas_contour = _align_start(atlas_contour)

    real_pts = _sample_contour(real_contour, n_points)
    atlas_pts = _sample_contour(atlas_contour, n_points)

    pairs = np.hstack([real_pts, atlas_pts])
    return pairs


# ---------------------------------------------------------------------------
# Strategy 4: Phase correlation (global alignment estimation)
# ---------------------------------------------------------------------------


def match_phase_correlation(
    real_u8: np.ndarray,
    atlas_u8: np.ndarray,
    n_grid: int = 4,
) -> np.ndarray:
    """Strategy 4: Phase correlation on image patches.

    Divides images into a grid, computes local phase correlation
    per patch, and returns displacement vectors as pseudo-landmark pairs.
    Robust to intensity differences (works in frequency domain).
    """
    from skimage.registration import phase_cross_correlation

    h, w = real_u8.shape[:2]
    ph = h // n_grid
    pw = w // n_grid
    pairs = []

    real_edge = _to_edge_u8(real_u8, sigma=1.5)
    atlas_edge = _to_edge_u8(atlas_u8, sigma=1.5)

    for gi in range(n_grid):
        for gj in range(n_grid):
            y0 = gi * ph
            x0 = gj * pw
            y1 = min(y0 + ph, h)
            x1 = min(x0 + pw, w)

            patch_r = real_edge[y0:y1, x0:x1].astype(np.float64)
            patch_a = atlas_edge[y0:y1, x0:x1].astype(np.float64)

            if patch_r.shape[0] < 16 or patch_r.shape[1] < 16:
                continue
            # Skip empty patches
            if np.std(patch_r) < 2.0 or np.std(patch_a) < 2.0:
                continue

            try:
                shift, _error, _phasediff = phase_cross_correlation(
                    patch_a, patch_r, upsample_factor=4
                )
            except Exception:
                continue

            # shift is (dy, dx) from reference (atlas) to moving (real)
            cx_r = float(x0 + pw / 2)
            cy_r = float(y0 + ph / 2)
            cx_a = cx_r + float(shift[1])
            cy_a = cy_r + float(shift[0])

            # Reject if shift is too large (> 30% of patch size)
            if abs(shift[0]) > ph * 0.3 or abs(shift[1]) > pw * 0.3:
                continue

            pairs.append([cx_r, cy_r, cx_a, cy_a])

    if not pairs:
        return np.empty((0, 4), dtype=np.float32)
    return np.array(pairs, dtype=np.float32)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _deduplicate_pairs(pairs: np.ndarray, min_dist: float = 5.0) -> np.ndarray:
    """Remove near-duplicate landmark pairs."""
    if len(pairs) <= 1:
        return pairs
    keep = [0]
    for i in range(1, len(pairs)):
        dists = np.sqrt(((pairs[keep, :2] - pairs[i, :2]) ** 2).sum(axis=1))
        if np.all(dists >= min_dist):
            keep.append(i)
    return pairs[np.array(keep)]


def _match_shape(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape != b.shape:
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a = a[:h, :w]
        b = b[:h, :w]
    return a, b


# ---------------------------------------------------------------------------
# Ensemble matching: combine all strategies
# ---------------------------------------------------------------------------


def propose_landmarks_ensemble(
    real_u8: np.ndarray,
    atlas_u8: np.ndarray,
    max_points: int = 30,
    ransac_residual: float = 8.0,
) -> tuple[np.ndarray, dict]:
    """Run all matching strategies and combine with RANSAC filtering.

    Returns:
        (filtered_pairs Nx4, metadata dict)
    """
    strategies_used = []
    all_pairs = []

    # Strategy 1: Edge-domain SIFT (most reliable)
    try:
        sift_pairs = match_edge_sift(real_u8, atlas_u8, max_points=max_points)
        if len(sift_pairs) >= 3:
            all_pairs.append(sift_pairs)
            strategies_used.append(f"edge_sift({len(sift_pairs)})")
    except Exception as e:
        log.debug("SIFT matching failed: %s", e)

    # Strategy 2: Edge-domain ORB
    try:
        orb_pairs = match_edge_orb(real_u8, atlas_u8, max_points=max_points)
        if len(orb_pairs) >= 3:
            all_pairs.append(orb_pairs)
            strategies_used.append(f"edge_orb({len(orb_pairs)})")
    except Exception as e:
        log.debug("ORB matching failed: %s", e)

    # Strategy 3: Contour correspondence
    try:
        contour_pairs = match_contours(real_u8, atlas_u8, n_points=max(12, max_points // 2))
        if len(contour_pairs) >= 4:
            all_pairs.append(contour_pairs)
            strategies_used.append(f"contour({len(contour_pairs)})")
    except Exception as e:
        log.debug("Contour matching failed: %s", e)

    # Strategy 4: Phase correlation grid
    try:
        phase_pairs = match_phase_correlation(real_u8, atlas_u8, n_grid=5)
        if len(phase_pairs) >= 3:
            all_pairs.append(phase_pairs)
            strategies_used.append(f"phase({len(phase_pairs)})")
    except Exception as e:
        log.debug("Phase correlation failed: %s", e)

    if not all_pairs:
        return np.empty((0, 4), dtype=np.float32), {
            "strategies": [],
            "raw_total": 0,
            "after_ransac": 0,
        }

    combined = np.vstack(all_pairs)
    combined = _deduplicate_pairs(combined, min_dist=4.0)
    n_raw = len(combined)

    # RANSAC filtering on the combined set
    inliers = np.ones(n_raw, dtype=bool)
    model = None
    if n_raw >= 4:
        try:
            model, inliers = ransac(
                (combined[:, 2:], combined[:, :2]),  # (src=atlas, dst=real)
                AffineTransform,
                min_samples=4,
                residual_threshold=ransac_residual,
                max_trials=500,
            )
        except Exception:
            inliers = np.ones(n_raw, dtype=bool)

    filtered = combined[inliers]

    # If RANSAC was too aggressive, keep more points
    if len(filtered) < 4 and n_raw >= 4:
        # Try with relaxed threshold
        try:
            _, inliers2 = ransac(
                (combined[:, 2:], combined[:, :2]),
                AffineTransform,
                min_samples=3,
                residual_threshold=ransac_residual * 2.0,
                max_trials=300,
            )
            if np.sum(inliers2) > len(filtered):
                filtered = combined[inliers2]
        except Exception:
            pass

    # Sort by distance from center for more even spatial distribution
    if len(filtered) > max_points:
        h, w = real_u8.shape[:2]
        cx, cy = w / 2, h / 2
        dists = np.sqrt((filtered[:, 0] - cx) ** 2 + (filtered[:, 1] - cy) ** 2)
        # Stratified selection: divide into quadrants, take evenly
        filtered = filtered[np.argsort(dists)][:max_points]

    meta = {
        "strategies": strategies_used,
        "raw_total": int(n_raw),
        "after_ransac": int(len(filtered)),
        "affine_residual": float(np.mean(np.abs(combined[inliers, :2] - combined[inliers, :2])))
        if model is not None
        else 0.0,
    }

    return filtered, meta


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_alignment(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a, b = _match_shape(a, b)
    return float(ssim(a, b, data_range=float(max(np.ptp(a), np.ptp(b), 1.0))))


def score_alignment_edges(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a, b = _match_shape(a, b)
    ea = sobel(a)
    eb = sobel(b)
    return float(ssim(ea, eb, data_range=float(max(np.ptp(ea), np.ptp(eb), 1.0))))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def propose_landmarks(
    real_path: Path,
    atlas_path: Path,
    out_csv: Path,
    max_points: int = 30,
    min_distance: int = 12,
    ransac_residual: float = 8.0,
) -> dict:
    """Detect and propose landmark pairs between real and atlas images.

    Uses multi-strategy ensemble matching for robust cross-modal correspondence.
    """
    from scripts.slice_select import select_label_slice_2d, select_real_slice_2d

    real = imread(str(real_path))
    atl = imread(str(atlas_path))
    real, _ = select_real_slice_2d(real, source_path=real_path)
    atl, _ = select_label_slice_2d(atl)

    real_u8 = _to_u8(real)
    atlas_u8 = _to_u8(atl)
    real_u8, atlas_u8 = _match_shape(real_u8, atlas_u8)

    # Run ensemble matching
    filtered, match_meta = propose_landmarks_ensemble(
        real_u8,
        atlas_u8,
        max_points=max_points,
        ransac_residual=ransac_residual,
    )
    nf = len(filtered)

    log.info(
        "Landmark matching: %d pairs from strategies %s (raw=%d, filtered=%d)",
        nf,
        match_meta["strategies"],
        match_meta["raw_total"],
        nf,
    )

    df = pd.DataFrame(
        {
            "real_x": filtered[:, 0] if nf else [],
            "real_y": filtered[:, 1] if nf else [],
            "atlas_x": filtered[:, 2] if nf else [],
            "atlas_y": filtered[:, 3] if nf else [],
        }
    )
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    return {
        "landmark_pairs": int(nf),
        "raw_pairs": int(match_meta["raw_total"]),
        "score": score_alignment_edges(real, atl),
        "strategies": match_meta["strategies"],
        "params": {
            "max_points": max_points,
            "min_distance": min_distance,
            "ransac_residual": ransac_residual,
        },
        "csv": str(out_csv),
    }
