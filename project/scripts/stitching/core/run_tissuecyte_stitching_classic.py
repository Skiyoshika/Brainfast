# ruff: noqa
"""Vendored from UCI-XuLab-RegTools (github.com/UCI-XuLab/UCI-XuLab-RegTools).

Original authors: Atchuth Naveen, Andy Thai — UC Irvine.
Vendored into Brainfast on 2026-04-23 as
project/scripts/stitching/core/run_tissuecyte_stitching_classic.py.
Only change vs upstream: the ``regtools.utils.logging_utils.TeeLogger``
import (which pulls in the full Xu Lab utils tree) is replaced by an
inline minimal equivalent so this file stands alone.

--- Original header ---
#!/usr/bin/env python
# -*- coding: utf-8 -*-
Code by Atchuth Naveen
Updated and maintained by Andy Thai
Code developed at UC Irvine.

Corrects deformation in individual images and stitches them into a large mosaic section.
"""

# Standard library imports
import sys
import argparse
import os
import gc
import time
import sys
import logging
logging.getLogger().setLevel(logging.INFO)
logging.captureWarnings(True)

# Enable ANSI escape sequences on Windows
import colorama
from colorama import Fore, Style
colorama.init()

# Color shortcuts for pretty printing (matches registration pipeline)
C_HEADER = Fore.CYAN + Style.BRIGHT      # Section headers
C_STEP = Fore.BLUE + Style.BRIGHT        # Step indicators
C_INFO = Fore.WHITE                       # General info
C_PATH = Fore.YELLOW                      # File paths
C_VALUE = Fore.MAGENTA                    # Numeric values
C_SUCCESS = Fore.GREEN + Style.BRIGHT    # Success messages
C_WARN = Fore.YELLOW + Style.BRIGHT      # Warnings
C_ERROR = Fore.RED + Style.BRIGHT        # Errors
C_RESET = Style.RESET_ALL                # Reset to default

# ANSI escape code stripping regex (for log file output)
import re as _re_ansi
_ANSI_ESCAPE = _re_ansi.compile(r'\x1b\[[0-9;]*m')

# Ensure project root is on sys.path when run as a standalone script.
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)


# Inline minimal TeeLogger (replaces the one from regtools.utils.logging_utils so
# this file does not depend on the wider Xu Lab utils tree). Semantics match
# upstream: dual stdout + file, strips ANSI for the file, restores on close().
class _Tee:
    _TEE_ANSI_RE = _re_ansi.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def __init__(self, log_path):
        self.terminal = sys.stdout
        self._original_stderr = sys.stderr
        self.log_file = open(log_path, 'w', encoding='utf-8')

    def write(self, message):
        try:
            self.terminal.write(message)
        except UnicodeEncodeError:
            self.terminal.write(message.encode('ascii', errors='replace').decode('ascii'))
        clean_message = self._TEE_ANSI_RE.sub('', message)
        self.log_file.write(clean_message)
        self.log_file.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        sys.stdout = self.terminal
        sys.stderr = self._original_stderr
        self.log_file.close()

# Third party imports
import glob
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm

# Third party imports - image and array processing
import cv2
import numpy as np

from scipy.special import binom
from concurrent.futures import ThreadPoolExecutor

# Custom imports
from .stitcher import Stitcher
from .tile import Tile


def bernstein(u, n: int, k: int) -> float:
    """Bernstein polynomial for deformation mapping.

    Args:
        u (_type_): Input value
        n (int): Top binomial coefficient
        k (int): Bottom binomial coefficient

    Returns:
        float: Bernstein polynomial output
    """
    return binom(n, k) * u**k * (1 - u)**(n - k)


def create_perfect_grid(nhs: int, nvs: int, lw: float, sw: float) -> np.ndarray:
    """Generates a perfect grid image.

    Args:
        nhs (int): number of horizontal squares
        nvs (int): number of vertical squares
        lw (float): line width
        sw (float): square width

    Returns:
        np.ndarray: Grid image
    """
    xs = 20
    ys = 20
    
    im = np.zeros((2 * xs + nvs * sw + lw, 2 * ys + nhs * sw + lw))
    # Generate horizontal lines
    for i in range(nhs + 1):
        cv2.line(im, (xs + i * sw, ys), (xs + i * sw, im.shape[0] - ys - int(lw / 2)), (255, 255, 255), thickness=lw)
    # Generate vertical lines
    for i in range(nvs + 1):
        cv2.line(im, (xs, ys + i * sw), (im.shape[1] - xs - int(lw / 2), ys + i * sw), (255, 255, 255), thickness=lw) 
    return im


def get_deformation_map(width: int, height: int, kx, ky) -> tuple:
    """Retrieves deformation map for image correction.

    Uses a separable Bernstein basis decomposition: the 2-D Bezier
    surface B(u,v) = sum_ij k_ij * B_i(u) * B_j(v) factors into two
    small 1-D basis matrices (one per axis).  This replaces the dense
    (2.3M x 25) Bernstein matrix (~468 MB) with two tiny matrices
    (~60 KB each), reducing memory usage by ~450 MB and speeding up
    the computation by 10-50x.

    Args:
        width (int): Width of map (rows of the cropped tile)
        height (int): Height of map (columns of the cropped tile)
        kx: Bezier patch x-parameters (25-element vector)
        ky: Bezier patch y-parameters (25-element vector)

    Returns:
        tuple: (map_x, map_y) — float32 2D coordinate maps for cv2.remap,
               shape (2*width, 2*height) each.
    """
    n, m = 4, 4
    n_rows = 2 * width    # corresponds to the 'i' / row axis
    n_cols = 2 * height   # corresponds to the 'j' / column axis

    # 1-D Bernstein basis vectors — tiny matrices
    u_vals = np.arange(n_cols, dtype=np.float64) / n_cols   # column param
    v_vals = np.arange(n_rows, dtype=np.float64) / n_rows   # row param

    Bu = np.column_stack([bernstein(u_vals, n, i) for i in range(n + 1)])  # (n_cols, 5)
    Bv = np.column_stack([bernstein(v_vals, m, j) for j in range(m + 1)])  # (n_rows, 5)

    kx_2d = np.asarray(kx).reshape(n + 1, m + 1)
    ky_2d = np.asarray(ky).reshape(n + 1, m + 1)

    # Separable outer product: (n_rows, 5) @ (5,5) @ (5, n_cols) = (n_rows, n_cols)
    # Note: kx_2d is transposed because barray indexed as kx[i*(m+1)+j],
    # so kx_2d[i,j] but the separable form Bv @ M @ Bu.T contracts as M[j,i].
    map_x = (Bv @ kx_2d.T @ Bu.T) * height
    map_y = (Bv @ ky_2d.T @ Bu.T) * width

    np.clip(map_x, 0, height - 1, out=map_x)
    np.clip(map_y, 0, width - 1, out=map_y)

    return map_x.astype(np.float32), map_y.astype(np.float32)


def correct_deformation(im0: np.ndarray, H, map_x, map_y) -> np.ndarray:
    """Apply homography + Bezier deformation correction to a tile.

    Uses cv2.remap() for the Bezier deformation — a single optimised C++
    call with SIMD that replaces the previous hand-coded bilinear
    interpolation (which allocated ~80 MB of temporaries per tile).

    Args:
        im0 (np.ndarray): Image array
        H: Homography matrix
        map_x: Bezier deformation column-coordinate map (float32, 2D)
        map_y: Bezier deformation row-coordinate map (float32, 2D)

    Returns:
        np.ndarray: Deformation corrected image (float32)
    """
    im_warp = cv2.warpPerspective(im0, H, (im0.shape[1], im0.shape[0]))
    im_warp = im_warp[20:794, 20:776]
    h, w = im_warp.shape

    # cv2.remap does bilinear interpolation in optimised C++ with SIMD.
    # map_x provides the source column for each destination pixel,
    # map_y provides the source row.  Output shape matches map shape.
    upsampled = cv2.remap(im_warp, map_x, map_y,
                          interpolation=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)
    del im_warp

    return cv2.resize(np.float32(upsampled), (w, h),
                      interpolation=cv2.INTER_AREA)


def get_missing_tile_paths(missing_tiles) -> list:
    """_summary_

    Args:
        missing_tiles (_type_): _description_

    Returns:
        list: Missing tile paths
    """
    paths = []

    for index, path in missing_tiles.items():
        spath = ','.join(map(str, path))
        logging.info('Writing missing tile path for tile {0} as {1}'.format(index, spath))
        paths.append(spath)

    return paths


def read_image(file_name: str) -> np.ndarray:
    """Reads image from filepath string.

    Uses cv2 instead of SimpleITK for lower overhead on standard TIFF tiles.

    Args:
        file_name (str): Input file path

    Returns:
        np.ndarray: Image array
    """
    im = cv2.imread(str(file_name), cv2.IMREAD_UNCHANGED)
    if im is None:
        raise IOError(f"Could not read image: {file_name}")
    return im


def write_output(imgarr: np.ndarray, path: str):
    """Writes image array to file.

    Uses cv2 instead of SimpleITK for lower overhead.  Clips to uint16 in-place
    when possible to avoid a full-size copy.

    Args:
        imgarr (np.ndarray): Image array
        path (str): Output file path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    out = np.ascontiguousarray(imgarr)
    np.clip(out, 0, 65535, out=out)
    if out.dtype != np.uint16:
        out = out.astype(np.uint16)
    cv2.imwrite(path, out)


def normalize_image_by_median(image: np.ndarray,
                              dark_level: float = 0.0) -> np.ndarray:
    """Creates a multiplicative gain map from an average-tile image.

    The gain map is ``(median - dark) / (image - dark)``, designed so that
    the flat-field correction ``(raw - dark) * gain + dark`` perfectly
    cancels the vignetting for any true signal level.

    Without the dark subtraction (i.e. ``median / image``), the camera
    baseline (~15 counts) biases the ratio at positions where the
    vignetting is strong: a pixel with avg=112 and dark=15 has true
    attenuation (112-15)/(204-15) = 0.513, but ``204/112 = 1.82``
    instead of the correct ``189/97 = 1.95``.  The 7 % error manifests
    as systematic 7-11 % intensity steps at every tile boundary.

    Invalid results (division by zero, near-zero avg) are clamped to 1.0
    (identity) rather than 0, so they pass through unchanged instead of
    destroying pixel data.  The gain is also clamped to [0.1, 3.0] to
    prevent extreme corrections from noisy low-count pixels.

    Args:
        image (np.ndarray): Average tile (float, all values >= 0).
        dark_level (float): Camera dark-current baseline to subtract
            before computing the gain ratio.  Default 0 preserves
            backward-compatible behaviour.

    Returns:
        np.ndarray: Gain map (same shape, float64).
    """
    median = np.median(image)

    if median > dark_level:
        with np.errstate(divide='ignore', invalid='ignore'):
            gain = np.divide(median - dark_level,
                             image - dark_level)
        gain[np.isnan(gain)] = 1.0
        gain[np.isinf(gain)] = 1.0
        np.clip(gain, 0.1, 3.0, out=gain)
        return gain

    return np.ones_like(image, dtype=np.float64)


def load_average_tile(path: str, dark_level: float = 0.0) -> np.ndarray:
    """Loads average tile from file and normalizes it by median.

    Args:
        path (str): File path
        dark_level (float): Camera dark-current baseline (passed to
            :func:`normalize_image_by_median`).

    Returns:
        np.ndarray: Normalized average tile
    """
    tile = read_image(path)
    return normalize_image_by_median(tile, dark_level=dark_level)


# ---------------------------------------------------------------------------
# Adaptive noise-floor estimation for vignetting correction
# ---------------------------------------------------------------------------
# The noise floor (camera baseline / dark current) varies between datasets
# depending on detector type, PMT gain, and acquisition settings.  Rather
# than setting a fixed threshold, we estimate it automatically from a
# random sample of tile medians using Otsu's method to separate empty
# (noise-only) tiles from illuminated (tissue + agarose) tiles.
# ---------------------------------------------------------------------------


def _estimate_noise_floors(section_jsons, n_channels, max_samples=200):
    """Auto-detect per-channel noise floor from tile median statistics.

    Samples a random subset of tiles, computes their median intensity, and
    uses Otsu's method on the *tile-median distribution* to separate the
    empty-tile cluster (noise) from the illuminated-tile cluster (tissue /
    agarose).  The noise floor is then estimated as the mean of the empty-
    tile cluster, and a per-pixel threshold is set at ``noise_mean + 3σ``
    (standard 3-sigma cutoff).

    This two-level thresholding avoids the problems with both Otsu *within*
    tiles (bimodal assumption fails on empty or uniformly-illuminated tiles)
    and with fixed thresholds (don't generalise across detectors):

    * **Tile-level** — tiles whose median is below the Otsu split are
      skipped entirely (they are pure noise and carry no vignetting info).
    * **Pixel-level** — within an illuminated tile, pixels below
      ``noise_mean + 3σ`` are excluded.  This removes background corners
      on partially-illuminated tiles without touching the vignetting-
      darkened edges (which are still well above the noise floor).

    Args:
        section_jsons (list): All section data dicts.
        n_channels (int): Number of image channels.
        max_samples (int): Max tiles to sample per channel (default 200).

    Returns:
        list[dict]: Per-channel dict with keys ``'tile_threshold'`` and
        ``'pixel_threshold'``.  Both are 0.0 when no empty tiles are
        detected (meaning include everything).
    """
    import random as _random
    rng = _random.Random(42)  # fixed seed for reproducibility

    # Collect all tile paths per channel
    tiles_by_ch = [[] for _ in range(n_channels)]
    for sj in section_jsons:
        for t in sj['tiles']:
            ch = t['channel'] - 1
            if 0 <= ch < n_channels:
                tiles_by_ch[ch].append(t['path'])

    results = []
    for ch in range(n_channels):
        paths = tiles_by_ch[ch]
        if not paths:
            results.append({'tile_threshold': 0.0, 'pixel_threshold': 0.0})
            continue

        # Sample a subset for speed
        sampled = (rng.sample(paths, min(max_samples, len(paths)))
                   if len(paths) > max_samples else paths)

        # Compute tile medians in parallel (I/O-bound; cv2.imread releases GIL)
        def _read_median(p):
            try:
                return float(np.median(read_image(p)))
            except (IOError, OSError, RuntimeError):
                return None

        with ThreadPoolExecutor(max_workers=8) as pool:
            medians = [m for m in pool.map(_read_median, sampled) if m is not None]

        if len(medians) < 2:
            results.append({'tile_threshold': 0.0, 'pixel_threshold': 0.0})
            continue

        medians = np.array(medians, dtype=np.float64)
        mn, mx = medians.min(), medians.max()

        # If all tile medians are nearly identical, no split is meaningful
        if mx - mn < 1.0:
            results.append({'tile_threshold': 0.0, 'pixel_threshold': 0.0})
            continue

        # --- Otsu on tile medians ---
        # Scale to uint8 for cv2.threshold (Otsu needs integer histogram).
        scaled = ((medians - mn) / (mx - mn) * 255).astype(np.uint8)
        otsu_val, _ = cv2.threshold(
            scaled.reshape(-1, 1), 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        tile_thresh = mn + (otsu_val / 255.0) * (mx - mn)

        # Safety: if the threshold falls above 50 % of the median range,
        # there are likely no truly empty tiles — include everything.
        if (tile_thresh - mn) / (mx - mn) > 0.5:
            results.append({'tile_threshold': 0.0, 'pixel_threshold': 0.0})
            continue

        # --- Noise floor from the dark-tile cluster ---
        dark_medians = medians[medians <= tile_thresh]
        if len(dark_medians) > 0:
            noise_mean = float(np.mean(dark_medians))
            noise_std = (float(np.std(dark_medians))
                         if len(dark_medians) > 1
                         else noise_mean * 0.5)
            pixel_thresh = noise_mean + 3.0 * noise_std
        else:
            noise_mean = 0.0
            pixel_thresh = 0.0

        results.append({
            'tile_threshold': float(tile_thresh),
            'pixel_threshold': float(max(pixel_thresh, 1.0)),
            'dark_level': float(noise_mean),
        })

    return results


def get_section_avg(tiles: list, n_channels: int = 4,
                    noise_thresholds=None):
    """Per-pixel illumination-weighted accumulation for vignetting estimation.

    Uses auto-detected thresholds (from :func:`_estimate_noise_floors`) at
    two levels:

    1. **Tile level** — tiles whose median falls below the channel's
       ``tile_threshold`` are skipped entirely (they are empty / noise).
    2. **Pixel level** — within accepted tiles, only pixels above the
       channel's ``pixel_threshold`` (noise floor + 3σ) are accumulated.
       This excludes background corners on partially-illuminated tiles
       without discarding the vignetting-darkened but still-illuminated
       edges.

    Returns ``(sums, counts)`` rather than pre-divided averages so that
    :func:`generate_avg_tiles` can compute a proper global pixel-weighted
    mean across all sections.

    Args:
        tiles (list): Tile information dicts (must contain 'path', 'channel').
        n_channels (int): Number of image channels.
        noise_thresholds (list[dict] | None): Per-channel dicts with
            ``'tile_threshold'`` and ``'pixel_threshold'`` keys, as
            returned by :func:`_estimate_noise_floors`.  If *None*,
            all tiles and pixels are included (no filtering).

    Returns:
        tuple: ``(sums, counts)`` — two lists of length *n_channels*.
            *sums*   — ``float32 (832, 832)`` accumulated pixel values.
            *counts* — ``int32   (832, 832)`` number of contributions.
    """
    sums = [np.zeros((832, 832), dtype=np.float32) for _ in range(n_channels)]
    counts = [np.zeros((832, 832), dtype=np.int32) for _ in range(n_channels)]

    for tile in tiles:
        try:
            im = read_image(tile['path'])
            ch = tile["channel"] - 1
            if ch >= n_channels:
                continue

            # --- Per-channel adaptive thresholds ---
            if noise_thresholds and ch < len(noise_thresholds):
                tile_thresh = noise_thresholds[ch]['tile_threshold']
                pixel_thresh = noise_thresholds[ch]['pixel_threshold']
            else:
                tile_thresh = 0.0
                pixel_thresh = 0.0

            # Skip entire tile if its median is below the tile threshold
            if tile_thresh > 0 and np.median(im) < tile_thresh:
                continue

            # Per-pixel: accumulate only pixels above the noise floor
            im_f = im.astype(np.float32)
            if pixel_thresh > 0:
                illuminated = im > pixel_thresh
                sums[ch] += np.where(illuminated, im_f, 0.0)
                counts[ch] += illuminated.astype(np.int32)
            else:
                # No threshold — include all pixels
                sums[ch] += im_f
                counts[ch] += 1  # broadcasts: every pixel counted

        except (IOError, OSError, RuntimeError):
            logging.info('Did not find image tile for channel %d (zero-indexed)',
                         tile.get('channel', 0) - 1)

    return sums, counts


# Sigma (in pixels) for the Gaussian smooth applied to the final
# average tile.  The vignetting field is an optical property of the
# objective and varies on the scale of the full tile; smoothing with
# sigma=80 (~9.6 % of the 832-pixel tile width) aggressively suppresses
# residual tissue-texture noise — especially important when only a few
# sections are available — while faithfully preserving the smooth,
# slowly-varying vignetting shape.
_VIGNETTE_SMOOTH_SIGMA = 30


def generate_avg_tiles(section_jsons: list, avg_tiles_dir: str, n_threads: int,
                      n_channels: int = 4):
    """Generates average tiles for each channel.

    First estimates the per-channel noise floor by sampling tile medians
    and applying Otsu to the tile-median distribution.  Then processes all
    sections in parallel, accumulating only illuminated pixels.  Finally
    computes the global average, fills gaps, and gently Gaussian-smooths
    to isolate the slowly-varying vignetting field.

    Args:
        section_jsons (list): Data for each section.
        avg_tiles_dir (str): Output directory for average tile TIFFs.
        n_threads (int): Number of parallel workers.
        n_channels (int): Number of image channels.
    """
    os.makedirs(avg_tiles_dir, exist_ok=True)

    # --- Auto-detect per-channel noise floor ---
    print(f"  {C_INFO}Estimating noise floor from tile medians...{C_RESET}")
    noise_thresholds = _estimate_noise_floors(section_jsons, n_channels)
    for ch, t in enumerate(noise_thresholds):
        if t['tile_threshold'] > 0:
            print(f"    {C_INFO}Ch {ch}: tile thresh = {C_VALUE}{t['tile_threshold']:.1f}{C_RESET}"
                  f"{C_INFO}, pixel thresh = {C_VALUE}{t['pixel_threshold']:.1f}{C_RESET}"
                  f"{C_INFO}, dark level = {C_VALUE}{t.get('dark_level', 0):.1f}{C_RESET}")
        else:
            print(f"    {C_INFO}Ch {ch}: no empty tiles detected, including all{C_RESET}")

    # --- Parallel per-section accumulation with adaptive thresholds ---
    results = Parallel(n_jobs=n_threads, verbose=13)(
        delayed(get_section_avg)(sj['tiles'], n_channels, noise_thresholds)
        for sj in section_jsons
    )

    # --- Global pixel-weighted mean across all sections ---
    global_sums = [np.zeros((832, 832), dtype=np.float64) for _ in range(n_channels)]
    global_counts = [np.zeros((832, 832), dtype=np.int64) for _ in range(n_channels)]

    for sec_sums, sec_counts in results:
        for ch in range(min(n_channels, len(sec_sums))):
            global_sums[ch] += sec_sums[ch]
            global_counts[ch] += sec_counts[ch]
    del results

    for ch in range(n_channels):
        valid = global_counts[ch] > 0
        avg = np.zeros((832, 832), dtype=np.float64)

        if np.any(valid):
            avg[valid] = global_sums[ch][valid] / global_counts[ch][valid]

            # Fill positions that were never covered by tissue with the
            # median of the valid region so the subsequent smooth
            # propagates reasonable values into the gaps.
            if not np.all(valid):
                avg[~valid] = np.median(avg[valid])

            # Gentle Gaussian smooth to isolate the slowly-varying
            # vignetting component and suppress any residual
            # tissue-texture noise that survived the cross-section
            # averaging.  With 30+ sections averaged, σ=30 is enough
            # to remove residual tissue structure while preserving
            # the true (potentially asymmetric) vignetting shape.
            #
            # Edge-aware padding: the vignetting field drops steeply
            # at tile edges.  OpenCV's default BORDER_REFLECT mirrors
            # brighter interior pixels into the beyond-edge region,
            # biasing the smoothed average HIGH at boundaries.  This
            # makes the flat-field gain at edges too LOW, under-
            # correcting both overlap edges and producing visible
            # dark seam lines.  Padding with 'odd' reflection
            # (linear extrapolation) continues the vignetting decline
            # past the tile boundary, eliminating the bias.
            _pad = int(np.ceil(3.0 * _VIGNETTE_SMOOTH_SIGMA))
            avg_padded = np.pad(avg, _pad, mode='reflect',
                                reflect_type='odd')
            np.clip(avg_padded, 1.0, None, out=avg_padded)
            h, w = avg.shape
            avg = cv2.GaussianBlur(avg_padded, (0, 0),
                                   sigmaX=_VIGNETTE_SMOOTH_SIGMA
                                   )[_pad:_pad + h,
                                     _pad:_pad + w].copy()
            del avg_padded
        else:
            avg[:] = 1.0  # no tissue found; identity gain

        cv2.imwrite(os.path.join(avg_tiles_dir, f"avg_tile_{ch}.tif"),
                    avg.astype(np.float32))

    return noise_thresholds


def _process_one_tile(tile_params, avg_tiles, H, pX_, pY_, save_undistorted,
                     vignetting_correction=True, noise_thresholds=None,
                     undistorted_dir=None):
    """Process a single tile: read, correct vignetting & deformation, create Tile.
    
    Extracted as a standalone function so tiles can be processed in parallel
    via ThreadPoolExecutor. All operations (cv2, numpy) release the GIL.
    """
    tile = tile_params.copy()
    try:
        im = read_image(tile['path'])

        # Flat-field vignetting correction: subtracts the camera dark
        # level before applying the gain map, then adds it back.  This
        # prevents the gain from amplifying the noise floor differently
        # across the tile (e.g. 0.9× at top vs 2.3× at bottom), which
        # would create sharp intensity seams at every tile boundary.
        #   corrected = (raw - dark) × gain + dark
        if vignetting_correction:
            ch_idx = tile['channel'] - 1
            if ch_idx < len(avg_tiles):
                avg = avg_tiles[ch_idx]
                im = im.astype(np.float32)
                dark = np.float32(0)
                if noise_thresholds and ch_idx < len(noise_thresholds):
                    dark = np.float32(noise_thresholds[ch_idx].get('dark_level', 0))
                if dark > 0:
                    im -= dark
                np.multiply(im, avg, out=im)
                if dark > 0:
                    im += dark

        # Correct deformation
        im_corrected = correct_deformation(im, H, pX_, pY_)
        del im  # free raw tile immediately
        tile['image'] = im_corrected
        if save_undistorted and undistorted_dir is not None:
            undistorted_tile_path = os.path.join(undistorted_dir, 
                                                 "ch{}".format(tile['channel'] - 1), 
                                                 os.path.split(tile['path'])[1])
            write_output(np.ascontiguousarray(im_corrected), undistorted_tile_path)
        tile['is_missing'] = False

    # If the tile is missing, set the image to None and is_missing to True
    except (IOError, OSError, RuntimeError) as err:
        tile['image'] = None
        tile['is_missing'] = True

    # Decrement channel by 1 to make it zero-indexed
    tile['channel'] = tile['channel'] - 1
    return Tile(**tile)


def generate_tiles(tiles: list, avg_tiles: list, 
                   H, pX_, pY_, ch: int = None, 
                   save_undistorted: bool = False,
                   vignetting_correction: bool = True,
                   noise_thresholds=None,
                   undistorted_dir=None):
    """Generate the images for the tiles and applies processing on them.

    Tiles are processed in parallel using threads. The per-tile work (I/O,
    masking, deformation correction) involves numpy/cv2 operations that
    release the GIL, so threading provides effective parallelism.

    Args:
        tiles (list): Tile information
        avg_tiles (list): List of average tiles
        H: Homography information
        pX_: Bezier deformation x-coordinate map
        pY_: Bezier deformation y-coordinate map
        ch (int, optional): Which channel to generate for. Defaults to None (all).
        save_undistorted (bool, optional): Save without distortion correction. Defaults to False.
        vignetting_correction (bool, optional): Apply average tile vignetting correction. Defaults to True.
        noise_thresholds (list[dict] | None): Per-channel noise thresholds
            with ``'dark_level'`` for flat-field correction.

    Returns:
        list: Processed Tile objects
    """    
    # Filter to specific channel if requested
    if ch is not None:
        tiles = [t for t in tiles if t['channel'] == ch + 1]

    if not tiles:
        return []

    # Process tiles in parallel with ThreadPoolExecutor
    # max_workers=4 provides I/O overlap without excessive contention
    # when called from within an outer joblib.Parallel loop
    max_workers = min(4, len(tiles))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(_process_one_tile, t, avg_tiles, H, pX_, pY_,
                            save_undistorted,
                            vignetting_correction, noise_thresholds,
                            undistorted_dir)
            for t in tiles
        ]
        return [f.result() for f in futures]


def create_section_json(sno: int, sectionName: str, mosaic_data: list):
    """Creates a JSON object for storing section information.

    Args:
        sno (int): Section index number
        sectionName (str): Name of the section
        mosaic_data (list): Mosaic data information

    Returns:
        _type_: Section JSON data
    """
    import re as _re
    
    tyx = -3
    tyy = -43
    txx = -25
    txy = 5
    margins = {"row": 0, "column": 0}
    size = {"row": 774, "column": 756}
    startx = 200
    starty = 200

    mcolumns = int(mosaic_data["mcolumns"])
    mrows = int(mosaic_data["mrows"])
    tiles_per_position = mrows * mcolumns
    image_dimensions = {"row": mrows * size['row'] + 2 * startx, 
                        "column": mcolumns * size['column'] + 2 * starty}
    
    # Auto-detect tile indices from actual files in the section folder.
    # This handles both old format (0-based per-section indices) and new
    # format (globally cumulative indices).
    all_tifs = glob.glob(os.path.join(sectionName, '*.tif'))
    tile_indices = set()
    for fpath in all_tifs:
        fname = os.path.basename(fpath)
        # Extract the numeric index from filename pattern: *-{index}_{channel}.tif
        m = _re.search(r'-(\d+)_\d+\.tif$', fname)
        if m:
            tile_indices.add(int(m.group(1)))
    tile_indices = sorted(tile_indices)
    
    # Use the first tiles_per_position indices (layer 0).
    layer_indices = tile_indices[:tiles_per_position]
    
    # Build a mapping from sequential position to actual tile index
    # Position order: column-major (ncol outer, nrow inner), matching the
    # original traversal order used by the microscope.
    section_json = {}
    section_json["mosaic_parameters"] = mosaic_data
    tiles = []
    pos = 0
    for ncol in range(mcolumns):
        for nrow in range(mrows):
            if pos >= len(layer_indices):
                pos += 1
                continue
            index = layer_indices[pos]
            pos += 1
            
            tile_paths = sorted(glob.glob(f'{sectionName}/*-{index}_*.tif'))
            if len(tile_paths) == 0:
                continue
            bounds = {}
            row = {}
            col = {}
            if ncol % 2 == 0:
                row["start"] = starty + nrow * size["row"] + nrow * tyy + ncol * txy 
                row["end"] = row["start"] + size["row"]
                col["start"] = startx + ncol * size["column"] + ncol * txx + nrow * tyx 
                col["end"] = col["start"] + size["column"]
            else:
                row["start"] = starty + (mrows - nrow - 1) * size["row"] + (mrows - nrow - 1) * tyy + ncol * txy 
                row["end"] = row["start"] + size["row"]
                col["start"] = startx + ncol * size["column"] + ncol * txx + (mrows - nrow - 1) * tyx 
                col["end"] = col["start"] + size["column"]
            bounds["row"] = row
            bounds["column"] = col
            for ch, path in enumerate(tile_paths):
                tile_data = {}
                tile_data["path"] = path
                tile_data["bounds"] = bounds
                tile_data["margins"]= margins
                tile_data["size"] = size
                tile_data["channel"] = ch + 1
                tile_data["index"] = index
                tiles.append(tile_data)
    # Derive channel list from actual tile files, not mosaic metadata.
    # The mosaic file may report more channels than exist on disk.
    unique_channels = sorted(set(t["channel"] for t in tiles))
    section_json["channels"] = unique_channels if unique_channels else [1]
    section_json["tiles"] = tiles
    section_json['slice_fname'] = os.path.split(sectionName)[-1] + "_1"
    section_json["image_dimensions"] = image_dimensions
    return section_json


def get_section_data(root_dir: str, n_threads: int, sectionNum: int = -1):
    """Retrieve and generate section data from root input directory.

    Args:
        root_dir (str): Input directory
        n_threads (int): How many threads to run the section data generation
        sectionNum (int, optional): Which specific section number to generate information for. 
                                    If set to -1, generates for all sections. Defaults to -1.

    Returns:
        _type_: Section data
    """
    files = glob.glob(root_dir + 'Mosaic*')
    print(f"  {C_INFO}Input directory: {C_PATH}{root_dir}{C_RESET}")
    print(f"  {C_INFO}Mosaic files found: {C_VALUE}{len(files)}{C_RESET}")
    # Look for mosaic file
    if len(files) == 0:
        raise FileNotFoundError(f"No Mosaic file found in {root_dir}. "
                                "Expected a file matching '*Mosaic*'.")
    else:
        mosaic_file = files[0]

    mosaic_data = {}
    with open(mosaic_file) as fp:
        for line in fp:
            k,v = line.rstrip("\n").split(":",1)
            mosaic_data[k]=v

    sectionNames = glob.glob(root_dir + mosaic_data["Sample ID"] + "*")
    # Filter to directories only (avoid matching log files or other non-directory entries)
    sectionNames = sorted([s for s in sectionNames if os.path.isdir(s)])
    
    # If a specific section number is provided, generate the section data for that section only
    if sectionNum != -1:
        sectionName = os.path.join(root_dir, "{}-{:04d}".format(mosaic_data["Sample ID"], sectionNum + 1))
        section_jsons = [create_section_json(sectionNum, sectionName, mosaic_data)]
        return mosaic_data, section_jsons
    
    # Otherwise, generate section data for all sections
    section_jsons = Parallel(n_jobs=n_threads)(delayed(create_section_json)(sno, sectionName, mosaic_data) 
                                               for sno,sectionName in enumerate(sectionNames))

    return mosaic_data, section_jsons


def stitch_section(data: dict, avg_tiles: list, output_dir: str, H, pX_, pY_, 
                   ch: int = None, save_undistorted: bool = False,
                   blend_mode: str = 'multiband',
                   vignetting_correction: bool = True,
                   noise_thresholds=None,
                   seam_correction: bool = True,
                   zero_background: bool = False):
    """Stitches the tiles together to create a complete section.

    Args:
        data (dict): Section data
        avg_tiles (list): List of average tiles for each channel
        output_dir (str): Output directory to save stitched images
        H (_type_): Homography information
        pX_ (_type_): _description_
        pY_ (_type_): _description_
        ch (int, optional): Which channel to stitch for. If None is provided, stitch all channels. Defaults to None.
        save_undistorted (bool, optional): Whether or not to save without distortion correction. Defaults to False.
        blend_mode (str, optional): Blending mode - 'multiband' or 'linear'. Defaults to 'multiband'.
        vignetting_correction (bool, optional): Apply average tile vignetting correction. Defaults to True.
        noise_thresholds (list[dict] | None): Per-channel noise thresholds for flat-field correction.
    """
    
    # Derive undistorted_dir from output_dir when save_undistorted is enabled
    _undistorted_dir = os.path.join(output_dir, 'undistorted') if save_undistorted else None

    tiles = generate_tiles(data['tiles'], avg_tiles, H, pX_, pY_, ch, save_undistorted,
                           vignetting_correction, noise_thresholds,
                           _undistorted_dir)

    # Derive per-channel noise floors for gain compensation.
    # Uses pixel_threshold + 5 to ensure background pixels are excluded
    # while tissue is included.  pixel_threshold is auto-detected via
    # Otsu's method on tile-median distributions, so this adapts to any
    # dataset without hardcoded thresholds.
    seam_noise_floors = None
    if noise_thresholds is not None:
        seam_noise_floors = [
            nt.get('pixel_threshold', 30.0) + 5.0
            for nt in noise_thresholds
        ]

    stitcher = Stitcher(data['image_dimensions'], tiles, data['channels'],
                        blend_mode=blend_mode,
                        noise_floors=seam_noise_floors,
                        seam_correction=seam_correction,
                        zero_background=zero_background)
    image, missing = stitcher.run()
    # Tiles' image data was already freed inside stitcher.run()
    del tiles, stitcher; gc.collect()
    missing_tile_paths = get_missing_tile_paths(missing)

    # Write each channel individually and free its slice to avoid keeping
    # the full multi-channel mosaic in memory while writing.
    if ch is None:
        for c in range(image.shape[2]):
            slice_path = os.path.join(output_dir, "stitched_ch{}".format(c), data['slice_fname'] + "_{}.tif".format(c))
            print(f"  {C_SUCCESS}Saved:{C_RESET} {C_PATH}{slice_path}{C_RESET}")
            write_output(image[:, :, c], slice_path)
    else:
        slice_path = os.path.join(output_dir, "stitched_ch{}".format(ch), data['slice_fname'] + "_{}.tif".format(ch))
        print(f"  {C_SUCCESS}Saved:{C_RESET} {C_PATH}{slice_path}{C_RESET}")
        write_output(image[:, :, ch], slice_path)
    del image; gc.collect()
       

def _preview_one_image(tif_path: str, dst_dir: str, scale: float):
    """Convert one stitched TIFF to a contrast-stretched 8-bit PNG preview.

    Downscales first to reduce memory and speed up the percentile
    computation, then applies 1st-99th percentile contrast stretching
    with a background-mode heuristic that pushes the low clip point
    above the camera dark level so tile-to-tile noise variations
    become uniformly black instead of a visible grid.
    """
    im = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return

    # Downscale first — percentile on a 15% image is ~44× cheaper
    if scale != 1.0:
        new_w = max(1, int(im.shape[1] * scale))
        new_h = max(1, int(im.shape[0] * scale))
        im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # ── Global percentile contrast stretching ──
    p_lo, p_hi = np.percentile(im, (1, 99.5))

    # Push p_lo above the background mode so tile-to-tile dark-level
    # variations become uniformly black instead of a visible grid.
    nz = im[im > 0].ravel()
    if len(nz) > 1000:
        q25 = float(np.percentile(nz, 25))
        hist_c, hist_e = np.histogram(
            nz, bins=50, range=(0, max(q25 * 2, 30)))
        bg_mode = (hist_e[int(np.argmax(hist_c))]
                   + hist_e[int(np.argmax(hist_c)) + 1]) / 2.0
        p_lo = max(p_lo, bg_mode + 2)

    # Enforce minimum display range to avoid extreme amplification
    _MIN_RANGE = 10.0
    if (p_hi - p_lo) < _MIN_RANGE:
        p_hi = p_lo + _MIN_RANGE

    # For low-dynamic-range channels (e.g. CH2 with tissue at 20-30
    # counts), integer quantization creates 1-count steps at tile
    # boundaries that stretch to ~15 preview levels.  An isotropic
    # Gaussian blur before stretching smooths both horizontal and
    # vertical tile steps without affecting tissue features
    # (sigma=3 at 15% scale ≈ 130 μm).
    # A median filter also removes isolated bright specks (agarose
    # particles, hot pixels) that would otherwise dominate the
    # stretched image.
    _LOW_RANGE_THRESH = 40.0
    display_range = p_hi - p_lo
    if display_range < _LOW_RANGE_THRESH:
        from scipy.ndimage import gaussian_filter, median_filter
        im = median_filter(im, size=3)
        im = gaussian_filter(im.astype(np.float32), sigma=3.0)

    stretched = np.clip((im.astype(np.float32) - p_lo) / (p_hi - p_lo) * 255.0,
                        0, 255).astype(np.uint8)

    out_name = os.path.splitext(os.path.basename(tif_path))[0] + ".png"
    cv2.imwrite(os.path.join(dst_dir, out_name), stretched)


def generate_preview_images(output_dir: str, n_channels: int,
                           channel: int = None, scale: float = 0.15):
    """Generate contrast-stretched 8-bit PNG previews of stitched sections.

    Reads each stitched TIFF, downscales, applies 1st-99th percentile
    contrast stretching, and saves as an 8-bit PNG in a ``preview/``
    subfolder.  Images are processed in parallel with threads (I/O-bound
    work; cv2/numpy release the GIL).

    Args:
        output_dir (str): Root output directory containing ``stitched_chN/``.
        n_channels (int): Number of image channels.
        channel (int | None): If set, only preview this channel.
        scale (float): Downscale factor for previews (default 0.15 = 15%).
    """
    import glob

    channels = [channel] if channel is not None else range(n_channels)

    for ch in channels:
        src_dir = os.path.join(output_dir, f"stitched_ch{ch}")
        if not os.path.isdir(src_dir):
            continue

        dst_dir = os.path.join(output_dir, "preview", f"ch{ch}")
        os.makedirs(dst_dir, exist_ok=True)

        tif_files = sorted(glob.glob(os.path.join(src_dir, "*.tif")))
        if not tif_files:
            continue

        max_workers = min(8, len(tif_files))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_preview_one_image, p, dst_dir, scale)
                for p in tif_files
            ]
            for f in futures:
                f.result()  # propagate any exceptions

        print(f"  {C_SUCCESS}Preview:{C_RESET} {C_PATH}{dst_dir}{C_RESET}  ({len(tif_files)} images)")


def _mask_preview_one_image(tif_path: str, preview_dir: str, mask_dir: str,
                            noise_floor: float, scale: float):
    """Generate a mask overlay preview for one stitched TIFF.

    Loads the corresponding preview PNG (already contrast-stretched),
    computes the brain mask from the full-res TIFF using the same
    morphological pipeline as ``_zero_background``, downscales the mask
    to preview resolution, and overlays background regions in
    transparent red.
    """
    from scipy.ndimage import (binary_fill_holes, binary_closing,
                                binary_opening, label)

    # Load the preview PNG
    base_name = os.path.splitext(os.path.basename(tif_path))[0] + ".png"
    preview_path = os.path.join(preview_dir, base_name)
    preview = cv2.imread(preview_path, cv2.IMREAD_UNCHANGED)
    if preview is None:
        return

    # Load the stitched TIFF to compute the mask
    im = cv2.imread(tif_path, cv2.IMREAD_UNCHANGED)
    if im is None:
        return

    # Build brain mask at 4x-downsampled resolution (same as _zero_background)
    _DS = 4
    H, W = im.shape[:2]
    small = im[::_DS, ::_DS].astype(np.float32)
    h_ds, w_ds = small.shape[:2]
    tissue_mask = small > noise_floor

    # Open to remove isolated noise pixels in the background
    kern_open = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (5, 5)).astype(bool)
    opened = binary_opening(tissue_mask, structure=kern_open)

    # Keep only large connected components (actual brain sections)
    labeled, n_features = label(opened)
    min_area = h_ds * w_ds * 0.005
    brain_mask = np.zeros_like(opened)
    for i in range(1, n_features + 1):
        comp = labeled == i
        if comp.sum() > min_area:
            brain_mask |= comp

    # Close to smooth edges and bridge small gaps
    kern_r = 15
    kern_close = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (2 * kern_r + 1, 2 * kern_r + 1)
    ).astype(bool)
    closed = binary_closing(brain_mask, structure=kern_close)
    filled = binary_fill_holes(closed)

    # Downscale the mask to preview resolution
    ph, pw = preview.shape[:2]
    mask_preview = cv2.resize(
        filled.astype(np.uint8), (pw, ph),
        interpolation=cv2.INTER_NEAREST
    ).astype(bool)

    # Convert grayscale preview to BGR for the red overlay
    if preview.ndim == 2:
        preview_bgr = cv2.cvtColor(preview, cv2.COLOR_GRAY2BGR)
    else:
        preview_bgr = preview.copy()

    # Overlay transparent red on background (outside mask)
    alpha = 0.4
    bg = ~mask_preview
    overlay = preview_bgr.copy()
    overlay[bg, 2] = np.clip(
        overlay[bg, 2].astype(np.float32) * (1 - alpha) + 255 * alpha,
        0, 255).astype(np.uint8)  # Red channel
    overlay[bg, 1] = (overlay[bg, 1].astype(np.float32) * (1 - alpha)
                      ).astype(np.uint8)  # Green channel
    overlay[bg, 0] = (overlay[bg, 0].astype(np.float32) * (1 - alpha)
                      ).astype(np.uint8)  # Blue channel

    cv2.imwrite(os.path.join(mask_dir, base_name), overlay)


def generate_mask_previews(output_dir: str, n_channels: int,
                           noise_thresholds: list,
                           channel: int = None, scale: float = 0.15):
    """Generate mask overlay previews showing background vs brain regions.

    For each stitched TIFF, overlays the brain mask in transparent red
    on the contrast-stretched preview PNG and saves to a
    ``preview/ch{N}_mask/`` subfolder.

    Args:
        output_dir (str): Root output directory.
        n_channels (int): Number of image channels.
        noise_thresholds (list[dict]): Per-channel noise threshold dicts.
        channel (int | None): If set, only process this channel.
        scale (float): Preview downscale factor (must match preview generation).
    """
    import glob

    # Derive the same seam_noise_floors used by the Stitcher.
    # When noise_thresholds is None (e.g. single-section runs or
    # vignetting correction disabled), fall back to a sensible default.
    if noise_thresholds is not None:
        seam_noise_floors = [
            nt.get('pixel_threshold', 30.0) + 5.0
            for nt in noise_thresholds
        ]
    else:
        seam_noise_floors = [35.0] * n_channels

    channels = [channel] if channel is not None else range(n_channels)

    for ch in channels:
        src_dir = os.path.join(output_dir, f"stitched_ch{ch}")
        preview_dir = os.path.join(output_dir, "preview", f"ch{ch}")
        mask_dir = os.path.join(output_dir, "preview", f"ch{ch}_mask")

        if not os.path.isdir(src_dir) or not os.path.isdir(preview_dir):
            continue

        os.makedirs(mask_dir, exist_ok=True)

        tif_files = sorted(glob.glob(os.path.join(src_dir, "*.tif")))
        if not tif_files:
            continue

        nf = seam_noise_floors[ch] if ch < len(seam_noise_floors) else 30.0

        max_workers = min(8, len(tif_files))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(_mask_preview_one_image, p, preview_dir,
                                mask_dir, nf, scale)
                for p in tif_files
            ]
            for f in futures:
                f.result()

        print(f"  {C_SUCCESS}Mask preview:{C_RESET} {C_PATH}{mask_dir}{C_RESET}  ({len(tif_files)} images)")


if __name__ == '__main__':
    # Setup import setting
    joblib_backend = None
    if sys.platform == 'win32':
        joblib_backend = 'multiprocessing'

    # Run main
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    parser = argparse.ArgumentParser(description='TissueCyte stitching pipeline')
    parser.add_argument('--input_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--bezier_path', type=str, default=None,
                        help='Path to bezier patch .pkl file (default: data/stitching_parameters/bezier16x.pkl)')
    parser.add_argument('--sectionNum', default=-1, type=int)
    parser.add_argument('--n_threads', default=8, type=int,
                        help='Number of parallel jobs (default: 8; use -1 for all CPUs)')
    parser.add_argument('--save_undistorted', action='store_true',
                        help='Save undistorted images')
    parser.add_argument('--linear_blend', action='store_true',
                        help='Use simple linear blending instead of multi-band Laplacian pyramid (default: multiband)')
    parser.add_argument('--no_vignetting', action='store_true',
                        help='Disable average tile vignetting correction (enabled by default)')
    parser.add_argument('--no_seam_correction', action='store_true',
                        help='Disable tile equalization and seam corrections (enabled by default)')
    parser.add_argument('--zero_background', action='store_true',
                        help='[Beta] Zero pixels outside the brain boundary for uniform black '
                             'background. Uses morphological brain mask to preserve dark '
                             'internal structures (ventricles, fiber tracts).')
    args = parser.parse_args()

    n_threads = args.n_threads
    channel = None
    blend_mode = 'linear' if args.linear_blend else 'multiband'
    save_undistorted = args.save_undistorted
    vignetting_correction = not args.no_vignetting
    seam_correction = not args.no_seam_correction
    zero_background = args.zero_background
    sectionNum = args.sectionNum

    # Setup bezier patch
    bezier_path = args.bezier_path
    if bezier_path is None:
        # Try default locations
        for candidate in [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "stitching_parameters", "bezier16x.pkl"),
            "bezier16x.pkl",
            os.path.join("data", "stitching_parameters", "bezier16x.pkl"),
        ]:
            if os.path.exists(candidate):
                bezier_path = candidate
                break
    if bezier_path is None or not os.path.exists(bezier_path):
        print(f"{C_ERROR}ERROR: Bezier patch file not found. Specify with --bezier_path{C_RESET}")
        sys.exit(1)

    # Print startup banner
    print(f"\n{C_HEADER}{'='*70}{C_RESET}")
    print(f"{C_HEADER}{'XuLab Stitching Pipeline':^70}{C_RESET}")
    print(f"{C_HEADER}{'='*70}{C_RESET}\n")

    print(f"{C_INFO}STARTUP PARAMETERS{C_RESET}")
    print(f"{'-'*50}")
    print(f"  Input directory:   {C_PATH}{args.input_dir}{C_RESET}")
    print(f"  Output directory:  {C_PATH}{args.output_dir}{C_RESET}")
    print(f"  Bezier patch:      {C_PATH}{bezier_path}{C_RESET}")
    print(f"  Section number:    {C_VALUE}{args.sectionNum if args.sectionNum != -1 else 'All'}{C_RESET}")
    print(f"  Parallel jobs:     {C_VALUE}{n_threads}{C_RESET}")
    print(f"  Save undistorted:  {C_VALUE}{'Yes' if save_undistorted else 'No'}{C_RESET}")
    print(f"  Vignetting corr:   {C_VALUE}{'Yes' if vignetting_correction else 'No'}{C_RESET}")
    print(f"  Seam correction:   {C_VALUE}{'Yes' if seam_correction else 'No'}{C_RESET}")
    print(f"  Zero background:   {C_VALUE}{'Yes' if zero_background else 'No'}{C_RESET}")
    print(f"  Blending mode:     {C_VALUE}{blend_mode.capitalize()}{C_RESET}")
    print(f"{'-'*50}")
    print(f"  Started at:        {C_VALUE}{time.strftime('%Y-%m-%d %H:%M:%S')}{C_RESET}")
    print(f"{'-'*50}\n")

    pipeline_start = time.time()

    # Setup globally
    corners1 = np.asarray([[33, 10], [796, 21], [30, 813], [793, 818]])
    corners2 = np.asarray([[20, 20], [776, 20], [20, 794], [776, 794]])
    H, _ = cv2.findHomography(corners1, corners2)
    gridp = create_perfect_grid(42, 43, 4, 18)
    gridp = gridp[20:794, 20:776]

    kx, ky = joblib.load(bezier_path)

    # Double the size to preserve sampling, need to downsample later
    pX_, pY_ = get_deformation_map(gridp.shape[0], gridp.shape[1], kx, ky)

    root_dir = os.path.join(args.input_dir, '')

    # Create a timestamped subfolder inside the output directory so each
    # run is isolated:  <output_dir>/YYYYMMDD_HHMMSS_<brain_name>/
    # Skip when the provided path already looks like a timestamped run
    # folder (e.g. the GUI already created one).
    import re as _re_ts
    base_output = os.path.join(args.output_dir, '')
    if not os.path.isdir(base_output):
        os.makedirs(base_output, exist_ok=True)
    _dir_leaf = os.path.basename(os.path.normpath(base_output))
    if _re_ts.match(r'\d{8}_\d{6}_', _dir_leaf):
        # Already a timestamped run folder — use it directly
        output_dir = base_output
    else:
        brain_name = os.path.basename(os.path.normpath(args.input_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        output_dir = os.path.join(base_output, f"{timestamp}_{brain_name}", '')
    os.makedirs(output_dir, exist_ok=True)

    # Mirror all console output to a log file (ANSI codes stripped)
    _tee = _Tee(os.path.join(output_dir, 'console.log'))
    sys.stdout = _tee

    print(f"{C_STEP}Step 1/5:{C_RESET} {C_INFO}Parsing mosaic data...{C_RESET}")
    mosaic_data, section_jsons = get_section_data(root_dir, n_threads, sectionNum)
    print(f"  {C_INFO}Sections found: {C_VALUE}{len(section_jsons)}{C_RESET}")

    # Derive actual channel count from tile files, not mosaic metadata
    # (the mosaic file may report more channels than exist on disk).
    channel_count = max(len(sj['channels']) for sj in section_jsons) if section_jsons else int(mosaic_data['channels'])
    print(f"  {C_INFO}Channels: {C_VALUE}{channel_count}{C_RESET}")

    print(f"\n{C_STEP}Step 2/5:{C_RESET} {C_INFO}Creating output directories...{C_RESET}")
    for ch in range(channel_count):
        ch_dir  = os.path.join(output_dir, "stitched_ch{}".format(ch),"")
        if not os.path.isdir(ch_dir):
            os.mkdir(ch_dir)

    if save_undistorted:
        undistorted_dir = output_dir + "/undistorted"

        if not os.path.isdir(undistorted_dir):
            os.mkdir(undistorted_dir)

        for ch in range(channel_count):
            ch_dir = os.path.join(undistorted_dir, "ch{}".format(ch),"")
            if not os.path.isdir(ch_dir):
                os.mkdir(ch_dir)

    # Generate average tiles for vignetting correction.
    # The average tile captures per-pixel illumination non-uniformity;
    # dividing each tile by it (via a pre-computed gain map) removes
    # the checkerboard pattern visible after stitching.
    average_tiles = []
    noise_thresholds = None
    if vignetting_correction and sectionNum == -1:
        print(f"\n{C_STEP}Step 3/5:{C_RESET} {C_INFO}Generating average tiles for vignetting correction...{C_RESET}")
        avg_tiles_dir = os.path.join(output_dir, "avg_tiles")
        noise_thresholds = generate_avg_tiles(section_jsons, avg_tiles_dir, n_threads,
                                              n_channels=channel_count)
        for i in range(channel_count):
            dark = noise_thresholds[i].get('dark_level', 0) if noise_thresholds and i < len(noise_thresholds) else 0
            average_tiles.append(load_average_tile(os.path.join(avg_tiles_dir, f"avg_tile_{i}.tif"),
                                                   dark_level=dark))
        print(f"  {C_SUCCESS}Average tiles generated ({channel_count} channels).{C_RESET}")
    else:
        if not vignetting_correction:
            print(f"\n{C_STEP}Step 3/5:{C_RESET} {C_INFO}Skipping vignetting correction (disabled){C_RESET}")
        else:
            print(f"\n{C_STEP}Step 3/5:{C_RESET} {C_INFO}Skipping vignetting correction (single section){C_RESET}")
        for i in range(channel_count):
            average_tiles.append(np.ones((832, 832)))

    print(f"\n{C_STEP}Step 4/5:{C_RESET} {C_INFO}Stitching {C_VALUE}{len(section_jsons)}{C_RESET}{C_INFO} sections ({blend_mode} blending)...{C_RESET}")
    #Parallel(n_jobs=1, backend=joblib_backend)(delayed(stitch_section)(section_json,average_tiles, output_dir) for section_json in tqdm(section_jsons))
    Parallel(n_jobs=n_threads, verbose=13)(delayed(stitch_section)(section_json, average_tiles, output_dir, 
                                                                   H, pX_, pY_, channel, save_undistorted,
                                                                   blend_mode,
                                                                   vignetting_correction,
                                                                   noise_thresholds,
                                                                   seam_correction,
                                                                   zero_background) for section_json in section_jsons)

    # Generate contrast-stretched preview images for quick QC
    print(f"\n{C_STEP}Step 5/5:{C_RESET} {C_INFO}Generating preview images...{C_RESET}")
    generate_preview_images(output_dir, channel_count, channel)

    # Generate mask overlay previews when background zeroing is enabled
    if zero_background:
        print(f"\n{C_INFO}Generating mask overlay previews...{C_RESET}")
        generate_mask_previews(output_dir, channel_count, noise_thresholds,
                               channel)

    # Print completion summary
    elapsed = time.time() - pipeline_start
    minutes = int(elapsed // 60)
    seconds = int(elapsed % 60)
    print(f"\n{C_HEADER}{'='*70}{C_RESET}")
    print(f"{C_SUCCESS}Stitching complete!{C_RESET}")
    print(f"  {C_INFO}Sections stitched: {C_VALUE}{len(section_jsons)}{C_RESET}")
    print(f"  {C_INFO}Output directory:  {C_PATH}{output_dir}{C_RESET}")
    print(f"  {C_INFO}Elapsed time:      {C_VALUE}{minutes}m {seconds}s{C_RESET}")
    print(f"{C_HEADER}{'='*70}{C_RESET}")

    # Close console.log
    _tee.close()

    # Offer to open the output folder (only when both stdin and stdout
    # are interactive terminals — avoids blocking when piped or tee'd)
    if sys.stdin.isatty() and sys.stdout.isatty():
        try:
            answer = input(f"\nOpen output folder? [Y/n] ").strip().lower()
            if answer in ('', 'y', 'yes'):
                import subprocess
                abs_dir = os.path.abspath(output_dir)
                if sys.platform == 'win32':
                    os.startfile(abs_dir)
                elif sys.platform == 'darwin':
                    subprocess.Popen(['open', abs_dir])
                else:
                    subprocess.Popen(['xdg-open', abs_dir])
        except (EOFError, KeyboardInterrupt):
            pass
