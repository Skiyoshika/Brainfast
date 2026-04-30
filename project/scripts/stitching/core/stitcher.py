# ruff: noqa
"""
Code provided from Allen Institute
Multi-band Laplacian pyramid blending added by Andy Thai.
"""


import logging
import operator as op
from collections import defaultdict
from functools import reduce

import numpy as np
import cv2

# Working precision for blending — float32 halves memory vs float64
# and provides more than enough precision for uint16 image data.
_BLEND_DTYPE = np.float32


def initialize_images(image_dimensions, n_channels):
    """Create the output mosaic and its overlap indicator.

    Parameters
    ----------
    image_dimensions : dict
        {'row': int, 'column': int} — pixel extents of the full mosaic.
    n_channels : int
        Number of image channels.

    Returns
    -------
    slice_image : np.ndarray, shape (rows, cols, n_channels), dtype float32
        Accumulates the blended mosaic.
    stitched_indicator : np.ndarray, shape (rows, cols, n_channels), dtype uint8
        Binary mask (0/1) recording which pixels have been written.
    """
    rows = int(image_dimensions['row'])
    cols = int(image_dimensions['column'])
    slice_image = np.zeros((rows, cols, n_channels), dtype=_BLEND_DTYPE)
    # uint8 instead of float64 — uses 8x less memory for the indicator
    stitched_indicator = np.zeros((rows, cols, n_channels), dtype=np.uint8)
    return slice_image, stitched_indicator


class Stitcher(object):


    def __init__(self, image_dimensions, tiles, channels, blend_mode='multiband',
                 noise_floors=None, seam_correction=True,
                 zero_background=False):

        logging.info('image_dimensions: {0}'.format(image_dimensions))
        self.image_dimensions = image_dimensions


        self.tiles = tiles
        self.channels = channels
        self.blend_mode = blend_mode  # 'multiband' or 'linear'
        self._meshgrid_cache = {}
        self._seam_correction = seam_correction
        self._zero_background = zero_background
        # Per-channel noise floors for seam correction.  Falls back to
        # the global _GAIN_COMP_NOISE constant when not provided.
        if noise_floors is not None:
            self._noise_floors = [float(nf) for nf in noise_floors]
        else:
            self._noise_floors = None


    def run(self, cb=np.array):

        slice_image, stitched_indicator = initialize_images(self.image_dimensions, len(self.channels))
        missing_tiles = {}

        # Compute geometry-dependent seam correction parameters once
        self._seam_params = _derive_seam_params(self.tiles)

        for tile in self.tiles:

            if tile.is_missing:

                missing_tiles[tile.index] = tile.get_missing_path()
                tile.initialize_image()

            else:
                tile.trim_self()

            self.stitch(slice_image, stitched_indicator, tile, cb)

            # Release tile image data immediately to free memory
            tile.image = None

        sp = self._seam_params

        if self._seam_correction:
            # === Tile-grid brightness equalization ===
            # Run BEFORE seam corrections so it doesn't interfere with the
            # local seam repairs.  Only activates for channels with small
            # P90/P10 spread (near-noise-floor signal with tile artifacts).
            _equalize_tile_grid(slice_image, self.tiles,
                                noise_floors=self._noise_floors)

            # === Seam correction (vertical then horizontal) ===
            # Two destripe passes per axis: the first pass corrects the
            # bulk of the seam artifacts; the second pass catches
            # residuals that only become visible after the first
            # correction shifts neighbouring columns/rows.
            _DESTRIPE_PASSES = 2
            for axis in ('column', 'row'):
                _equalize_seam_steps(slice_image, self.tiles, axis=axis,
                                     noise_floors=self._noise_floors,
                                     seam_params=sp)
                for _ in range(_DESTRIPE_PASSES):
                    _destripe_seams(slice_image, self.tiles, axis=axis,
                                    noise_floors=self._noise_floors,
                                    seam_params=sp)

        # === Background zeroing ===
        if self._zero_background:
            _zero_background(slice_image, noise_floors=self._noise_floors)

        return slice_image, missing_tiles


    def stitch(self, slice_image, stitched_indicator, tile, cb=np.array):

        region = tile.get_image_region()

        current_region = slice_image[region[0], region[1], region[2]]
        indicator_region = stitched_indicator[region[0], region[1], region[2]]

        # Equalize tile intensity to already-stitched mosaic before blending
        ch_noise = (self._noise_floors[tile.channel]
                    if self._noise_floors is not None
                    and tile.channel < len(self._noise_floors)
                    else None)
        taper = self._seam_params['gain_taper_px'] if hasattr(self, '_seam_params') else 200
        tile.image = _overlap_gain_compensate(
            tile.image, current_region, indicator_region,
            noise_floor=ch_noise, taper_px=taper
        )

        stup = (tile.size['row'], tile.size['column'])
        if stup not in self._meshgrid_cache:
            self._meshgrid_cache[stup] = np.meshgrid(*map(np.arange, stup), indexing='ij')
        blend_mask = get_blend(indicator_region, stup, cb, meshes=self._meshgrid_cache[stup])

        if self.blend_mode == 'multiband':
            blended = multiband_blend(tile.image, current_region, blend_mask)
        else:
            blended = linear_blend(tile.image, current_region, blend_mask)

        slice_image[region[0], region[1], region[2]] = blended
        stitched_indicator[region[0], region[1], region[2]] = 1


# ============================================================================
# Overlap Gain Compensation
# ============================================================================
# After flat-field (vignetting) correction, adjacent tiles can still differ
# in overall brightness due to:
#   - residual vignetting correction error (gain map is uncertain at edges
#     where correction gains exceed 1.5×, amplifying small model errors)
#   - laser power drift between tile acquisitions
#   - PMT gain or detector fluctuations
#
# This creates visible seam steps at tile boundaries, especially where the
# overlap is narrow (e.g. 25 px horizontal).
#
# Overlap gain compensation measures the median intensity ratio in the
# overlap zone and applies a single scalar correction to the incoming
# tile so its level matches the already-stitched mosaic.  This is done
# BEFORE blending, so the multiband blend only needs to reconcile the
# remaining high-frequency texture differences.

_GAIN_COMP_NOISE = _BLEND_DTYPE(30.0)     # noise floor (~2× camera dark level)
_GAIN_COMP_MIN_PX = 100                   # min signal pixels for stable ratio
_GAIN_COMP_CLAMP = (0.70, 1.42)           # safety clamp on ratio

# ---- Fractional/ratio safety limits (hardware-independent) ----
_DESTRIPE_MAX_CORR = 0.20      # max fractional correction (±20%)
_STEP_MAX_CORR_HALF = 0.10     # max correction per side (10%)


def _estimate_tile_geometry(tiles):
    """Derive tile pitch, overlap width, and tile size from tile bounds.

    Analyses the tile spans along both axes to compute geometry-dependent
    parameters for seam correction.  This makes all pixel-dimension
    constants adaptive to the actual tile layout rather than hardcoded
    for a specific tile size.

    Returns
    -------
    dict with keys:
        'tile_pitch_col', 'tile_pitch_row' : float
            Distance between adjacent tile centres (start-to-start).
        'overlap_col', 'overlap_row' : float
            Overlap width in pixels.
        'tile_w', 'tile_h' : float
            Single tile extent (column and row axis).
    """
    if not tiles:
        return None
    first_ch = min(int(t.channel) for t in tiles)
    ch_tiles = [t for t in tiles if int(t.channel) == first_ch]

    result = {}
    for axis, key_size, key_pitch, key_ov in [
        ('column', 'tile_w', 'tile_pitch_col', 'overlap_col'),
        ('row',    'tile_h', 'tile_pitch_row', 'overlap_row'),
    ]:
        spans = sorted({
            (int(t.bounds[axis]['start']), int(t.bounds[axis]['end']))
            for t in ch_tiles
        })
        widths = [e - s for s, e in spans]
        tile_size = float(np.median(widths)) if widths else 800.0

        pitches = []
        overlaps = []
        for i in range(len(spans) - 1):
            ov = spans[i][1] - spans[i + 1][0]
            if 0 < ov < tile_size * 0.4:
                pitches.append(spans[i + 1][0] - spans[i][0])
                overlaps.append(ov)

        result[key_size] = tile_size
        result[key_pitch] = float(np.median(pitches)) if pitches else tile_size
        result[key_ov] = float(np.median(overlaps)) if overlaps else max(10.0, tile_size * 0.03)

    return result


def _derive_seam_params(tiles):
    """Compute geometry-dependent seam correction parameters from tiles.

    All pixel-dimension constants for destripe, step equalization, and
    gain compensation are derived as fractions of tile geometry, making
    them adaptive to any tile size and overlap width.

    Returns
    -------
    dict with all parameters needed by seam correction functions.
    """
    geom = _estimate_tile_geometry(tiles)
    if geom is None:
        geom = {'tile_w': 756.0, 'tile_h': 774.0,
                'tile_pitch_col': 731.0, 'tile_pitch_row': 731.0,
                'overlap_col': 25.0, 'overlap_row': 43.0}

    avg_overlap = (geom['overlap_col'] + geom['overlap_row']) / 2.0
    avg_tile = (geom['tile_w'] + geom['tile_h']) / 2.0
    avg_pitch = (geom['tile_pitch_col'] + geom['tile_pitch_row']) / 2.0

    return {
        # Gain comp: spatial taper ≈ 27% of tile pitch
        'gain_taper_px': int(round(avg_pitch * 0.274)),

        # Destripe: zone around boundary ≈ 1.2× average overlap
        'destripe_zone_half': max(10, int(round(avg_overlap * 1.2))),
        # Reference zone: starts just outside destripe zone
        'destripe_ref_near': max(15, int(round(avg_overlap * 1.33))),
        # Reference zone far edge ≈ 16% of tile size
        'destripe_ref_far': max(30, int(round(avg_tile * 0.157))),
        # Vertical band height ≈ 52% of tile size
        'destripe_band_h': max(50, int(round(avg_tile * 0.523))),
        'destripe_band_stride': max(25, int(round(avg_tile * 0.26))),
        'destripe_band_min_px': max(10, int(round(avg_overlap * 0.9))),
        'destripe_smooth_sigma': max(30.0, avg_tile * 0.262),

        # Step eq: near ref ≈ 1.18× overlap (was 40px for 34px overlap)
        'step_meas_near': max(15, int(round(avg_overlap * 1.18))),
        'step_meas_far': max(30, int(round(avg_tile * 0.157))),
        # Taper width ≈ 41% of tile pitch
        'step_taper_px': max(30, int(round(avg_pitch * 0.41))),
        # S-transition ≈ 4% of tile pitch
        'step_trans_px': max(5, int(round(avg_pitch * 0.041))),
        'step_band_h': max(50, int(round(avg_tile * 0.523))),
        'step_band_stride': max(25, int(round(avg_tile * 0.26))),
        'step_band_min_px': max(10, int(round(avg_overlap * 1.5))),
        'step_smooth_sigma': max(30.0, avg_tile * 0.392),
    }


def _overlap_gain_compensate(tile_image, current_region, indicator_region,
                             noise_floor=None, taper_px=200):
    """Scale tile signal near the overlap to match the existing mosaic.

    Computes the ratio of medians for signal pixels (well above the noise
    floor) in the overlap, then applies a **spatially tapered, signal-only**
    correction:

    * **Spatial taper** — the ratio is applied at full strength inside and
      near the overlap zone, fading linearly to 1.0 (no correction) over
      ``taper_px`` pixels into the tile interior.  This prevents the
      overlap-derived ratio from biasing the entire tile, which would
      shift the intensity of the exclusive zone and create a new seam at
      the *opposite* edge.

    * **Signal-only** — within the spatially tapered region, the ratio
      scales only the signal component above the noise floor.  Background
      / dark-current pixels remain unchanged regardless of the ratio.
      A smooth transition around the noise floor prevents hard edges
      between corrected and uncorrected regions.

    Parameters
    ----------
    tile_image : np.ndarray
        Incoming tile pixel data (float32 after flat-field correction).
    current_region : np.ndarray, float32
        Existing stitched mosaic data in the tile's footprint.
    indicator_region : np.ndarray, uint8
        Binary mask: 1 where existing data is present, 0 where new-only.
    noise_floor : float or None, optional
        Per-channel noise floor.  When provided, overrides the global
        ``_GAIN_COMP_NOISE`` constant.
    taper_px : int, optional
        Spatial taper width in pixels (default 200).  Derived from tile
        geometry by ``_derive_seam_params``.

    Returns
    -------
    np.ndarray
        Gain-compensated tile image.
    """
    overlap = indicator_region > 0
    if not overlap.any():
        return tile_image

    nf = _BLEND_DTYPE(noise_floor) if noise_floor is not None else _GAIN_COMP_NOISE
    # Adaptive minimum-signal threshold: signal must be well above the
    # channel's noise floor for a reliable ratio.  For the default
    # NF=30 this gives 35 (slightly below the original 50).  For low-
    # signal channels like CH2 (NF≈20) this gives 25, allowing tissue
    # at ~25–36 to be corrected.
    min_signal = _BLEND_DTYPE(float(nf) + 5.0)

    exist_vals = current_region[overlap]                        # float32
    new_vals = np.asarray(tile_image, dtype=_BLEND_DTYPE)[overlap]

    # Restrict to pixels with real signal (both sides must be well above noise)
    valid = (exist_vals > nf) & (new_vals > nf)
    if int(valid.sum()) < _GAIN_COMP_MIN_PX:
        return tile_image

    # Use the lower quartile (P5-P40) of tissue intensities to compute
    # the gain ratio.  This focuses on neuropil (the diffuse background
    # tissue) rather than bright soma cells.  Neuropil intensity reflects
    # the underlying illumination field, while soma intensities vary with
    # cell density — using the full IQR lets dense bright cells (e.g.
    # CA1 soma) bias the ratio when they cluster on one side of the
    # overlap, creating sharp vertical bands.
    ev = exist_vals[valid]
    nv = new_vals[valid]
    all_signal = np.concatenate([ev, nv])
    p05, p40 = np.percentile(all_signal, [5, 40])
    trim_mask = (ev >= p05) & (ev <= p40) & (nv >= p05) & (nv <= p40)
    if int(trim_mask.sum()) >= _GAIN_COMP_MIN_PX:
        med_exist = float(np.median(ev[trim_mask]))
        med_new   = float(np.median(nv[trim_mask]))
    else:
        med_exist = float(np.median(ev))
        med_new   = float(np.median(nv))

    # Both medians must be well above noise — otherwise we're comparing
    # amplified noise to tissue and the ratio is meaningless.
    if med_new < min_signal or med_exist < min_signal:
        return tile_image

    ratio = med_exist / med_new
    ratio = max(_GAIN_COMP_CLAMP[0], min(_GAIN_COMP_CLAMP[1], ratio))

    if abs(ratio - 1.0) < 0.005:       # skip trivial corrections (<0.5%)
        return tile_image

    # --- Spatial taper: full correction at/near overlap, fading into interior ---
    # Distance from each pixel to the nearest overlap pixel.  Pixels inside
    # the overlap have dist=0; pixels in the exclusive zone have dist>0
    # increasing toward the tile center / opposite edge.
    inv_ov = np.ones_like(indicator_region, dtype=np.uint8)
    inv_ov[overlap] = 0
    dist = cv2.distanceTransform(inv_ov, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    spatial = np.clip(
        _BLEND_DTYPE(1.0) - dist.astype(_BLEND_DTYPE) / _BLEND_DTYPE(taper_px),
        _BLEND_DTYPE(0.0), _BLEND_DTYPE(1.0),
    )
    del inv_ov, dist

    # --- Signal-only scaling with smooth taper around the noise floor ---
    # Use a narrower transition so low-signal tissue (e.g. CH2 at
    # ~28 counts, NF≈20) still receives most of the correction.
    # Transition zone: NF → NF + NF*0.5, so tissue at NF+8 gets
    # t≈1.0 instead of the old t≈0.2 from a 2×NF ramp.
    tile_f = np.asarray(tile_image, dtype=_BLEND_DTYPE)
    t = np.clip(
        (tile_f - nf) / max(nf * _BLEND_DTYPE(0.5), _BLEND_DTYPE(5.0)),
        _BLEND_DTYPE(0.0), _BLEND_DTYPE(1.0),
    )
    delta = _BLEND_DTYPE(ratio - 1.0)
    return tile_f * (_BLEND_DTYPE(1.0) + t * spatial * delta)


# ============================================================================
# Post-stitch Tile-Grid Brightness Equalization
# ============================================================================
# On low-signal channels (e.g. CH2 with tissue at ~25–35 counts), the
# overlap-based gain compensation cannot fully equalise tile-level
# brightness differences because:
#   1. The signal-only taper attenuates corrections for dim tissue
#      (t = (val−NF)/(2·NF) ≈ 0.2 for CH2)
#   2. The spatial taper limits corrections to the overlap zone
#      (200 px), while tile centres (~378 px away) get zero correction
#   3. Chain-based matching (each tile to its neighbour) accumulates
#      drift when tile-level variation is large (CV ≈ 10%)
#
# This post-stitch step takes a GLOBAL approach: measure the median
# tissue brightness in every tile-sized region, compute each region's
# ratio to the global median, and apply a smoothly interpolated
# correction map.  This avoids chain drift and provides full-tile
# coverage.  Only activates when tile-level CV exceeds a threshold.

_TILE_EQ_MAX_SPREAD = 2.0    # max P90/P10 ratio to trigger equalization
_TILE_EQ_MIN_TISSUE_PX = 200 # minimum tissue pixels per tile region
_TILE_EQ_CLAMP = (0.70, 1.42) # safety clamp on correction ratios
_TILE_EQ_MEAS_BUFFER = 5.0   # extra margin above noise floor for measurement


def _equalize_tile_grid(mosaic, tiles, noise_floors=None):
    """Equalize tile-level brightness variations in the stitched mosaic.

    Divides the mosaic into tile-sized regions (defined by tile boundary
    positions), measures the median tissue brightness in each region, and
    applies a smooth correction map to bring all tiles to the same level.

    Only activates when the P90/P10 ratio of tile medians is below
    ``_TILE_EQ_MAX_SPREAD`` — this ensures channels with large
    biological variation (bright tissue vs background) are left
    untouched, while channels with subtle tile-level artifacts
    (near-noise-floor signal) are corrected.

    Parameters
    ----------
    mosaic : np.ndarray, shape (H, W, C), dtype float32
        Stitched multi-channel image (modified **in-place**).
    tiles : list of Tile
        Tile objects with position/bounds information.
    noise_floors : list of float or None
        Per-channel noise floor values.  Defaults to ``_GAIN_COMP_NOISE``.
    """
    from scipy.ndimage import gaussian_filter

    H, W = mosaic.shape[:2]
    n_ch = mosaic.shape[2] if mosaic.ndim >= 3 else 1
    _default_noise = float(_GAIN_COMP_NOISE)

    # Collect column and row tile boundaries
    col_bnds = _collect_boundaries(tiles, axis='column')
    row_bnds = _collect_boundaries(tiles, axis='row')
    if not col_bnds or not row_bnds:
        return

    col_edges = [0] + col_bnds + [W]
    row_edges = [0] + row_bnds + [H]
    n_rows = len(row_edges) - 1
    n_cols = len(col_edges) - 1

    for ch in range(n_ch):
        if mosaic.ndim >= 3:
            channel = mosaic[:, :, ch]
        else:
            channel = mosaic

        nf = (noise_floors[ch] if noise_floors is not None
              and ch < len(noise_floors) else _default_noise)
        nf = float(nf)
        # Use a higher threshold for *measuring* tissue medians so
        # that background scatter (noise_floor < val < noise_floor+5)
        # doesn't contaminate the medians — critical for low-signal
        # channels like CH2 where tissue is only ~5 counts above NF.
        nf_meas = nf + _TILE_EQ_MEAS_BUFFER

        # --- Fast spread gate: subsample 1/64 of pixels to detect
        # biological variation early, skipping 154-tile median loop
        # for channels that won't be equalized anyway (saves ~75%
        # of per-section overhead). ---
        # Use P25/P75 (interquartile range) instead of P10/P90 so that
        # sparse bright cells (e.g. CA1 somata) do not inflate the
        # spread and falsely disable equalization on high-signal
        # channels.  P75/P25 > 5.0 allows channels with moderate
        # biological variation (bright CA1 soma mixed with dim neuropil)
        # to still receive tile-level correction, while genuine
        # large-scale variation (e.g. autofluorescence) is excluded.
        sub = channel[::8, ::8]
        tissue_sub = sub[sub > nf_meas]
        if len(tissue_sub) < 100:
            continue
        sp25, sp75 = np.percentile(tissue_sub, [25, 75])
        if sp25 <= 0 or sp75 / sp25 > 5.0:
            continue

        # --- Measure median tissue brightness per tile region ---
        medians = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
        for ri in range(n_rows):
            r0, r1 = row_edges[ri], row_edges[ri + 1]
            for ci in range(n_cols):
                c0, c1 = col_edges[ci], col_edges[ci + 1]
                region = channel[r0:r1, c0:c1]
                tile_area = (r1 - r0) * (c1 - c0)
                min_px = max(_TILE_EQ_MIN_TISSUE_PX,
                             int(tile_area * 0.05))
                tissue = region[region > nf_meas]
                if len(tissue) >= min_px:
                    # Use 10th-30th percentile trimmed mean to measure
                    # the neuropil (background tissue) brightness per
                    # tile.  Focusing on the lower quartile avoids bias
                    # from bright soma cells (e.g. CA1) that differ in
                    # density across tiles — we want to equalize the
                    # underlying illumination field, not biological
                    # signal.
                    p10, p30 = np.percentile(tissue, [10, 30])
                    trimmed = tissue[(tissue >= p10) & (tissue <= p30)]
                    if len(trimmed) > 0:
                        medians[ri, ci] = float(np.mean(trimmed))
                    else:
                        medians[ri, ci] = float(np.median(tissue))

        valid_meds = medians[~np.isnan(medians)]
        if len(valid_meds) < 4:
            continue

        # --- Check whether equalization is needed ---
        # Use the tile-median spread to decide.  Even after trimmed-
        # mean measurement, tiles with dense bright cells may still
        # show some spread.  Allow up to 2.5× P90/P10 ratio (was 2.0)
        # so that moderate tile-level gain errors still get corrected.
        p10, p90 = np.percentile(valid_meds, [10, 90])
        if p10 <= 0 or p90 / p10 > 2.5:
            continue  # too much biological variation — don't equalize

        target = float(np.median(valid_meds))

        # --- Build flat correction ratios per tile ---
        ratios = np.full((n_rows, n_cols), np.nan, dtype=np.float32)
        for ri in range(n_rows):
            for ci in range(n_cols):
                m = medians[ri, ci]
                if np.isfinite(m) and m > nf_meas:
                    ratios[ri, ci] = target / m
        np.clip(ratios, _TILE_EQ_CLAMP[0], _TILE_EQ_CLAMP[1],
                out=ratios)

        # Leave outermost tile rows/columns uncorrected — edge
        # tiles have unreliable medians (partial tissue, artifacts).
        ratios[0, :] = np.nan
        ratios[-1, :] = np.nan
        ratios[:, 0] = np.nan
        ratios[:, -1] = np.nan

        # Fill invalid positions with nearest valid ratio.
        from scipy.ndimage import distance_transform_edt
        valid_r = ~np.isnan(ratios)
        if valid_r.any():
            _, nearest_idx = distance_transform_edt(
                ~valid_r, return_distances=True,
                return_indices=True)
            ratios[:] = ratios[tuple(nearest_idx)]
        else:
            ratios[:] = 1.0

        # --- Smooth the coarse ratio grid ---
        smooth = gaussian_filter(
            ratios.astype(np.float64), sigma=0.3
        ).astype(np.float32)
        np.clip(smooth, _TILE_EQ_CLAMP[0], _TILE_EQ_CLAMP[1],
                out=smooth)

        # --- Build per-pixel correction map ---
        h, w = channel.shape
        corr_map = np.ones((h, w), dtype=np.float32)
        for ri in range(n_rows):
            r0, r1 = row_edges[ri], row_edges[ri + 1]
            for ci in range(n_cols):
                c0, c1 = col_edges[ci], col_edges[ci + 1]
                corr_map[r0:r1, c0:c1] = smooth[ri, ci]
        # Apply only to tissue pixels (above noise floor) so that
        # background pixels are not shifted by inter-tile corrections.
        if np.max(np.abs(corr_map - 1.0)) > 0.003:
            nf32 = np.float32(nf)
            sig = channel > nf32
            channel[sig] = np.clip(
                channel[sig] * corr_map[sig], 0, 65535)
        del corr_map


# ============================================================================
# Post-stitch Seam Equalization
# ============================================================================
# After stitching, systematic vignetting residuals create ~50 px dimming
# zones at tile edges.  The flat-field gain map cannot perfectly capture
# per-tile vignetting variation, and the spatial-taper gain compensation
# can overcorrect nearby columns, leaving dark dips AND bright peaks at
# boundary positions.
#
# This function measures column profiles in **row bands** (not the whole
# section at once) so the correction adapts to local tissue brightness.
# Both dips and peaks that deviate from the linearly-interpolated
# reference are corrected.  The per-band corrections are linearly
# interpolated vertically to avoid introducing new horizontal artefacts.
#
# Memory-efficient: only a (H × zone_width) slice is allocated per
# boundary (~160 columns ≈ 7 MB), not the full (H × W) field.


# ============================================================================
# Background Zeroing
# ============================================================================

def _zero_background(mosaic, noise_floors=None):
    """Zero out background pixels while preserving dark brain regions.

    Builds a per-channel brain mask by thresholding above the noise floor,
    removing noise with morphological opening, keeping only large connected
    components (the actual brain), then closing + hole-filling to preserve
    dark interior structures (ventricles, fiber tracts).  Only pixels
    **outside** the final brain mask are zeroed.

    Parameters
    ----------
    mosaic : np.ndarray, shape (H, W, C), dtype float32
        Stitched image (modified **in-place**).
    noise_floors : list[float] | None
        Per-channel noise floor values.  Falls back to ``_GAIN_COMP_NOISE``.
    """
    from scipy.ndimage import (binary_fill_holes, binary_closing,
                                binary_opening, label)

    n_ch = mosaic.shape[2] if mosaic.ndim >= 3 else 1
    _default_noise = float(_GAIN_COMP_NOISE)

    if noise_floors is None:
        noise_floors = [_default_noise] * n_ch

    # Work at reduced resolution for the morphological ops to save memory
    # and computation.  A 4× downsample is fine for whole-brain masks.
    _DS = 4
    H, W = mosaic.shape[:2]

    for ch in range(n_ch):
        nf = (noise_floors[ch] if ch < len(noise_floors)
              else _default_noise)
        nf = float(nf)

        if mosaic.ndim >= 3:
            channel = mosaic[:, :, ch]
        else:
            channel = mosaic

        # 1. Downsample for fast morphology
        small = channel[::_DS, ::_DS]
        h_ds, w_ds = small.shape[:2]
        tissue_mask = small > nf  # bool (h_ds, w_ds)

        # 2. Morphological OPEN to remove isolated noise pixels in the
        #    background.  Small kernel — just enough to clean up noise
        #    without eroding real tissue edges.
        kern_open = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (5, 5)).astype(bool)
        opened = binary_opening(tissue_mask, structure=kern_open)

        # 3. Connected component analysis — keep only components larger
        #    than 0.5% of the downsampled image area (i.e. actual brain
        #    sections, not stray noise clusters).
        labeled, n_features = label(opened)
        min_area = h_ds * w_ds * 0.005
        brain_mask = np.zeros_like(opened)
        for i in range(1, n_features + 1):
            comp = labeled == i
            if comp.sum() > min_area:
                brain_mask |= comp

        # 4. Morphological CLOSE to smooth ragged edges and bridge
        #    small gaps at tile boundaries.
        kern_r = 15
        kern_close = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * kern_r + 1, 2 * kern_r + 1)
        ).astype(bool)
        closed = binary_closing(brain_mask, structure=kern_close)

        # 5. Fill holes — ventricles, fiber tracts, any enclosed dark
        #    region is preserved regardless of size.
        filled = binary_fill_holes(closed)

        # 6. Upscale the mask back to full resolution via nearest-neighbor
        mask_full = cv2.resize(
            filled.astype(np.uint8), (W, H),
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        # 7. Zero only pixels outside the brain mask
        channel[~mask_full] = 0.0

    logging.info('Background zeroing applied (morphological brain mask).')


def _collect_boundaries(tiles, axis='column'):
    """Return sorted overlap-centre positions along the given axis.

    Only returns boundaries where the overlap between adjacent tile
    spans is *narrow* (≤ 35% of the median tile size).  This filters
    out spurious within-column overlaps caused by ``txy`` serpentine
    offsets (which produce huge overlaps ≈ tile width) and keeps only
    the real inter-column overlaps (typically ~25 px wide).

    Clusters of nearby centres (within half the median tile size) are
    merged to a single position at the cluster median.
    """
    if not tiles:
        return []
    first_ch = min(int(t.channel) for t in tiles)
    spans = sorted({
        (int(t.bounds[axis]['start']), int(t.bounds[axis]['end']))
        for t in tiles if int(t.channel) == first_ch
    })

    # Only keep overlaps that are narrow (real inter-column seams).
    # Within-column tile overlaps span almost the full tile width
    # and are not visual seam lines.  Threshold adapts to tile size.
    widths = [e - s for s, e in spans]
    median_tile_size = float(np.median(widths)) if widths else 800.0
    max_overlap_width = median_tile_size * 0.35
    raw_centres = []
    for i in range(len(spans) - 1):
        overlap = spans[i][1] - spans[i + 1][0]
        if 0 < overlap <= max_overlap_width:
            raw_centres.append((spans[i + 1][0] + spans[i][1]) // 2)

    if not raw_centres:
        return []

    # Merge nearby centres into single boundaries.
    # Any centres within merge_gap of each other are likely
    # the same physical tile-column boundary shifted by offsets.
    merge_gap = max(20, int(median_tile_size * 0.13))
    merged = []
    cluster = [raw_centres[0]]
    for c in raw_centres[1:]:
        if c - cluster[-1] <= merge_gap:
            cluster.append(c)
        else:
            merged.append(int(np.median(cluster)))
            cluster = [c]
    merged.append(int(np.median(cluster)))
    return merged


def _destripe_seams(mosaic, tiles, axis='column', noise_floors=None,
                    seam_params=None):
    """Row-adaptive correction of narrow artefacts at tile boundaries.

    After step equalization has removed the wide brightness step between
    adjacent tile columns (or rows), a narrow dip or peak (10–20 px
    wide) often remains right at the tile boundary.  This function
    corrects both dips AND peaks using per-band measurements so the
    correction adapts to local tissue brightness at every position.

    When ``axis='row'``, the mosaic is internally transposed so that
    horizontal boundaries are processed with the same column-based
    logic, then the corrections propagate back via the transposed view.

    Parameters
    ----------
    mosaic : np.ndarray, shape (H, W, C), dtype float32
        Stitched mosaic image.  Modified **in-place**.
    tiles : list of Tile
        Tile objects (image data may already be freed).
    axis : str, optional
        ``'column'`` for vertical boundaries (default), ``'row'`` for
        horizontal boundaries.
    noise_floors : list of float or None, optional
        Per-channel noise floor values.
    seam_params : dict or None, optional
        Geometry-derived parameters from ``_derive_seam_params``.
    """
    from scipy.ndimage import gaussian_filter1d

    # Extract parameters (use seam_params if provided, else compute)
    if seam_params is None:
        seam_params = _derive_seam_params(tiles)
    zone_half = seam_params['destripe_zone_half']
    ref_near = seam_params['destripe_ref_near']
    ref_far = seam_params['destripe_ref_far']
    band_h = seam_params['destripe_band_h']
    band_stride = seam_params['destripe_band_stride']
    band_min_px = seam_params['destripe_band_min_px']
    smooth_sigma = seam_params['destripe_smooth_sigma']

    # For row boundaries, transpose so the same column logic works.
    # numpy transpose returns a *view*, so in-place edits propagate.
    if axis == 'row':
        work = mosaic.transpose(1, 0, 2)            # (W, H, C)
    else:
        work = mosaic                                # (H, W, C)

    h, w, n_ch = work.shape
    _default_noise = float(_GAIN_COMP_NOISE)

    bndry = _collect_boundaries(tiles, axis=axis)
    if not bndry:
        return

    # Cosine-bell taper: 1 at centre, 0 at ±zone edges
    zone_full = 2 * zone_half
    taper_1d = np.cos(np.linspace(-np.pi / 2, np.pi / 2, zone_full))
    taper_1d = (taper_1d ** 2).astype(np.float64)

    for ch in range(n_ch):
        channel = work[:, :, ch]
        noise_floor = (noise_floors[ch] if noise_floors is not None
                       and ch < len(noise_floors) else _default_noise)

        for bc in bndry:
            zL = max(0, bc - zone_half)
            zR = min(w, bc + zone_half)
            zone_w = zR - zL
            if zone_w < 10:
                continue

            taper = taper_1d[:zone_w] if zone_w < zone_full else taper_1d

            rL_lo = max(0, bc - ref_far)
            rL_hi = max(0, bc - ref_near)
            rR_lo = min(w, bc + ref_near)
            rR_hi = min(w, bc + ref_far)
            if (rL_hi - rL_lo) < 5 or (rR_hi - rR_lo) < 5:
                continue

            band_centres = []
            band_ratios = []

            for y0 in range(0, h, band_stride):
                y1 = min(y0 + band_h, h)
                band = channel[y0:y1, :]

                left_ref = band[:, rL_lo:rL_hi]
                right_ref = band[:, rR_lo:rR_hi]
                lt_mask = left_ref > noise_floor
                rt_mask = right_ref > noise_floor

                lt_cnt = int(lt_mask.sum())
                rt_cnt = int(rt_mask.sum())

                if lt_cnt < band_min_px * 5 or rt_cnt < band_min_px * 5:
                    band_centres.append((y0 + y1) / 2.0)
                    band_ratios.append(np.ones(zone_w, dtype=np.float64))
                    continue

                # Use lower-quartile (P25) of tissue pixels to measure the
                # illumination field, avoiding bias from bright soma cells
                # (e.g. CA1) that may differ in density across reference
                # zones.  Neuropil intensity is a better proxy for the
                # uniform illumination field that seam correction targets.
                lt_vals = left_ref[lt_mask]
                rt_vals = right_ref[rt_mask]
                med_L = float(np.percentile(lt_vals, 25))
                med_R = float(np.percentile(rt_vals, 25))

                if med_L < noise_floor or med_R < noise_floor:
                    band_centres.append((y0 + y1) / 2.0)
                    band_ratios.append(np.ones(zone_w, dtype=np.float64))
                    continue

                # Expected baseline across zone (linear interpolation)
                fracs = np.linspace(0.0, 1.0, zone_w, dtype=np.float64)
                baseline = med_L * (1.0 - fracs) + med_R * fracs

                # Actual per-column tissue-P25 in the zone (vectorized).
                # Using P25 (lower quartile) instead of median to focus on
                # neuropil rather than bright soma cells, consistent with
                # the reference zone measurement above.
                zone_band = band[:, zL:zR]
                tissue_mask = zone_band > noise_floor
                col_counts = tissue_mask.sum(axis=0)       # (zone_w,)
                zone_f = zone_band.astype(np.float64)
                zone_f[~tissue_mask] = np.nan
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', RuntimeWarning)
                    actuals = np.nanpercentile(zone_f, 25, axis=0)  # (zone_w,)

                ratio_vec = np.ones(zone_w, dtype=np.float64)
                valid_cols = ((col_counts >= band_min_px)
                              & (actuals > noise_floor))
                ratio_vec[valid_cols] = (baseline[valid_cols]
                                         / np.maximum(actuals[valid_cols],
                                                      1.0))

                # Clamp to safety range
                ratio_vec = np.clip(ratio_vec,
                                    1.0 - _DESTRIPE_MAX_CORR,
                                    1.0 + _DESTRIPE_MAX_CORR)

                band_centres.append((y0 + y1) / 2.0)
                band_ratios.append(ratio_vec)

            if not band_ratios:
                continue

            # --- Smooth band ratios, then interpolate to per-row ---
            centres = np.array(band_centres)
            ratios_arr = np.array(band_ratios)          # (n_bands, zone_w)

            # Smooth at band level (57 pts, sigma≈1) instead of full
            # resolution (11K pts, sigma=200).  Same spatial scale
            # because band_stride ≈ smooth_sigma.
            band_sigma = smooth_sigma / max(band_stride, 1)
            if ratios_arr.shape[0] > 2:
                ratios_arr = gaussian_filter1d(
                    ratios_arr, sigma=band_sigma, axis=0)

            np.clip(ratios_arr, 1.0 - _DESTRIPE_MAX_CORR,
                    1.0 + _DESTRIPE_MAX_CORR, out=ratios_arr)

            all_rows = np.arange(h, dtype=np.float64)
            corr_2d = np.ones((h, zone_w), dtype=np.float64)
            for ci in range(zone_w):
                corr_2d[:, ci] = np.interp(all_rows, centres,
                                           ratios_arr[:, ci])

            # Light post-interp smooth to remove interpolation artifacts
            # (~15% of original sigma — 6× cheaper than full-resolution
            # smoothing but eliminates band-transition edges).
            post_sigma = smooth_sigma * 0.15
            if post_sigma > 5:
                corr_2d = gaussian_filter1d(
                    corr_2d, sigma=post_sigma, axis=0)
                np.clip(corr_2d, 1.0 - _DESTRIPE_MAX_CORR,
                        1.0 + _DESTRIPE_MAX_CORR, out=corr_2d)

            # Skip if negligible
            if np.max(np.abs(corr_2d - 1.0)) < 0.003:
                continue

            # Apply bell taper at zone edges
            corr_2d = 1.0 + (corr_2d - 1.0) * taper[np.newaxis, :]

            channel[:, zL:zR] *= corr_2d.astype(_BLEND_DTYPE)


# ============================================================================
# Post-stitch Seam Step Equalization  (row-adaptive)
# ============================================================================
# After gain-comp + destripe, tile column boundaries still exhibit a
# brightness *step*.  The step magnitude varies with vertical position
# (y) — e.g., +5 % in one row band, −10 % in another — so a single
# global correction is insufficient.
#
# This function divides the image into overlapping horizontal bands,
# measures the local step at each boundary in each band, vertically
# smooths the resulting step profile with a Gaussian, and builds a
# 2-D correction map (H × zone_w) for each boundary.
#
# Horizontal shape:  3-phase cosine S-curve (identical to the original
#   global approach, just applied per-row with varying amplitude δ(y)):
#
#   Phase 1 (far left → near boundary):  0 → +δ   (cosine ramp up)
#   Phase 2 (across boundary):           +δ → −δ  (cosine S transition)
#   Phase 3 (near boundary → far right): −δ → 0   (cosine ramp down)
#
# Vertical shape: δ(y) is the locally-measured (and smoothed) half-step.


def _equalize_seam_steps(mosaic, tiles, axis='column', noise_floors=None,
                         seam_params=None):
    """Row-adaptive step equalization at tile boundaries.

    Measures the brightness step at each boundary in overlapping
    horizontal bands, smooths vertically, then applies a 2-D
    correction that adapts to the local step at every height.

    When ``axis='row'``, the mosaic is internally transposed so that
    horizontal boundaries are processed with the same column-based
    logic, then the corrections propagate back via the transposed view.

    Parameters
    ----------
    mosaic : np.ndarray, shape (H, W, C), dtype float32
        Stitched mosaic image.  Modified **in-place**.
    tiles : list of Tile
        Tile objects (image data may already be freed).
    axis : str, optional
        ``'column'`` for vertical boundaries (default), ``'row'`` for
        horizontal boundaries.
    noise_floors : list of float or None, optional
        Per-channel noise floor values.
    seam_params : dict or None, optional
        Geometry-derived parameters from ``_derive_seam_params``.
    """
    from scipy.ndimage import gaussian_filter1d

    if seam_params is None:
        seam_params = _derive_seam_params(tiles)
    meas_near = seam_params['step_meas_near']
    meas_far = seam_params['step_meas_far']
    taper_px = seam_params['step_taper_px']
    trans_px = seam_params['step_trans_px']
    s_band_h = seam_params['step_band_h']
    s_band_stride = seam_params['step_band_stride']
    s_band_min_px = seam_params['step_band_min_px']
    s_smooth_sigma = seam_params['step_smooth_sigma']

    # For row boundaries, transpose so the same column logic works.
    if axis == 'row':
        work = mosaic.transpose(1, 0, 2)
    else:
        work = mosaic

    h, w, n_ch = work.shape
    _default_noise = float(_GAIN_COMP_NOISE)

    bndry = _collect_boundaries(tiles, axis=axis)
    if not bndry:
        return

    T = trans_px / taper_px if taper_px > 0 else 0.1

    for ch in range(n_ch):
        channel = work[:, :, ch]
        noise_floor = (noise_floors[ch] if noise_floors is not None
                       and ch < len(noise_floors) else _default_noise)

        for bc in bndry:
            zL = max(0, bc - taper_px)
            zR = min(w, bc + taper_px)
            zone_w = zR - zL
            if zone_w < 20:
                continue

            xs = np.arange(zone_w, dtype=np.float64)
            pos = np.clip((xs + zL - bc) / taper_px, -1.0, 1.0)

            shape_curve = np.empty(zone_w, dtype=np.float64)
            m1 = pos < -T
            f1 = (pos[m1] + 1.0) / (1.0 - T)
            shape_curve[m1] = 0.5 * (1.0 - np.cos(np.pi * f1))
            m2 = (~m1) & (pos <= T)
            f2 = (pos[m2] + T) / (2.0 * T)
            shape_curve[m2] = np.cos(np.pi * f2)
            m3 = pos > T
            f3 = (pos[m3] - T) / (1.0 - T)
            shape_curve[m3] = -0.5 * (1.0 + np.cos(np.pi * f3))

            band_centres = []
            band_steps = []

            for y0 in range(0, h, s_band_stride):
                y1 = min(y0 + s_band_h, h)
                band = channel[y0:y1, :]

                left_strip = band[:, max(0, bc - meas_far):
                                     max(0, bc - meas_near)]
                right_strip = band[:, min(w, bc + meas_near):
                                      min(w, bc + meas_far)]

                left_tissue_mask = left_strip > noise_floor
                right_tissue_mask = right_strip > noise_floor

                n_left = left_tissue_mask.sum()
                n_right = right_tissue_mask.sum()

                if n_left < s_band_min_px or n_right < s_band_min_px:
                    band_centres.append((y0 + y1) / 2.0)
                    band_steps.append(0.0)
                    continue

                # Use P25 of tissue pixels to measure the illumination
                # step, avoiding bias from bright soma cells (CA1) that
                # differ in density between left and right reference zones.
                med_L = float(np.percentile(left_strip[left_tissue_mask], 25))
                med_R = float(np.percentile(right_strip[right_tissue_mask], 25))

                avg = (med_L + med_R) / 2.0
                if avg < 1.0:
                    band_centres.append((y0 + y1) / 2.0)
                    band_steps.append(0.0)
                    continue

                step_frac = (med_R - med_L) / avg
                step_frac = max(-0.20, min(0.20, step_frac))

                band_centres.append((y0 + y1) / 2.0)
                band_steps.append(step_frac)

            if not band_steps:
                continue

            centres = np.array(band_centres)
            steps = np.array(band_steps)

            step_per_row = np.interp(np.arange(h, dtype=np.float64),
                                     centres, steps)

            step_per_row = gaussian_filter1d(step_per_row,
                                             sigma=s_smooth_sigma)

            corr_half = np.clip(step_per_row / 2.0,
                                -_STEP_MAX_CORR_HALF,
                                _STEP_MAX_CORR_HALF)

            if np.max(np.abs(corr_half)) < 0.001:
                continue

            corr_map = corr_half[:, np.newaxis] * shape_curve[np.newaxis, :]

            channel[:, zL:zR] *= (
                _BLEND_DTYPE(1.0) + corr_map.astype(_BLEND_DTYPE)
            )


# ============================================================================
# Multi-band Laplacian Pyramid Blending
# ============================================================================
# Decomposes images into frequency bands via Laplacian pyramids and blends
# each band with a progressively smoothed mask. Low frequencies are blended
# over wide transitions (eliminating intensity seams) while high frequencies
# are blended over narrow transitions (preserving detail).

NUM_PYRAMID_LEVELS = 5  # number of pyramid levels (adjustable)


def _gaussian_pyramid(img, levels):
    """Build a Gaussian pyramid by successive downsampling."""
    pyramid = [img.astype(_BLEND_DTYPE)]
    for _ in range(levels):
        img = cv2.pyrDown(pyramid[-1])
        pyramid.append(img)
    return pyramid


def _laplacian_pyramid(img, levels):
    """Build a Laplacian pyramid from an image.

    Memory note: the Gaussian pyramid is discarded level-by-level as
    Laplacian differences are computed, so peak usage is ~1.5x one
    full-resolution image rather than 2x.
    """
    gauss = _gaussian_pyramid(img, levels)
    lap = []
    for i in range(levels):
        h, w = gauss[i].shape[:2]
        upsampled = cv2.pyrUp(gauss[i + 1], dstsize=(w, h))
        lap.append(gauss[i] - upsampled)
        gauss[i] = None  # free immediately
    lap.append(gauss[-1])  # coarsest level is the residual
    return lap


def _reconstruct_from_laplacian(lap_pyramid):
    """Reconstruct an image from its Laplacian pyramid.

    Frees each level after it has been consumed to keep peak memory low.
    """
    img = lap_pyramid[-1]
    for i in range(len(lap_pyramid) - 2, -1, -1):
        h, w = lap_pyramid[i].shape[:2]
        img = cv2.pyrUp(img, dstsize=(w, h))
        img += lap_pyramid[i]
        lap_pyramid[i] = None  # free consumed level
    return img


def linear_blend(new_tile, existing_region, blend_mask):
    """Simple linear blend: result = (1 - mask) * new_tile + mask * existing.

    This is the original blending method. Fast but can produce visible seams
    at tile boundaries.  Operates in float32 to save memory.
    """
    mask = blend_mask.astype(_BLEND_DTYPE)
    a = existing_region.astype(_BLEND_DTYPE)
    b = new_tile.astype(_BLEND_DTYPE)
    # In-place operations to avoid temporaries
    np.multiply(mask, a, out=a)
    inv_mask = np.subtract(np.float32(1.0), mask)
    np.multiply(inv_mask, b, out=b)
    b += a
    return b


def multiband_blend(new_tile, existing_region, blend_mask):
    """Blend new_tile onto existing_region using multi-band Laplacian pyramid blending.

    blend_mask: float array in [0, 1] where 1 = keep existing, 0 = use new tile.
    This matches the convention of the original make_blended_tile:
        result = (1 - blend) * new_tile + blend * existing

    If there is no overlap (blend_mask is all zeros), just returns the new tile.
    Uses float32 throughout to halve peak memory vs float64.
    """
    overlap = blend_mask.max() > 0
    if not overlap:
        return new_tile.astype(_BLEND_DTYPE)

    # blend_mask is the weight for existing_region; (1 - blend_mask) is weight for new_tile
    # We treat existing_region as A (mask=1) and new_tile as B (mask=0)
    mask = blend_mask.astype(_BLEND_DTYPE)
    a = existing_region.astype(_BLEND_DTYPE)
    b = new_tile.astype(_BLEND_DTYPE)

    h, w = a.shape[:2]

    # Need minimum size for pyramid (at least 2^levels pixels in each dimension)
    min_dim = min(h, w)
    levels = min(NUM_PYRAMID_LEVELS, max(1, int(np.log2(max(min_dim, 1))) - 1))

    # Also cap levels so the overlap zone has ≥2 pixels at the coarsest level.
    # Without this, narrow overlaps (e.g. 25 px) produce a hard step at the
    # lowest frequency band because the mask is fully downsampled away.
    mask_active = mask > 0.01
    if mask_active.any():
        ov_cols = int(mask_active.any(axis=0).sum())
        ov_rows = int(mask_active.any(axis=1).sum())
        min_ov = min(ov_cols, ov_rows) if min(ov_cols, ov_rows) > 0 else max(ov_cols, ov_rows)
        if 0 < min_ov < min_dim:
            levels = min(levels, max(1, int(np.log2(max(min_ov, 1))) - 1))

    if levels < 2:
        # Too small for meaningful pyramid — fall back to linear blend
        return linear_blend(new_tile, existing_region, blend_mask)

    # Build Laplacian pyramids for both images
    lap_a = _laplacian_pyramid(a, levels)
    del a
    lap_b = _laplacian_pyramid(b, levels)
    del b

    # Build Gaussian pyramid for the mask
    mask_gauss = _gaussian_pyramid(mask, levels)
    del mask

    # Blend each level in-place into lap_a to avoid a third pyramid allocation
    for i, (la, lb, gm) in enumerate(zip(lap_a, lap_b, mask_gauss)):
        inv_gm = np.subtract(np.float32(1.0), gm)
        np.multiply(gm, la, out=la)
        np.multiply(inv_gm, lb, out=lb)
        la += lb
        lap_a[i] = la
        lap_b[i] = None  # free immediately
        mask_gauss[i] = None
    del lap_b, mask_gauss

    # Reconstruct from blended pyramid
    result = _reconstruct_from_laplacian(lap_a)

    # Clip to valid range (uint16 data)
    np.clip(result, 0, 65535, out=result)

    return result


def get_indicator_bound_point(indicator, lg, axis):
    '''Finds the index of first change in a binary mask
    along a specified axis in a specified direction
    '''

    delta = np.diff(indicator, axis=axis)
    points = np.where(lg(delta, 0))
    del delta

    points = np.unique(points[axis])
    size = indicator.shape[axis]
    points = points[lg(points, size / 2.0)]

    if len(points) > 0:
        return points[-1]
    return None


def blend_component_from_point(point, mesh, lg):
    '''Obtains a normalized component of the blend, which describes depth of
    overlap along a specified axis in a specified direction
    '''

    # this has the effect that the shallowest part of the blend
    # is always 0 - symmetric with the deepest after normalization.
    blend = point - mesh + 1
    blend[lg(blend, 0)] = 0

    blend = np.fabs(blend)
    mx = np.amax(blend)
    mx = mx if mx > 0.0 else 1.0

    return blend / mx


def get_blend_component(indicator, lg, axis, meshes):
    '''
    '''

    point = get_indicator_bound_point(indicator, lg, axis)
    if point is None:
        return []

    return [blend_component_from_point(point, meshes[axis], lg)]


def get_overall_blend(indicator, meshes):
    '''
    '''

    blends = []

    for lg in (op.lt, op.gt):
        for axis in (0, 1):
            blends.extend(get_blend_component(indicator, lg, axis, meshes))

    if len(blends) == 0:
        return np.zeros_like(indicator)
    return reduce(np.maximum, blends)


def get_blend(indicator_region, stup, cb=np.array, meshes=None):
    '''
    '''

    if meshes is None:
        meshes = np.meshgrid(*map(np.arange, stup), indexing='ij')
    blend = get_overall_blend(indicator_region, meshes)

    return cb(np.multiply(blend, indicator_region))
