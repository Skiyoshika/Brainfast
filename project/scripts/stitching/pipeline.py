# ruff: noqa
"""Vendored from UCI-XuLab-RegTools (github.com/UCI-XuLab/UCI-XuLab-RegTools).

Vendored into Brainfast on 2026-04-23 as
project/scripts/stitching/pipeline.py. Only change vs upstream: the
``regtools.stitching.core`` import becomes ``scripts.stitching.core`` so
the vendored path resolves without the Xu Lab package layout.

--- Original header ---
CLI stitching pipeline script.

Usage:
    python -m scripts.stitching.pipeline --input_dir /path/to/tiles --output_dir /path/to/output
    python -m scripts.stitching.pipeline --input_dir /path/to/tiles --output_dir /path/to/output --section 5
    python -m scripts.stitching.pipeline --input_dir /path/to/tiles --output_dir /path/to/output --channel 0
"""

import os
import sys
import argparse
import logging
import time
import gc

# Ensure project root is on sys.path.
# Compute from __file__ to avoid circular import when run as a standalone script.
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import numpy as np
import cv2
import joblib
from joblib import Parallel, delayed

from scripts.stitching.core import run_tissuecyte_stitching_classic

# Enable ANSI escape sequences on Windows
import colorama
from colorama import Fore, Style
colorama.init()

C_HEADER = Fore.CYAN + Style.BRIGHT
C_STEP = Fore.BLUE + Style.BRIGHT
C_INFO = Fore.WHITE
C_PATH = Fore.YELLOW
C_VALUE = Fore.MAGENTA
C_SUCCESS = Fore.GREEN + Style.BRIGHT
C_WARN = Fore.YELLOW + Style.BRIGHT
C_ERROR = Fore.RED + Style.BRIGHT
C_RESET = Style.RESET_ALL


def stitch_pipeline(input_dir, output_dir, section_num=-1,
                    channel=None, bezier_path=None,
                    n_threads=-3,
                    save_undistorted=False, vignetting_correction=True,
                    verbose=True):
    """Run the full stitching pipeline.
    
    Parameters
    ----------
    input_dir : str
        Input directory containing tile data and Mosaic file.
    output_dir : str
        Output directory for stitched images.
    section_num : int
        Specific section to stitch (-1 for all sections).
    channel : int or None
        Specific channel to stitch (None for all channels).
    bezier_path : str or None
        Path to Bezier patch file. If None, uses default.
    n_threads : int
        Number of parallel threads.
    save_undistorted : bool
        Whether to save undistorted images.
    vignetting_correction : bool
        Apply average tile vignetting correction (default True).
    verbose : bool
        Print progress information.
        
    Returns
    -------
    str
        Path to the output directory.
    """
    start_time = time.time()
    
    if verbose:
        print(f"\n{C_HEADER}{'='*60}{C_RESET}")
        print(f"{C_HEADER}STITCHING PIPELINE{C_RESET}")
        print(f"{C_HEADER}{'='*60}{C_RESET}")
        print(f"  Input:  {C_PATH}{input_dir}{C_RESET}")
        print(f"  Output: {C_PATH}{output_dir}{C_RESET}")
    
    # Setup Bezier patch
    if bezier_path is None:
        bezier_path = os.path.join(_project_root, "data", "stitching_parameters", "bezier16x.pkl")
    
    if not os.path.isfile(bezier_path):
        raise FileNotFoundError(f"Bezier patch file not found: {bezier_path}")
    
    corners1 = np.asarray([[33, 10], [796, 21], [30, 813], [793, 818]])
    corners2 = np.asarray([[20, 20], [776, 20], [20, 794], [776, 794]])
    H, _ = cv2.findHomography(corners1, corners2)
    gridp = run_tissuecyte_stitching_classic.create_perfect_grid(42, 43, 4, 18)
    gridp = gridp[20:794, 20:776]
    
    kx, ky = joblib.load(bezier_path)
    pX_, pY_ = run_tissuecyte_stitching_classic.get_deformation_map(
        gridp.shape[0], gridp.shape[1], kx, ky
    )
    
    root_dir = os.path.join(input_dir, '')
    output_dir = os.path.join(output_dir, '')
    os.makedirs(output_dir, exist_ok=True)
    
    # Get section data
    if verbose:
        print(f"\n{C_STEP}Step 1:{C_RESET} {C_INFO}Loading section data...{C_RESET}")
    
    mosaic_data, section_jsons = run_tissuecyte_stitching_classic.get_section_data(
        root_dir, n_threads, section_num
    )
    
    # Derive actual channel count from tile files, not mosaic metadata
    channel_count = max(len(sj['channels']) for sj in section_jsons) if section_jsons else int(mosaic_data['channels'])
    if verbose:
        print(f"  Found {C_VALUE}{len(section_jsons)}{C_RESET} section(s), {C_VALUE}{channel_count}{C_RESET} channel(s)")
    
    # Create output directories
    for ch in range(channel_count):
        os.makedirs(os.path.join(output_dir, f"stitched_ch{ch}"), exist_ok=True)
    
    if save_undistorted:
        undistorted_dir = os.path.join(output_dir, "undistorted")
        os.makedirs(undistorted_dir, exist_ok=True)
        for ch in range(channel_count):
            os.makedirs(os.path.join(undistorted_dir, f"ch{ch}"), exist_ok=True)
    
    # Generate average tiles for vignetting correction
    if verbose:
        print(f"\n{C_STEP}Step 2:{C_RESET} {C_INFO}Generating average tiles for vignetting correction...{C_RESET}")
    
    average_tiles = []
    noise_thresholds = None
    if vignetting_correction and section_num == -1:
        avg_tiles_dir = os.path.join(output_dir, "avg_tiles")
        noise_thresholds = run_tissuecyte_stitching_classic.generate_avg_tiles(
            section_jsons, avg_tiles_dir, n_threads,
            n_channels=channel_count
        )
        for i in range(channel_count):
            dark = noise_thresholds[i].get('dark_level', 0) if noise_thresholds and i < len(noise_thresholds) else 0
            average_tiles.append(
                run_tissuecyte_stitching_classic.load_average_tile(
                    os.path.join(avg_tiles_dir, f"avg_tile_{i}.tif"),
                    dark_level=dark
                )
            )
    else:
        for i in range(channel_count):
            average_tiles.append(np.ones((832, 832)))
    
    # Stitch sections
    if verbose:
        print(f"\n{C_STEP}Step 3:{C_RESET} {C_INFO}Stitching sections...{C_RESET}")
    
    joblib_backend = 'multiprocessing' if sys.platform == 'win32' else None
    
    Parallel(n_jobs=n_threads, verbose=13 if verbose else 0)(
        delayed(run_tissuecyte_stitching_classic.stitch_section)(
            section_json, average_tiles, output_dir,
            H, pX_, pY_, channel, save_undistorted,
            'multiband', vignetting_correction,
            noise_thresholds
        ) for section_json in section_jsons
    )
    
    # Generate contrast-stretched preview images for quick QC
    if verbose:
        print(f"\n{C_STEP}Generating preview images...{C_RESET}")
    run_tissuecyte_stitching_classic.generate_preview_images(
        output_dir, channel_count, channel
    )

    elapsed = time.time() - start_time
    minutes, seconds = divmod(elapsed, 60)
    
    if verbose:
        print(f"\n{C_SUCCESS}{'='*60}{C_RESET}")
        print(f"{C_SUCCESS}Stitching completed in {int(minutes)}m {seconds:.1f}s{C_RESET}")
        print(f"{C_SUCCESS}{'='*60}{C_RESET}")
        print(f"Output: {C_PATH}{output_dir}{C_RESET}\n")
    
    return output_dir


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    parser = argparse.ArgumentParser(description="Stitching pipeline for brain section tiles")
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Input directory containing tile data and Mosaic file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for stitched images')
    parser.add_argument('--section', type=int, default=-1,
                        help='Specific section number to stitch (-1 for all)')
    parser.add_argument('--channel', type=int, default=None,
                        help='Specific channel to stitch (default: all channels)')
    parser.add_argument('--bezier', type=str, default=None,
                        help='Path to Bezier patch .pkl file')
    parser.add_argument('--threads', type=int, default=-3,
                        help='Number of parallel threads (default: -3)')
    parser.add_argument('--save_undistorted', action='store_true',
                        help='Save undistorted images')
    
    args = parser.parse_args()
    
    # Mirror console output to log file
    os.makedirs(args.output_dir, exist_ok=True)
    from scripts.stitching.core.run_tissuecyte_stitching_classic import _Tee
    _tee = _Tee(os.path.join(args.output_dir, 'console.log'))
    sys.stdout = _tee

    stitch_pipeline(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        section_num=args.section,
        channel=args.channel,
        bezier_path=args.bezier,
        n_threads=args.threads,
        save_undistorted=args.save_undistorted,
    )

    _tee.close()
