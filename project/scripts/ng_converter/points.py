# ruff: noqa
"""Vendored from UCI-XuLab-RegTools (github.com/UCI-XuLab/UCI-XuLab-RegTools).

Vendored into Brainfast on 2026-04-23 as
project/scripts/ng_converter/points.py. Unmodified from upstream.

--- Original header ---
Convert point coordinate files to Neuroglancer precomputed point annotations.

Supports two input formats:
  - **Legacy** (``inputpoints.txt``): two header lines (``index`` + count),
    then ``z y x`` per line.
  - **Napari CSV**: CSV with columns ``index, axis-0, axis-1, axis-2``
    where axes map to ``z, y, x``.
"""

import os
import json
import struct
from pathlib import Path

import numpy as np


def _detect_format(file_path):
    """Return 'napari_csv' or 'legacy' based on the first line."""
    with open(file_path, "r") as f:
        first_line = f.readline().strip()
    if "," in first_line and "axis" in first_line.lower():
        return "napari_csv"
    return "legacy"


def load_legacy_points(file_path):
    """Load points from the legacy ``inputpoints.txt`` format.

    Returns an Nx3 array with columns (z, y, x).
    """
    points = np.loadtxt(file_path, skiprows=2)
    if points.ndim == 1:
        points = points.reshape(1, -1)
    return points


def load_napari_csv(file_path):
    """Load points from a napari-exported CSV file.

    Expected columns: ``index, axis-0, axis-1, axis-2``  →  ``(_, z, y, x)``

    Returns an Nx3 array with columns (z, y, x).
    """
    data = np.loadtxt(file_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    # Drop the index column (column 0), keep z, y, x
    return data[:, 1:4]


def load_points(file_path):
    """Auto-detect format and load points. Returns Nx3 array (z, y, x)."""
    fmt = _detect_format(file_path)
    if fmt == "napari_csv":
        return load_napari_csv(file_path)
    return load_legacy_points(file_path)


def write_precomputed_annotations(points, output_dir, spacing_um=None):
    """Write Neuroglancer precomputed point annotations.

    Parameters
    ----------
    points : np.ndarray
        Nx3 array of point coordinates (z, y, x).
    output_dir : str or Path
        Output directory. A subfolder ``precomputed_points`` is created.
    spacing_um : tuple of 3 floats, optional
        (Z, Y, X) voxel spacing in micrometers.
        Defaults to (50, 1.25, 1.25).
    """
    if spacing_um is None:
        spacing_um = (50.0, 1.25, 1.25)

    output_dir = Path(output_dir)
    points_dir = output_dir / "precomputed_points"
    points_dir.mkdir(parents=True, exist_ok=True)

    z_sp, y_sp, x_sp = spacing_um

    if len(points) > 0:
        lower = points.min(axis=0).astype(float).tolist()
        upper = (points.max(axis=0) + 1).astype(float).tolist()
        chunk_size = (points.max(axis=0) + 1).astype(int).tolist()
    else:
        lower = [0, 0, 0]
        upper = [1, 1, 1]
        chunk_size = [1, 1, 1]

    info = {
        "@type": "neuroglancer_annotations_v1",
        "annotation_type": "POINT",
        "by_id": {"key": "by_id"},
        "dimensions": {
            "z": [z_sp, "um"],
            "y": [y_sp, "um"],
            "x": [x_sp, "um"],
        },
        "lower_bound": lower,
        "upper_bound": upper,
        "properties": [],
        "relationships": [],
        "spatial": [
            {
                "chunk_size": chunk_size,
                "grid_shape": [1, 1, 1],
                "key": "spatial0",
                "limit": len(points) + 1,
            }
        ],
    }

    info_path = points_dir / "info"
    with info_path.open("w") as f:
        json.dump(info, f, indent=2)

    spatial_dir = points_dir / "spatial0"
    spatial_dir.mkdir(parents=True, exist_ok=True)
    spatial_path = spatial_dir / "0_0_0"

    with spatial_path.open("wb") as f:
        n = len(points)
        buf = struct.pack("<Q", n)
        for z, y, x in points:
            buf += struct.pack("<3f", float(z), float(y), float(x))
        buf += struct.pack("<%dQ" % n, *range(n))
        f.write(buf)

    return str(points_dir)


def convert_points_file(file_path, output_dir, spacing_um=None):
    """Auto-detect format, load points, and write precomputed annotations.

    Returns the path to the created precomputed_points directory.
    """
    points = load_points(file_path)
    return write_precomputed_annotations(points, output_dir, spacing_um=spacing_um)
