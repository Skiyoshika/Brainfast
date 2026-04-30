"""Neuroglancer OME-Zarr + point-annotation converter.

Ported from UCI-XuLab-RegTools (github.com/UCI-XuLab/UCI-XuLab-RegTools).
Vendored into Brainfast 2026-04-23 with Xu-Lab-specific IO (load_nifti_obj,
load_tiff_volume) replaced by direct nibabel + tifffile calls so this
module stands alone without the rest of the Xu Lab utils tree.

Optional runtime deps: install with ``pip install -e ".[neuroglancer]"``.
"""

from .converter import (
    convert_auto,
    convert_nii,
    convert_tiff_dir,
    convert_tiff_selection,
    get_spacing_from_nii,
)
from .points import (
    convert_points_file,
    load_points,
    write_precomputed_annotations,
)

__all__ = [
    "convert_auto",
    "convert_nii",
    "convert_tiff_dir",
    "convert_tiff_selection",
    "get_spacing_from_nii",
    "convert_points_file",
    "load_points",
    "write_precomputed_annotations",
]
