"""Self-hosted Neuroglancer viewer.

Ported from UCI-XuLab-RegTools (github.com/UCI-XuLab/UCI-XuLab-RegTools).
Vendored into Brainfast 2026-04-23 with Xu-Lab-specific IO replaced by
direct nibabel + tifffile calls. GLSL shaders, layer helpers, and the
``launch_viewer`` API are unchanged from upstream.

Optional runtime deps: ``neuroglancer``, ``zarr``. Install with
``pip install -e ".[neuroglancer]"``.
"""

from .viewer import (
    DEFAULT_SPACING_UM,
    add_image_layer,
    add_points_layer,
    add_segmentation_layer,
    add_volume_layer,
    image_shader,
    launch_viewer,
    segmentation_shader,
    volume_shader,
)

__all__ = [
    "DEFAULT_SPACING_UM",
    "add_image_layer",
    "add_points_layer",
    "add_segmentation_layer",
    "add_volume_layer",
    "image_shader",
    "launch_viewer",
    "segmentation_shader",
    "volume_shader",
]
