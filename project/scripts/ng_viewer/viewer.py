"""Self-hosted Neuroglancer viewer.

Ported from UCI-XuLab-RegTools (github.com/UCI-XuLab/UCI-XuLab-RegTools)
on 2026-04-23. Xu-Lab-specific IO (``load_nifti_obj`` + ``load_tiff_volume``
from ``regtools.utils.io``) replaced by direct nibabel + tifffile calls.
GLSL shaders, layer helpers, and the ``launch_viewer`` API are unchanged.

Optional runtime deps (import at call time): ``neuroglancer``, ``zarr``.
"""

from __future__ import annotations

import inspect
import os
import webbrowser
from pathlib import Path

import nibabel as nib
import numpy as np
from natsort import natsorted

from ..ng_converter.points import load_points

DEFAULT_SPACING_UM = (50.0, 1.25, 1.25)


def image_shader() -> str:
    code = """
    #uicontrol invlerp normalized(range=[0, 3500], window=[0, 65535])
    void main() {
        emitGrayscale(normalized());
    }
    """
    return inspect.cleandoc(code)


def volume_shader() -> str:
    code = """
    #uicontrol float brightness slider(min=-10, max=10, default=0, step=0.1)
    #uicontrol float tissueMinValue slider(min=0, max=65535, default=0, step=1)
    #uicontrol float tissueMaxValue slider(min=0, max=65535, default=400, step=1)
    #uicontrol vec3 tissueColor color(default=\"#FFFFFF\")
    #uicontrol float cellMinValue slider(min=0, max=65535, default=500, step=1)
    #uicontrol vec3 cellColor color(default=\"#FFFF00\")

    void main() {
        float x = float(getInterpolatedDataValue().value);
        float norm = (x - tissueMinValue) / (tissueMaxValue - tissueMinValue);
        vec4 voxelColor = vec4(tissueColor, norm*exp(brightness));
        if (x > cellMinValue) {
            voxelColor = vec4(cellColor, norm*exp(brightness));
        }
        emitRGBA(voxelColor);
    }
    """
    return inspect.cleandoc(code)


def segmentation_shader() -> str:
    code = """
    #uicontrol float brightness slider(min=-1, max=1, default=0)
    #uicontrol float contrast slider(min=0, max=2, default=1)

    void main() {
        uint value = uint(getDataValue());
        float r = fract(sin(float(value) * 12.9898) * 43758.5453);
        float g = fract(sin(float(value) * 78.233) * 43758.5453);
        float b = fract(sin(float(value) * 39.346) * 43758.5453);
        vec3 color = vec3(r, g, b);
        color = (color - 0.5) * contrast + 0.5 + brightness;
        emitRGB(color);
    }
    """
    return inspect.cleandoc(code)


def _require_neuroglancer():
    try:
        import neuroglancer
    except ImportError as exc:
        raise RuntimeError(
            "Neuroglancer viewer requires the 'neuroglancer' extras. "
            "Install with: pip install -e \".[neuroglancer]\""
        ) from exc
    return neuroglancer


def _make_dimensions(spacing_um):
    neuroglancer = _require_neuroglancer()
    z, y, x = spacing_um
    return neuroglancer.CoordinateSpace(
        names=["z", "y", "x"], units=["um", "um", "um"], scales=[z, y, x],
    )


def _load_volume_from_tiff_dir(tiff_dir) -> np.ndarray:
    import tifffile

    src = Path(tiff_dir)
    paths = natsorted(str(p) for p in src.glob("*.tif"))
    if not paths:
        paths = natsorted(str(p) for p in src.glob("*.tiff"))
    if not paths:
        raise FileNotFoundError(f"No .tif files found in {src}")
    sections = [tifffile.imread(p) for p in paths]
    return np.stack(sections, axis=0)


def _load_volume_from_nii(nii_path) -> np.ndarray:
    img = nib.load(str(nii_path))
    vol = np.asarray(img.dataobj)
    if vol.ndim > 3:
        vol = vol[..., 0]
    return vol


def _load_volume_from_zarr(zarr_path) -> np.ndarray:
    try:
        import zarr as _zarr
    except ImportError as exc:
        raise RuntimeError(
            "OME-Zarr loading requires the 'neuroglancer' extras. "
            "Install with: pip install -e \".[neuroglancer]\""
        ) from exc
    root = _zarr.open(str(zarr_path), mode="r")
    if "0" in root:
        data = np.asarray(root["0"])
    else:
        data = np.asarray(root)
    data = np.squeeze(data)
    if data.ndim > 3:
        data = data[0]
    return data


def add_image_layer(state, name, volume, dims, shader=None):
    neuroglancer = _require_neuroglancer()
    layer = neuroglancer.LocalVolume(
        data=volume, dimensions=dims, volume_type="image", voxel_offset=[0, 0, 0]
    )
    state.layers.append(
        name=name,
        layer=layer,
        opacity=1.0,
        shader=shader or image_shader(),
    )
    return layer


def add_volume_layer(state, name, volume, dims):
    neuroglancer = _require_neuroglancer()
    layer = neuroglancer.LocalVolume(
        data=volume, dimensions=dims, volume_type="image", voxel_offset=[0, 0, 0]
    )
    state.layers.append(
        name=name,
        layer=layer,
        opacity=0,
        shader=volume_shader(),
        shaderControls={"brightness": -1.5, "tissueMinValue": 40},
        blend="additive",
        volume_rendering=True,
        volumeRenderingGain=-7.1,
    )
    return layer


def add_segmentation_layer(state, name, volume, dims):
    neuroglancer = _require_neuroglancer()
    layer = neuroglancer.LocalVolume(
        data=volume, dimensions=dims, volume_type="image", voxel_offset=[0, 0, 0]
    )
    state.layers.append(name=name, layer=layer, shader=segmentation_shader())
    return layer


def add_points_layer(state, name, points, color="#ff0000"):
    neuroglancer = _require_neuroglancer()
    annotations = [
        neuroglancer.PointAnnotation(
            id=f"point{i}", point=[float(pt[2]), float(pt[1]), float(pt[0])]
        )
        for i, pt in enumerate(points)
    ]
    layer = neuroglancer.AnnotationLayer(
        annotations=annotations, annotation_color=color,
    )
    state.layers.append(name=name, layer=layer)
    return layer


POINT_COLORS = ["#ff0000", "#ffff00", "#00ff00", "#00ffff", "#ff00ff", "#ffffff"]


def launch_viewer(
    image_inputs=None,
    points_inputs=None,
    segmentation_path=None,
    spacing_um=None,
    bind_address="127.0.0.1",
    port=0,
    open_browser=True,
    orient_tiffs=True,
):
    """Create a self-hosted Neuroglancer viewer and return it.

    Parameters match the upstream Xu Lab implementation. See
    ``regtools/ng_viewer/viewer.py`` for the full doc.
    """
    _ = orient_tiffs
    neuroglancer = _require_neuroglancer()
    if spacing_um is None:
        spacing_um = DEFAULT_SPACING_UM
    dims = _make_dimensions(spacing_um)

    neuroglancer.set_server_bind_address(bind_address, port)
    viewer = neuroglancer.Viewer()

    volumes = []
    if image_inputs:
        for idx, inp in enumerate(image_inputs):
            path = inp["path"]
            itype = inp.get("type", "nii")
            name = inp.get("name", f"Layer {idx + 1}")
            print(f"Loading {itype}: {path}")

            if itype == "tiff_dir":
                vol = _load_volume_from_tiff_dir(path)
            elif itype == "nii":
                vol = _load_volume_from_nii(path)
            elif itype == "zarr":
                vol = _load_volume_from_zarr(path)
            elif itype == "zarr_url":
                volumes.append({"name": name, "url": path})
                continue
            else:
                print(f"  Unknown type '{itype}', skipping")
                continue

            print(f"  Shape: {vol.shape}")
            volumes.append({"name": name, "volume": vol})

    points_data = []
    if points_inputs:
        for ppath in points_inputs:
            try:
                pts = load_points(ppath)
                pts_sorted = pts[pts[:, 0].argsort()[::-1]]
                points_data.append(
                    (os.path.splitext(os.path.basename(ppath))[0], pts_sorted)
                )
                print(f"Loaded {len(pts_sorted)} points from {ppath}")
            except Exception as e:  # noqa: BLE001
                print(f"Warning: could not load points from {ppath}: {e}")

    seg_vol = None
    if segmentation_path:
        print(f"Loading segmentation: {segmentation_path}")
        seg_vol = _load_volume_from_nii(segmentation_path)
        print(f"  Shape: {seg_vol.shape}")

    with viewer.txn() as s:
        for item in volumes:
            if "url" in item:
                s.layers.append(
                    name=item["name"],
                    source=f"zarr://{item['url']}",
                    shader=image_shader(),
                )
            else:
                add_image_layer(s, item["name"], item["volume"], dims)

        local_vols = [v for v in volumes if "volume" in v]
        if local_vols:
            add_volume_layer(s, "Volume Rendering", local_vols[0]["volume"], dims)

        for ci, (pname, pts) in enumerate(points_data):
            color = POINT_COLORS[ci % len(POINT_COLORS)]
            add_points_layer(s, pname, pts, color=color)

        if seg_vol is not None:
            add_segmentation_layer(s, "Segmentation Atlas", seg_vol, dims)

        s.voxel_coordinates = [0, 0, 0]

    url = viewer.get_viewer_url()
    print(f"\nNeuroglancer URL: {url}")
    if open_browser:
        webbrowser.open(url)

    return viewer
