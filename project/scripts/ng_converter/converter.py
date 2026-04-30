"""OME-Zarr converter for Neuroglancer viewing.

Ported from UCI-XuLab-RegTools (github.com/UCI-XuLab/UCI-XuLab-RegTools)
on 2026-04-23. Xu-Lab-specific IO (``load_nifti_obj`` + ``load_tiff_volume``
from ``regtools.utils.io``) replaced by direct nibabel + tifffile calls so
this module stands alone without requiring the Xu Lab utils tree. The
OME-Zarr write path and coordinate-transform structure are unchanged from
upstream.

Optional runtime deps (import at call time): ``neuroglancer``, ``zarr``,
``ome-zarr``. Install with ``pip install -e ".[neuroglancer]"``.
"""

from __future__ import annotations

import gc
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from natsort import natsorted

DEFAULT_SPACING_UM = (50.0, 1.25, 1.25)


def _load_tiff_volume(source) -> np.ndarray:
    """Stack a directory of TIFFs (or explicit list) into a 3D (Z,Y,X) array."""
    import tifffile

    if isinstance(source, (list, tuple)):
        paths = natsorted(str(p) for p in source)
        if not paths:
            raise ValueError("No TIFF files provided")
    else:
        src = Path(source)
        paths = natsorted(str(p) for p in src.glob("*.tif"))
        if not paths:
            paths = natsorted(str(p) for p in src.glob("*.tiff"))
        if not paths:
            raise FileNotFoundError(f"No .tif files found in {src}")

    sections = [tifffile.imread(p) for p in paths]
    return np.stack(sections, axis=0)


def get_spacing_from_nii(nii_path) -> tuple[float, float, float]:
    """Return (z, y, x) voxel spacing in micrometers from a NIfTI header.

    NIfTI convention is mm; Neuroglancer convention is um. If the header
    zooms are all > 1.0 we assume micrometer-encoded (Allen CCF style);
    otherwise we convert mm -> um.
    """
    img = nib.load(str(nii_path))
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    if all(z > 1.0 for z in zooms):
        return zooms
    return tuple(z * 1000.0 for z in zooms)


def _build_ome_zarr(volume, output_dir, spacing_um, name="volume") -> str:
    """Write a 3D numpy array as an OME-Zarr store with 5-level pyramid."""
    try:
        import zarr
        from ome_zarr.io import parse_url
        from ome_zarr.writer import write_image
    except ImportError as exc:
        raise RuntimeError(
            "OME-Zarr conversion requires the 'neuroglancer' extras. "
            "Install with: pip install -e \".[neuroglancer]\""
        ) from exc

    zarr_path = os.path.join(output_dir, f"{name}.zarr")
    store = parse_url(zarr_path, mode="w").store
    root = zarr.group(store=store, overwrite=True)

    # OME-Zarr expects 5-D: (T, C, Z, Y, X)
    data_5d = volume[np.newaxis, np.newaxis, ...]

    z_um, y_um, x_um = spacing_um
    axes = [
        {"name": "t", "type": "time", "unit": "second"},
        {"name": "c", "type": "channel"},
        {"name": "z", "type": "space", "unit": "micrometer"},
        {"name": "y", "type": "space", "unit": "micrometer"},
        {"name": "x", "type": "space", "unit": "micrometer"},
    ]

    n_levels = 5
    coord_transforms = []
    for level in range(n_levels):
        factor = 2 ** level
        coord_transforms.append(
            [{"type": "scale", "scale": [1, 1, z_um, y_um * factor, x_um * factor]}]
        )

    write_image(
        image=data_5d,
        group=root,
        axes=axes,
        coordinate_transformations=coord_transforms,
        storage_options=dict(
            chunks=(1, 1, 1, min(volume.shape[1], 512), min(volume.shape[2], 512))
        ),
    )
    store.close()
    return zarr_path


def _apply_axes_transforms(volume, axes_order=None, axes_flips=None) -> np.ndarray:
    if axes_order is not None:
        volume = np.transpose(volume, axes_order)
    if axes_flips:
        for ax in axes_flips:
            volume = np.flip(volume, axis=ax)
    return np.ascontiguousarray(volume)


def convert_tiff_dir(
    tiff_dir,
    output_dir,
    spacing_um=None,
    channel=0,
    orient=True,
    axes_order=None,
    axes_flips=None,
    progress_callback=None,
):
    """Convert a directory of TIFF sections to OME-Zarr.

    `channel` and `orient` are accepted for API parity with Xu Lab; Brainfast's
    simpler loader reads all TIFFs in the directory and does not apply any
    orientation correction (upstream inputs are expected to already be in
    sample voxel space).
    """
    _ = (channel, orient)  # parity with upstream; loader does not filter
    if spacing_um is None:
        spacing_um = DEFAULT_SPACING_UM

    os.makedirs(output_dir, exist_ok=True)
    volume = _load_tiff_volume(tiff_dir)
    if progress_callback:
        progress_callback(1, 2)
    volume = _apply_axes_transforms(volume, axes_order, axes_flips)

    name = os.path.basename(os.path.normpath(str(tiff_dir)))
    zarr_path = _build_ome_zarr(volume, output_dir, spacing_um, name=name)
    del volume
    gc.collect()
    if progress_callback:
        progress_callback(2, 2)
    return zarr_path


def convert_tiff_selection(
    tiff_files,
    output_dir,
    spacing_um=None,
    orient=True,
    axes_order=None,
    axes_flips=None,
    progress_callback=None,
):
    """Convert an explicit list of TIFF files to OME-Zarr."""
    _ = orient
    if spacing_um is None:
        spacing_um = DEFAULT_SPACING_UM
    if not tiff_files:
        raise ValueError("No TIFF files provided")

    os.makedirs(output_dir, exist_ok=True)
    volume = _load_tiff_volume(tiff_files)
    if progress_callback:
        progress_callback(1, 2)
    volume = _apply_axes_transforms(volume, axes_order, axes_flips)
    zarr_path = _build_ome_zarr(volume, output_dir, spacing_um, name="selected_volume")
    del volume
    gc.collect()
    if progress_callback:
        progress_callback(2, 2)
    return zarr_path


def convert_nii(
    nii_path,
    output_dir,
    spacing_um=None,
    axes_order=None,
    axes_flips=None,
    progress_callback=None,
):
    """Convert a NIfTI file to OME-Zarr.

    Spacing is read from the NIfTI header unless overridden.
    """
    os.makedirs(output_dir, exist_ok=True)

    img = nib.load(str(nii_path))
    volume = np.asarray(img.dataobj)
    if progress_callback:
        progress_callback(1, 2)

    if spacing_um is None:
        spacing_um = get_spacing_from_nii(nii_path)

    if volume.ndim > 3:
        volume = volume[..., 0]

    volume = _apply_axes_transforms(volume, axes_order, axes_flips)

    name = os.path.splitext(os.path.splitext(os.path.basename(str(nii_path)))[0])[0]
    zarr_path = _build_ome_zarr(volume, output_dir, spacing_um, name=name)
    del volume
    gc.collect()

    if progress_callback:
        progress_callback(2, 2)
    return zarr_path


def convert_auto(
    input_path,
    output_dir,
    spacing_um=None,
    orient=True,
    axes_order=None,
    axes_flips=None,
    progress_callback=None,
):
    """Auto-detect input format and convert to OME-Zarr.

    Supports a TIFF directory, a single NIfTI file, or a single TIFF file.
    Raises ``ValueError`` when the input type is ambiguous.
    """
    p = Path(str(input_path))
    if p.is_dir():
        return convert_tiff_dir(
            p, output_dir,
            spacing_um=spacing_um,
            orient=orient,
            axes_order=axes_order,
            axes_flips=axes_flips,
            progress_callback=progress_callback,
        )

    suffix = p.suffix.lower()
    if suffix in (".nii",) or str(p).lower().endswith(".nii.gz"):
        return convert_nii(
            p, output_dir,
            spacing_um=spacing_um,
            axes_order=axes_order,
            axes_flips=axes_flips,
            progress_callback=progress_callback,
        )
    if suffix in (".tif", ".tiff"):
        return convert_tiff_selection(
            [p], output_dir,
            spacing_um=spacing_um,
            orient=orient,
            axes_order=axes_order,
            axes_flips=axes_flips,
            progress_callback=progress_callback,
        )
    raise ValueError(f"Unsupported input for OME-Zarr conversion: {p}")
