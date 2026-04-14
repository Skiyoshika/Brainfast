from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
from tifffile import TiffFile, imread


@dataclass(frozen=True)
class VolumeSourceInfo:
    source_path: str
    source_type: str
    shape: tuple[int, ...]
    axes: str
    dtype: str
    num_slices: int


def inspect_volume_source(src: Path) -> VolumeSourceInfo:
    src = Path(src)
    if src.is_dir():
        files = sorted(src.glob("z*.tif"))
        if not files:
            raise FileNotFoundError(f"No z*.tif files found in {src}")
        sample = _select_channel(imread(str(files[0])), channel=0)
        return VolumeSourceInfo(
            source_path=str(src.resolve()),
            source_type="slice_dir",
            shape=(len(files), int(sample.shape[0]), int(sample.shape[1])),
            axes="ZYX",
            dtype=str(sample.dtype),
            num_slices=len(files),
        )

    if not src.exists():
        raise FileNotFoundError(f"volume source not found: {src}")

    with TiffFile(str(src)) as tf:
        series = tf.series[0]
        shape = tuple(int(v) for v in series.shape)
        return VolumeSourceInfo(
            source_path=str(src.resolve()),
            source_type="stack_tiff",
            shape=shape,
            axes=str(getattr(series, "axes", "")),
            dtype=str(series.dtype),
            num_slices=int(shape[0]) if len(shape) >= 3 else 1,
        )


def volume_source_to_nifti(
    src: Path,
    output_path: Path,
    pixel_um_xy: float,
    z_spacing_um: float,
    *,
    target_um: float | None = 25.0,
    channel: int = 0,
    every_n: int = 1,
    z_min: int = 0,
    z_max: int = -1,
    pad_z: int = 0,
    pad_y: int = 0,
    pad_x: int = 0,
    normalize: bool = False,
) -> tuple[Path, tuple[int, int, int]]:
    src = Path(src)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    every_n = max(1, int(every_n))
    ds = _resolve_downsample(pixel_um_xy, target_um)
    pad_z = max(0, int(pad_z))
    pad_y = max(0, int(pad_y))
    pad_x = max(0, int(pad_x))

    if src.is_dir():
        vol = _slice_dir_to_volume(
            src,
            channel=channel,
            ds=ds,
            every_n=every_n,
            z_min=z_min,
            z_max=z_max,
            pad_z=pad_z,
            pad_y=pad_y,
            pad_x=pad_x,
            normalize=normalize,
        )
    else:
        vol = _stack_tiff_to_volume(
            src,
            channel=channel,
            ds=ds,
            every_n=every_n,
            z_min=z_min,
            z_max=z_max,
            pad_z=pad_z,
            pad_y=pad_y,
            pad_x=pad_x,
            normalize=normalize,
        )

    vox_mm = (
        float(z_spacing_um) * every_n / 1000.0,
        float(pixel_um_xy) * ds / 1000.0,
        float(pixel_um_xy) * ds / 1000.0,
    )
    affine = _make_brainfast_affine(vox_mm)
    img = nib.Nifti1Image(vol, affine)
    img.set_qform(affine, code=1)
    img.set_sform(affine, code=1)
    img.header.set_zooms(vox_mm)
    nib.save(img, str(output_path))
    return output_path, tuple(int(v) for v in vol.shape)


def _make_brainfast_affine(vox_mm: tuple[float, float, float]) -> np.ndarray:
    """Match the historical Brainfast/UCI volumetric orientation (PIR).

    The volume array is stored as (z, y, x). The legacy pipeline writes NIfTI
    with voxel axes mapped so that ANTs/ITK sees the same orientation as the
    previously validated half-brain workflow:
      axis 0 (stack depth) -> posterior/anterior with negative direction
      axis 1 (image rows)  -> inferior/superior with negative direction
      axis 2 (image cols)  -> right/left with positive direction
    """

    z_mm, y_mm, x_mm = (float(vox_mm[0]), float(vox_mm[1]), float(vox_mm[2]))
    affine = np.zeros((4, 4), dtype=np.float32)
    affine[0, 2] = x_mm
    affine[1, 0] = -z_mm
    affine[2, 1] = -y_mm
    affine[3, 3] = 1.0
    return affine


def _resolve_downsample(pixel_um_xy: float, target_um: float | None) -> int:
    if target_um is None:
        return 1
    try:
        pixel = float(pixel_um_xy)
        target = float(target_um)
    except Exception:
        return 1
    if pixel <= 0 or target <= pixel:
        return 1
    return max(1, int(round(target / pixel)))


def _downsampled_size(n: int, step: int) -> int:
    return (int(n) + int(step) - 1) // int(step)


def _select_channel(arr: np.ndarray, channel: int) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim != 3:
        raise ValueError(f"unsupported slice shape: {arr.shape}")

    if arr.shape[-1] <= 4:
        idx = min(max(int(channel), 0), arr.shape[-1] - 1)
        return arr[..., idx]
    if arr.shape[0] <= 4:
        idx = min(max(int(channel), 0), arr.shape[0] - 1)
        return arr[idx, ...]
    return arr[0, ...]


def _normalize_slice(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    x = arr.astype(np.float32, copy=False)
    x = np.clip((x - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return (x * 65535.0).astype(np.uint16)


def _estimate_range_from_arrays(arrays: list[np.ndarray]) -> tuple[float, float]:
    if not arrays:
        return 0.0, 1.0
    sample = np.concatenate([a.ravel() for a in arrays]).astype(np.float32, copy=False)
    lo = float(np.percentile(sample, 1.0))
    hi = float(np.percentile(sample, 99.5))
    if hi <= lo:
        hi = lo + 1.0
    return lo, hi


def _slice_dir_to_volume(
    src: Path,
    *,
    channel: int,
    ds: int,
    every_n: int,
    z_min: int,
    z_max: int,
    pad_z: int,
    pad_y: int,
    pad_x: int,
    normalize: bool,
) -> np.ndarray:
    files = sorted(src.glob("z*.tif"))
    if not files:
        raise FileNotFoundError(f"No z*.tif files found in {src}")
    z_indices = _slice_indices(len(files), z_min=z_min, z_max=z_max, every_n=every_n)

    first = _select_channel(imread(str(files[z_indices[0]])), channel=channel)[::ds, ::ds]
    out_dtype = np.uint16 if normalize else np.float32
    out = np.zeros(
        (
            len(z_indices) + pad_z * 2,
            _downsampled_size(first.shape[0], 1) + pad_y * 2,
            _downsampled_size(first.shape[1], 1) + pad_x * 2,
        ),
        dtype=out_dtype,
    )

    lo, hi = 0.0, 1.0
    if normalize:
        sample_step = max(1, len(z_indices) // 24)
        samples = []
        for idx in z_indices[::sample_step]:
            arr = _select_channel(imread(str(files[idx])), channel=channel)[::ds, ::ds]
            stride_y = max(1, arr.shape[0] // 128)
            stride_x = max(1, arr.shape[1] // 128)
            samples.append(arr[::stride_y, ::stride_x])
        lo, hi = _estimate_range_from_arrays(samples)

    for out_idx, src_idx in enumerate(z_indices):
        arr = _select_channel(imread(str(files[src_idx])), channel=channel)[::ds, ::ds]
        prepared = _normalize_slice(arr, lo, hi) if normalize else arr.astype(np.float32)
        out[
            pad_z + out_idx,
            pad_y : pad_y + prepared.shape[0],
            pad_x : pad_x + prepared.shape[1],
        ] = prepared

    return out


def _stack_tiff_to_volume(
    src: Path,
    *,
    channel: int,
    ds: int,
    every_n: int,
    z_min: int,
    z_max: int,
    pad_z: int,
    pad_y: int,
    pad_x: int,
    normalize: bool,
) -> np.ndarray:
    with TiffFile(str(src)) as tf:
        series = tf.series[0]
        use_series_array = len(tf.pages) == 1 and len(series.shape) >= 3
        n_pages = int(series.shape[0]) if use_series_array else max(1, len(tf.pages))
        z_indices = _slice_indices(n_pages, z_min=z_min, z_max=z_max, every_n=every_n)
        if use_series_array:
            series_arr = series.asarray()
            first_src = series_arr[z_indices[0]]
        else:
            series_arr = None
            first_src = tf.pages[z_indices[0]].asarray()
        first = _select_channel(first_src, channel=channel)[::ds, ::ds]
        out_dtype = np.uint16 if normalize else np.float32
        out = np.zeros(
            (
                len(z_indices) + pad_z * 2,
                _downsampled_size(first.shape[0], 1) + pad_y * 2,
                _downsampled_size(first.shape[1], 1) + pad_x * 2,
            ),
            dtype=out_dtype,
        )

        lo, hi = 0.0, 1.0
        if normalize:
            sample_step = max(1, len(z_indices) // 24)
            samples = []
            for idx in z_indices[::sample_step]:
                src_arr = series_arr[idx] if use_series_array else tf.pages[idx].asarray()
                arr = _select_channel(src_arr, channel=channel)[::ds, ::ds]
                stride_y = max(1, arr.shape[0] // 128)
                stride_x = max(1, arr.shape[1] // 128)
                samples.append(arr[::stride_y, ::stride_x])
            lo, hi = _estimate_range_from_arrays(samples)

        for out_idx, src_idx in enumerate(z_indices):
            src_arr = series_arr[src_idx] if use_series_array else tf.pages[src_idx].asarray()
            arr = _select_channel(src_arr, channel=channel)[::ds, ::ds]
            prepared = _normalize_slice(arr, lo, hi) if normalize else arr.astype(np.float32)
            out[
                pad_z + out_idx,
                pad_y : pad_y + prepared.shape[0],
                pad_x : pad_x + prepared.shape[1],
            ] = prepared

    return out


def _slice_indices(length: int, *, z_min: int, z_max: int, every_n: int) -> list[int]:
    if length <= 0:
        raise ValueError("volume length must be > 0")
    start = max(0, int(z_min))
    stop = length - 1 if int(z_max) < 0 else min(int(z_max), length - 1)
    if stop < start:
        raise ValueError(f"invalid z range: {start}..{stop}")
    return list(range(start, stop + 1, max(1, int(every_n))))
