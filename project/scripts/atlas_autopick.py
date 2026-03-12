from __future__ import annotations

from pathlib import Path
import numpy as np
from tifffile import imread, imwrite
from skimage.filters import sobel
from skimage.metrics import structural_similarity as ssim


def _roi_bbox_from_real(real_img: np.ndarray, pad: int = 4) -> tuple[int, int, int, int]:
    x = real_img.astype(np.float32)
    if x.ndim == 3:
        x = x[..., 0]
    thr = float(np.percentile(x, 88))
    mask = x > thr
    from skimage import measure
    lbl = measure.label(mask, connectivity=2)
    props = measure.regionprops(lbl)
    if not props:
        h, w = x.shape
        return (0, 0, int(w), int(h))
    props = sorted(props, key=lambda r: r.area, reverse=True)
    y0, x0, y1, x1 = props[0].bbox
    h, w = x.shape
    x0 = max(0, int(x0) - pad)
    y0 = max(0, int(y0) - pad)
    x1 = min(w, int(x1) + pad)
    y1 = min(h, int(y1) + pad)
    return (x0, y0, int(x1 - x0), int(y1 - y0))


def _prep_real_edges(real_img: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    if real_img.ndim == 3:
        real_img = real_img[..., 0]
    
    # Match spatial dimensions through padding or center crop
    th, tw = target_shape
    sh, sw = real_img.shape
    out = np.zeros((th, tw), dtype=np.float32)
    min_h = min(th, sh)
    min_w = min(tw, sw)
    
    start_y_out = (th - min_h) // 2
    start_x_out = (tw - min_w) // 2
    start_y_in = (sh - min_h) // 2
    start_x_in = (sw - min_w) // 2
    
    out[start_y_out : start_y_out + min_h, start_x_out : start_x_out + min_w] = \
        real_img[start_y_in : start_y_in + min_h, start_x_in : start_x_in + min_w]

    return sobel(out)


def _label_edge_score(real_edges: np.ndarray, label_slice: np.ndarray) -> float:
    lbl = (label_slice > 0).astype(np.float32)
    lbl_e = sobel(lbl)
    return float(ssim(real_edges, lbl_e, data_range=float(max(np.ptp(real_edges), np.ptp(lbl_e), 1.0))))


def autopick_best_z(real_path: Path, annotation_nii: Path, out_label_tif: Path, z_step: int = 1, pixel_size_um: float = 0.65, slicing_plane: str = "coronal", roi_mode: str = "auto") -> dict:
    import nibabel as nib

    real = imread(str(real_path))
    nii = nib.load(str(annotation_nii))
    vol = np.asarray(nii.get_fdata(), dtype=np.int32)

    plane = str(slicing_plane or "coronal").lower()

    def _get_slice(v: np.ndarray, z: int, p: str) -> np.ndarray:
        if p == "coronal":
            return v[z, :, :]
        if p == "sagittal":
            return v[:, :, z]
        if p in ("horizontal", "axial"):
            return v[:, z, :]
        raise ValueError(f"unsupported slicing_plane: {p}")

    if plane == "coronal":
        z_dim = vol.shape[0]
    elif plane == "sagittal":
        z_dim = vol.shape[2]
    elif plane in ("horizontal", "axial"):
        z_dim = vol.shape[1]
    else:
        raise ValueError(f"unsupported slicing_plane: {plane}")

    sample = _get_slice(vol, 0, plane)
    target_shape = sample.shape
    
    # ROI-first (optional) before scale/matching
    roi_bbox = (0, 0, int(real.shape[1]), int(real.shape[0]))
    roi = real
    if str(roi_mode or "off").lower() in ("auto", "on", "true"):
        x0, y0, rw, rh = _roi_bbox_from_real(real)
        roi_bbox = (int(x0), int(y0), int(rw), int(rh))
        roi = real[y0:y0 + rh, x0:x0 + rw]

    # Pre-scale real ROI to atlas space to improve Z-matching performance
    scale = pixel_size_um / 25.0
    real_f = roi.astype(np.float32)
    from skimage.transform import rescale
    real_scaled = rescale(real_f, scale, order=1, preserve_range=True, anti_aliasing=True)
    real_edges = _prep_real_edges(real_scaled, target_shape)

    best_z, best_score = 0, -1.0
    for z in range(0, z_dim, max(1, int(z_step))):
        atlas_slice = _get_slice(vol, z, plane)
        s = _label_edge_score(real_edges, atlas_slice)
        if s > best_score:
            best_score = s
            best_z = z

    best_slice = _get_slice(vol, best_z, plane).astype(np.uint16)
    out_label_tif.parent.mkdir(parents=True, exist_ok=True)
    imwrite(str(out_label_tif), best_slice)

    return {
        "best_z": int(best_z),
        "best_score": float(best_score),
        "label_slice_tif": str(out_label_tif),
        "shape": [int(x) for x in vol.shape],
        "slicing_plane": plane,
        "slice_shape": [int(x) for x in best_slice.shape],
        "roi_mode": str(roi_mode),
        "roi_bbox": [int(roi_bbox[0]), int(roi_bbox[1]), int(roi_bbox[2]), int(roi_bbox[3])],
    }
