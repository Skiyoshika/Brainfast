from __future__ import annotations

import importlib
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import map_coordinates

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from scripts.laplacian_refine_3d import refine_registered_volume
    from scripts.pipeline_progress import write_stage_progress
    from scripts.registration_3d_ants import run_ants_registration
    from scripts.registration_3d_volume import (
        build_volume_from_tiffs,
        prepare_half_template_inputs,
    )
    from scripts.truth_export_3d import export_registered_truth_slices
except ImportError:
    from laplacian_refine_3d import refine_registered_volume
    from pipeline_progress import write_stage_progress
    from registration_3d_ants import run_ants_registration
    from registration_3d_volume import build_volume_from_tiffs, prepare_half_template_inputs
    from truth_export_3d import export_registered_truth_slices


def run_quantification_from_truth(**kwargs) -> dict:
    fn = kwargs.pop("quantify_fn")
    return fn(**kwargs)


def _apply_refinement_field_to_annotation_volume(
    annotation_path: Path,
    field_path: Path,
    output_path: Path,
) -> Path:
    annotation_img = nib.load(str(annotation_path))
    annotation = np.asarray(annotation_img.dataobj, dtype=np.float32)
    field = np.load(str(field_path)).astype(np.float32, copy=False)

    if field.ndim != 4 or field.shape[0] != annotation.ndim:
        raise ValueError(
            f"refinement field must have shape ({annotation.ndim}, ...); got {tuple(field.shape)}"
        )
    if tuple(field.shape[1:]) != tuple(annotation.shape):
        raise ValueError(
            f"refinement field shape {tuple(field.shape[1:])} does not match annotation volume {tuple(annotation.shape)}"
        )

    base_coords = np.meshgrid(
        *[np.arange(size, dtype=np.float32) for size in annotation.shape],
        indexing="ij",
    )
    sample_coords = np.stack(base_coords, axis=0) - field
    refined = map_coordinates(
        annotation,
        sample_coords,
        order=0,
        mode="nearest",
    ).astype(np.int32)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(refined, annotation_img.affine, annotation_img.header),
        str(output_path),
    )
    return output_path


def _fix_transform_extensions(transforms: list[str]) -> list[str]:
    """Ensure ANTs displacement fields use .nii.gz extension, not bare .gz."""
    fixed = []
    for tf in transforms:
        tf = str(tf)
        if tf.endswith(".gz") and not tf.endswith(".nii.gz"):
            nii_gz = tf[:-3] + ".nii.gz"
            if Path(nii_gz).exists():
                tf = nii_gz
        fixed.append(tf)
    return fixed


def _count_nonzero_and_coverage(vol: np.ndarray, label: str) -> tuple[int, float]:
    """Count nonzero voxels and print coverage stats. Returns (nonzero, fraction)."""
    total = int(vol.size)
    nonzero = int(np.count_nonzero(vol))
    frac = nonzero / total if total > 0 else 0.0
    print(f"[warp] {label}: {nonzero:,} nonzero / {total:,} total ({frac:.1%} coverage)")
    return nonzero, frac


# Minimum fraction of output voxels that should be nonzero for the warp
# to be considered successful.  The Allen annotation has ~55% nonzero in
# template space; even with imperfect registration we expect at least 20%.
_MIN_COVERAGE_FRACTION = 0.10


def _extrapolate_annotation_to_tissue(
    annotation_path: Path,
    input_volume_path: Path,
    merged_slice_paths: list[Path],
) -> Path:
    """Fill annotation gaps where tissue exists but the half-hemisphere template had no coverage.

    For each Z-slice, builds a tissue mask from the real input, then for every
    tissue pixel without an annotation, copies the label from the nearest
    annotated pixel along the X (ML) axis.  This fills the medial gap left by
    the half-hemisphere crop without altering already-annotated regions.
    """
    from scipy.ndimage import distance_transform_edt
    from tifffile import imread as _tif_read

    ann_img = nib.load(str(annotation_path))
    ann_vol = np.asarray(ann_img.dataobj, dtype=np.int32)

    # Build a tissue mask from real slices (more reliable than the downsampled volume)
    # We just need a binary mask: is there tissue here?
    n_slices = ann_vol.shape[0]
    filled_count = 0

    for z in range(n_slices):
        ann_slice = ann_vol[z]
        has_annotation = ann_slice > 0

        # Skip slices that are fully annotated or fully empty
        if has_annotation.all() or not has_annotation.any():
            continue

        # Read the real image to get tissue mask
        if z < len(merged_slice_paths):
            try:
                real_img = _tif_read(str(merged_slice_paths[z]))
                if real_img.ndim == 3:
                    real_img = real_img[..., 0]
                # Resize to annotation slice shape if needed
                if real_img.shape != ann_slice.shape:
                    from skimage.transform import resize

                    real_img = resize(
                        real_img.astype(np.float32),
                        ann_slice.shape,
                        order=1,
                        preserve_range=True,
                    ).astype(np.float32)
            except Exception:
                continue
        else:
            continue

        # Build tissue mask: anything above background
        bg_estimate = np.percentile(real_img[real_img > 0], 5) if (real_img > 0).any() else 0
        tissue_mask = real_img > max(bg_estimate, 1.0)

        # Find pixels that have tissue but no annotation
        needs_fill = tissue_mask & ~has_annotation
        if not needs_fill.any():
            continue

        # Use distance_transform_edt to find the index of nearest annotated pixel
        # for every unannotated pixel
        # edt with return_indices gives us the coordinates of the nearest foreground pixel
        _, nearest_idx = distance_transform_edt(~has_annotation, return_indices=True)

        # Fill: for each needs_fill pixel, copy the label from nearest annotated pixel
        fill_y, fill_x = np.where(needs_fill)
        source_y = nearest_idx[0][fill_y, fill_x]
        source_x = nearest_idx[1][fill_y, fill_x]
        ann_vol[z, fill_y, fill_x] = ann_vol[z, source_y, source_x]
        filled_count += len(fill_y)

    if filled_count > 0:
        nib.save(
            nib.Nifti1Image(ann_vol, ann_img.affine, ann_img.header),
            str(annotation_path),
        )
        new_nonzero = int(np.count_nonzero(ann_vol))
        new_frac = new_nonzero / ann_vol.size
        print(
            f"[extrapolate] Filled {filled_count:,} tissue pixels with nearest-neighbor labels → "
            f"{new_nonzero:,} nonzero ({new_frac:.1%} coverage)"
        )
    else:
        print("[extrapolate] No gaps to fill (annotation already covers all tissue)")

    return Path(annotation_path)


def _warp_annotation_volume_to_input_space(
    annotation_path: Path,
    reference_volume_path: Path,
    inverse_transforms: list[str],
    output_path: Path,
    *,
    forward_transforms: list[str] | None = None,
    ants_result_path: Path | None = None,
) -> Path:
    """Warp atlas annotation into input-volume space.

    Tries multiple strategies in order:
    1. ANTs inverse transforms with nearestNeighbor interpolation
       (avoids the genericLabel coverage loss with spacing mismatch)
    2. ANTs forward transforms with whichtoinvert + nearestNeighbor
       (alternative warp direction that can sometimes fill better)
    3. Direct Z-mapping fallback from the ANTs registered result
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    warped_ok = False
    best_nonzero = 0

    # --- Strategy 1: Inverse transforms + nearestNeighbor ---
    if inverse_transforms and not warped_ok:
        try:
            ants = importlib.import_module("ants")
            annotation_img = ants.image_read(str(annotation_path))
            reference_img = ants.image_read(str(reference_volume_path))
            _inv = _fix_transform_extensions(inverse_transforms)

            warped = ants.apply_transforms(
                fixed=reference_img,
                moving=annotation_img,
                transformlist=_inv,
                interpolator="nearestNeighbor",
            )
            ants.image_write(warped, str(output_path))
            result_vol = np.asarray(nib.load(str(output_path)).dataobj, dtype=np.int32)
            nonzero, frac = _count_nonzero_and_coverage(
                result_vol, "Strategy 1 (inverse + nearestNeighbor)"
            )
            if frac >= _MIN_COVERAGE_FRACTION:
                warped_ok = True
                best_nonzero = nonzero
                print("[warp] Strategy 1 accepted")
            else:
                best_nonzero = nonzero
                print(
                    f"[warp] Strategy 1 coverage {frac:.1%} < {_MIN_COVERAGE_FRACTION:.0%} threshold, "
                    "trying next strategy"
                )
        except Exception as exc:
            print(f"[warp] Strategy 1 failed ({exc}), trying next strategy")

    # --- Strategy 2: Forward transforms with whichtoinvert + nearestNeighbor ---
    if forward_transforms and not warped_ok:
        try:
            ants = importlib.import_module("ants")
            annotation_img = ants.image_read(str(annotation_path))
            reference_img = ants.image_read(str(reference_volume_path))
            _fwd = _fix_transform_extensions(forward_transforms)

            # Forward transforms map input→template.  To go template→input
            # we apply them with whichtoinvert.  For a typical [warp, affine]
            # list: invert the affine (index 1), keep the warp as-is (index 0)
            # because ANTs automatically uses the inverse warp field.
            invert_flags = [False] * len(_fwd)
            for idx, tf in enumerate(_fwd):
                # Affine / .mat transforms should be inverted
                if tf.endswith(".mat") or tf.endswith(".txt"):
                    invert_flags[idx] = True

            warped = ants.apply_transforms(
                fixed=reference_img,
                moving=annotation_img,
                transformlist=_fwd,
                whichtoinvert=invert_flags,
                interpolator="nearestNeighbor",
            )
            ants.image_write(warped, str(output_path))
            result_vol = np.asarray(nib.load(str(output_path)).dataobj, dtype=np.int32)
            nonzero, frac = _count_nonzero_and_coverage(
                result_vol, "Strategy 2 (forward + whichtoinvert + nearestNeighbor)"
            )
            if frac >= _MIN_COVERAGE_FRACTION and nonzero > best_nonzero:
                warped_ok = True
                best_nonzero = nonzero
                print("[warp] Strategy 2 accepted")
            elif nonzero > best_nonzero:
                # Better than strategy 1 but still below threshold — keep it
                # and let fallback try to improve.
                best_nonzero = nonzero
                print(
                    f"[warp] Strategy 2 coverage {frac:.1%} still below threshold "
                    f"but better than previous ({nonzero:,} > previous), keeping"
                )
            else:
                print(f"[warp] Strategy 2 coverage {frac:.1%} not better, trying fallback")
        except Exception as exc:
            print(f"[warp] Strategy 2 failed ({exc}), trying fallback")

    # --- Strategy 3: Direct Z-mapping fallback ---
    if not warped_ok:
        warped_ok = _direct_z_mapping_fallback(
            annotation_path=annotation_path,
            reference_volume_path=reference_volume_path,
            output_path=output_path,
            ants_result_path=ants_result_path,
        )

    if not warped_ok:
        raise RuntimeError(
            "Failed to warp annotation volume into input space (all strategies failed)"
        )
    return output_path


def _direct_z_mapping_fallback(
    annotation_path: Path,
    reference_volume_path: Path,
    output_path: Path,
    ants_result_path: Path | None = None,
) -> bool:
    """Fallback: directly map annotation slices to input space using Z-offset
    derived from the ANTs-registered result volume.

    For each input Z-slice, finds the corresponding atlas Z-slice.
    The annotation is kept at its NATIVE atlas resolution (not downsampled
    to the tiny input-volume grid) so that brain-region detail is preserved.
    truth_export_3d.py will resize each slice to match the real image later.
    """
    ann_img = nib.load(str(annotation_path))
    ann_vol = np.asarray(ann_img.dataobj, dtype=np.int32)

    ref_img = nib.load(str(reference_volume_path))
    num_input_z = int(ref_img.shape[0])
    num_atlas_z = int(ann_vol.shape[0])
    atlas_yz = (int(ann_vol.shape[1]), int(ann_vol.shape[2]))

    # Determine Z-offset from ANTs registered result (input warped into atlas space)
    z_offset = 0
    if ants_result_path is not None and Path(ants_result_path).exists():
        res_vol = np.asarray(nib.load(str(ants_result_path)).dataobj, dtype=np.float32)
        z_sums = np.array([float(np.sum(res_vol[z])) for z in range(res_vol.shape[0])])
        nonzero_z = np.where(z_sums > 0)[0]
        if len(nonzero_z) > 0:
            z_offset = int(nonzero_z[0])
            print(
                f"[warp-fallback] Z-offset from ANTs result: {z_offset} "
                f"(input spans atlas Z {nonzero_z[0]}..{nonzero_z[-1]})"
            )
    else:
        # Heuristic: centre input within atlas
        z_offset = max(0, (num_atlas_z - num_input_z) // 2)
        print(f"[warp-fallback] No ANTs result available, using centred Z-offset: {z_offset}")

    # Build output volume at native atlas YZ resolution (preserves region detail)
    out_vol = np.zeros((num_input_z, atlas_yz[0], atlas_yz[1]), dtype=np.int32)

    mapped_count = 0
    for i in range(num_input_z):
        atlas_z = z_offset + i
        if 0 <= atlas_z < num_atlas_z:
            ann_slice = ann_vol[atlas_z]
            if int(np.count_nonzero(ann_slice)) > 0:
                out_vol[i] = ann_slice
                mapped_count += 1

    if mapped_count == 0:
        print("[warp-fallback] No annotation slices mapped — fallback failed")
        return False

    print(
        f"[warp-fallback] Mapped {mapped_count}/{num_input_z} slices at native {atlas_yz} resolution "
        f"(atlas Z {z_offset}..{z_offset + num_input_z - 1})"
    )

    # Use the annotation's affine (native atlas coordinate system)
    nib.save(
        nib.Nifti1Image(out_vol, ann_img.affine, ann_img.header),
        str(output_path),
    )
    return True


# ---------------------------------------------------------------------------
# Dual-channel fast path — reuse registration artifacts from a prior channel
# ---------------------------------------------------------------------------
# A typical dual-channel sample has C0 (e.g. 560nm reporter) and C1 (e.g. 640nm
# co-label). Because ANTs registration aligns the *tissue silhouette* (shared
# between channels) to the Allen atlas, running the full 4-hour pipeline a
# second time for C1 is wasted work — the registration, Laplacian refinement,
# and atlas-back-warp are all identical. Only cell detection on the C1 slice
# stack produces channel-specific output.
#
# When ``registration.reuse_from_dir`` points at a completed C0 run's
# outputs_dir, we short-circuit stages 1-5 and run only Quantification (stage 6)
# with C1's merged_slice_paths as the detection source. Net cost for C1 drops
# from ~4h to ~15-30min (pure detection + aggregation).
#
# Required artifacts in the prior dir:
#   ants_registration/annotation_registered.nii.gz
#   ants_registration/fwd_transform_0.nii.gz + fwd_transform_1.mat
#   ants_registration/inv_transform_0.mat + inv_transform_1.nii.gz
#   laplacian_refinement/annotation_refined.nii.gz
#   laplacian_refinement/laplacian_deformation_field.npy
#   truth_export/slice_XXXX_registered_label.tif (one per slice)


_REUSE_REQUIRED_FILES = (
    "ants_registration/annotation_registered.nii.gz",
    "laplacian_refinement/annotation_refined.nii.gz",
    "laplacian_refinement/laplacian_deformation_field.npy",
)


def _reuse_prior_registration_and_quantify(
    prior_dir: Path,
    input_dir: Path,
    outputs_dir: Path,
    merged_slice_paths: list[Path],
    cfg: dict,
    quantify_fn,
    emit,
) -> dict:
    """Reuse a prior channel's registration artifacts and only re-run
    quantification on the current channel's merged_slice_paths.
    """
    prior_dir = Path(prior_dir)
    if not prior_dir.exists():
        raise FileNotFoundError(f"reuse_from_dir does not exist: {prior_dir}")
    missing = [f for f in _REUSE_REQUIRED_FILES if not (prior_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"reuse_from_dir {prior_dir} is missing required artifacts: {missing}"
        )

    emit(
        "Quantification",
        6,
        10,
        f"Reusing registration artifacts from {prior_dir.name}; detecting current channel cells",
        {"reused_from": str(prior_dir)},
    )

    prior_ants_dir = prior_dir / "ants_registration"
    prior_refine_dir = prior_dir / "laplacian_refinement"
    prior_truth_dir = prior_dir / "truth_export"

    # Reconstruct meta dicts the quantifier expects. We only populate the
    # fields that are actually read downstream.
    ants_meta = {
        "registered_volume": prior_ants_dir / "ants_result.nii.gz",
        "forward_transforms": sorted(prior_ants_dir.glob("fwd_transform_*")),
        "inverse_transforms": sorted(prior_ants_dir.glob("inv_transform_*")),
        "metrics_csv": prior_ants_dir / "registration_metrics.csv",
    }
    refine_meta = {
        "final_registered_path": prior_refine_dir / "final_registered.nii.gz",
        "field_path": prior_refine_dir / "laplacian_deformation_field.npy",
        "metrics_csv": prior_refine_dir / "refinement_metrics.csv",
    }
    volume_meta = {
        "volume_path": prior_dir / "volume" / "input_volume.nii.gz",
        "ml_flipped": False,
    }
    template_meta = {
        "template_path": prior_dir / "template_prep" / "template_half.nii.gz",
        "annotation_path": prior_dir / "template_prep" / "annotation_half.nii.gz",
    }

    # Build truth_rows by walking prior truth_export/ and pairing each
    # registered_label.tif with the matching CURRENT-channel slice path so
    # cell detection runs on the new channel's fluorescence. slice_id is the
    # integer index derived from the filename.
    truth_rows = []
    if prior_truth_dir.exists() and merged_slice_paths:
        prior_labels = sorted(prior_truth_dir.glob("slice_*_registered_label.tif"))
        for i, label_path in enumerate(prior_labels):
            if i >= len(merged_slice_paths):
                break
            overlay_path = prior_truth_dir / label_path.name.replace(
                "_registered_label.tif", "_overlay.png"
            )
            truth_rows.append(
                {
                    "slice_id": i,
                    "real_slice_path": str(merged_slice_paths[i]),
                    "registered_label_path": str(label_path),
                    "overlay_path": str(overlay_path) if overlay_path.exists() else "",
                }
            )

    refined_annotation_path = prior_refine_dir / "annotation_refined.nii.gz"
    registered_annotation_path = prior_ants_dir / "annotation_registered.nii.gz"

    quant_meta = run_quantification_from_truth(
        cfg=cfg,
        quantify_fn=quantify_fn,
        input_dir=Path(input_dir),
        outputs_dir=outputs_dir,
        merged_slice_paths=list(merged_slice_paths),
        truth_rows=truth_rows,
        truth_source="3d_registered_volume",
        volume_meta=volume_meta,
        template_meta=template_meta,
        ants_meta=ants_meta,
        refine_meta=refine_meta,
    )
    emit(
        "Quantification",
        6,
        100,
        "Quantification (reused registration) completed",
        {"cells_mapped_csv": str(quant_meta.get("cells_mapped_csv", ""))},
    )

    return {
        "truth_source": "3d_registered_volume",
        "volume_meta": volume_meta,
        "template_meta": template_meta,
        "ants_meta": ants_meta,
        "refine_meta": refine_meta,
        "truth_rows": truth_rows,
        "quant_meta": quant_meta,
        "refined_annotation_path": refined_annotation_path,
        "annotation_registered_path": registered_annotation_path,
        "reused_from": str(prior_dir),
    }


def run_whole_brain_3d(
    cfg: dict,
    input_dir: Path,
    outputs_dir: Path,
    merged_slice_paths: list[Path],
    progress_cb=None,
) -> dict:
    cfg = dict(cfg or {})
    input_cfg = dict(cfg.get("input", {}) or {})
    reg_cfg = dict(cfg.get("registration", {}) or {})
    quantify_fn = cfg.get("quantify_fn")
    if not callable(quantify_fn):
        raise ValueError("cfg['quantify_fn'] must be callable for whole-brain 3D quantification")
    outputs_dir = Path(outputs_dir)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    project_root = Path(__file__).resolve().parents[1]
    stage_count = 6

    def _emit(
        stage_name: str,
        stage_index: int,
        percent: int,
        message: str,
        artifacts: dict | None = None,
    ) -> None:
        write_stage_progress(
            outputs_dir=outputs_dir,
            stage_name=stage_name,
            stage_index=stage_index,
            stage_count=stage_count,
            percent=percent,
            message=message,
            artifacts=artifacts,
        )
        if progress_cb is not None:
            progress_cb(
                stage_name,
                stage_index,
                stage_count,
                percent,
                message,
                dict(artifacts or {}),
            )

    pixel_size_um_xy = float(input_cfg.get("pixel_size_um_xy", 25.0))
    # Auto-correct sub-micron pixel size when images clearly show full brain
    # sections.  A mouse brain half-section is ~8 mm wide; if the declared
    # pixel size yields a field-of-view < 2 mm, the metadata is almost
    # certainly the raw scanner resolution before any downsampling.
    if merged_slice_paths and pixel_size_um_xy < 2.0:
        from tifffile import imread as _tif_rd

        _sample = _tif_rd(str(merged_slice_paths[len(merged_slice_paths) // 2]))
        if _sample.ndim == 3:
            _sample = _sample[0]
        _fov_um = max(_sample.shape) * pixel_size_um_xy
        if _fov_um < 2000:
            # Estimate real pixel size assuming the largest image dimension
            # spans ~8 mm (typical half-brain coronal section).
            _estimated_px = 8000.0 / max(_sample.shape)
            print(
                f"[pixel-size] declared {pixel_size_um_xy} um yields FOV "
                f"{_fov_um:.0f} um — too small for brain section; "
                f"auto-correcting to {_estimated_px:.1f} um"
            )
            pixel_size_um_xy = _estimated_px
    slice_spacing_um = float(input_cfg.get("slice_spacing_um", 25.0))
    slicing_plane = str(input_cfg.get("slicing_plane", "coronal")).lower()
    hemisphere = str(reg_cfg.get("atlas_hemisphere", "left"))
    template_path = Path(
        reg_cfg.get("template_path", "configs/allen_ref_cache/average_template_25.nii.gz")
    )
    if not template_path.is_absolute():
        template_path = project_root / template_path
    annotation_path = Path(reg_cfg.get("annotation_path", "annotation_25.nii.gz"))
    if not annotation_path.is_absolute():
        annotation_path = project_root / annotation_path

    # Fallback: derive a grayscale template from annotation edges when template is missing
    if not template_path.exists() and annotation_path.exists():
        from scipy.ndimage import gaussian_gradient_magnitude

        print(f"[template] {template_path} not found – deriving from annotation edges")
        template_path.parent.mkdir(parents=True, exist_ok=True)
        ann_img = nib.load(str(annotation_path))
        ann_data = np.asarray(ann_img.dataobj, dtype=np.float32)
        # Create edge-based template: non-zero mask + gradient magnitude
        mask = (ann_data > 0).astype(np.float32)
        edges = gaussian_gradient_magnitude(mask, sigma=1.0)
        # Combine: interior brightness + strong edges
        template_data = mask * 0.5 + edges / max(float(edges.max()), 1e-6) * 0.5
        template_data = np.clip(template_data * 65535, 0, 65535).astype(np.uint16)
        tpl_img = nib.Nifti1Image(template_data, ann_img.affine, ann_img.header)
        nib.save(tpl_img, str(template_path))
        print(f"[template] saved derived template to {template_path}")
    ap_start = int(reg_cfg.get("ap_start", 0))
    ap_end = int(reg_cfg.get("ap_end", 528))

    # Auto-calculate AP range from input slice Z numbers when using defaults.
    # This avoids registering against a full 528-slice template when the input
    # only covers a sub-range, which would leave anterior/posterior slices empty.
    if ap_start == 0 and ap_end == 528:
        import re as _re

        z_scale = float(reg_cfg.get("atlas_z_z_scale", 0.2))
        z_offset = int(reg_cfg.get("atlas_z_offset", 0))
        # Always read z-numbers from the ORIGINAL source directory — merged
        # files are renamed to merged_####.tif during staging and no longer
        # carry the "z<digits>" token that the regex below needs. Using the
        # source dir keeps AP auto-compute working for both direct and merged
        # pipeline invocations.
        _s_dir = Path(input_dir)
        _s_glob = str(input_cfg.get("slice_glob", "z*.tif"))
        _s_paths = sorted(_s_dir.glob(_s_glob))
        if _s_paths and reg_cfg.get("atlas_z_from_filename", False):
            _z_nums = []
            for sp in _s_paths:
                m = _re.search(r"z(\d+)", sp.stem)
                if m:
                    _z_nums.append(int(m.group(1)))
            if _z_nums:
                ap_start = max(0, int(min(_z_nums) * z_scale) + z_offset - 10)
                ap_end = min(528, int(max(_z_nums) * z_scale) + z_offset + 10)
                print(
                    f"[AP auto] Computed AP range from input Z[{min(_z_nums)}..{max(_z_nums)}]: "
                    f"atlas AP [{ap_start}, {ap_end}]"
                )

    ants_transform = str(reg_cfg.get("ants_transform", "SyN"))
    random_seed = int(reg_cfg.get("random_seed", 42))
    laplacian_lambda = float(reg_cfg.get("laplacian_lambda", 0.18))
    laplacian_maxiter = int(reg_cfg.get("laplacian_maxiter", 100))
    skip_laplacian = bool(reg_cfg.get("skip_laplacian_refinement", False))
    if merged_slice_paths:
        volume_slice_dir = Path(merged_slice_paths[0]).parent
        volume_glob = "*.tif"
    else:
        volume_slice_dir = Path(input_dir)
        volume_glob = str(input_cfg.get("slice_glob", "z*.tif"))

    volume_path = outputs_dir / "volume" / "input_volume.nii.gz"
    template_dir = outputs_dir / "template_prep"
    ants_dir = outputs_dir / "ants_registration"
    refine_dir = outputs_dir / "laplacian_refinement"
    truth_dir = outputs_dir / "truth_export"
    refined_annotation_path = refine_dir / "annotation_refined.nii.gz"
    registered_annotation_path = ants_dir / "annotation_registered.nii.gz"

    # -----------------------------------------------------------------
    # Dual-channel fast path: reuse registration artifacts from a prior
    # channel's outputs_dir and skip stages 1-5. Only the quantification
    # stage runs, using the *current* channel's slice paths for detection.
    # -----------------------------------------------------------------
    reuse_from_dir = reg_cfg.get("reuse_from_dir")
    if reuse_from_dir:
        reuse_result = _reuse_prior_registration_and_quantify(
            prior_dir=Path(str(reuse_from_dir)),
            input_dir=Path(input_dir),
            outputs_dir=outputs_dir,
            merged_slice_paths=list(merged_slice_paths),
            cfg=cfg,
            quantify_fn=quantify_fn,
            emit=_emit,
        )
        return reuse_result

    _emit(
        "Volume Build",
        1,
        10,
        "Building 3D input volume from merged TIFF slices",
        {"volume_path": str(volume_path)},
    )
    volume_meta = build_volume_from_tiffs(
        slice_dir=volume_slice_dir,
        output_path=volume_path,
        pixel_um_xy=pixel_size_um_xy,
        z_spacing_um=slice_spacing_um,
        glob_pattern=volume_glob,
        xy_downsample_cap=reg_cfg.get("xy_downsample_cap"),
    )
    # Trust the actual path returned by the builder, not the requested path.
    volume_path = Path(volume_meta["volume_path"])

    # If caller passed an empty merged_slice_paths, populate it from the same
    # glob used for volume building so that truth_export receives the real paths.
    if not merged_slice_paths:
        merged_slice_paths = sorted(volume_slice_dir.glob(volume_glob))
        if not merged_slice_paths:
            raise FileNotFoundError(
                f"No {volume_glob} found in {volume_slice_dir} for truth export"
            )

    # Flip the input volume along the ML axis (axis 2) so medial/lateral
    # orientation matches the Allen CCFv3 convention.  Microscope images
    # typically have medial on the LEFT, but the Allen left-hemisphere
    # template has medial on the RIGHT.  The flip is reversed after
    # inverse-warping the annotation back into the original image space.
    ml_flip = bool(reg_cfg.get("ml_flip", False))
    if ml_flip:
        vol_img = nib.load(str(volume_path))
        vol_data = np.asarray(vol_img.dataobj)[:, :, ::-1].copy()
        flipped_path = volume_path.parent / "input_volume_flipped.nii.gz"
        nib.save(nib.Nifti1Image(vol_data, vol_img.affine, vol_img.header), str(flipped_path))
        volume_meta["volume_path"] = flipped_path
        volume_meta["ml_flipped"] = True
        print(f"[volume] ML-flipped volume saved to {flipped_path}")

    _emit(
        "Template Prep",
        2,
        25,
        "Preparing hemisphere-specific template inputs",
        {
            "template_path": str(template_dir / "template_half.nii.gz"),
            "annotation_path": str(template_dir / "annotation_half.nii.gz"),
        },
    )
    template_meta = prepare_half_template_inputs(
        template_path=template_path,
        annotation_path=annotation_path,
        hemisphere=hemisphere,
        ap_start=ap_start,
        ap_end=ap_end,
        out_dir=template_dir,
    )

    # Optional intensity adaptation stage (Phase α of closed-loop plan).
    # Pre-aligns the moving volume's histogram + local contrast to the
    # template so ANTs MI/CC has a stronger cross-modality signal. Opt in
    # via `registration.intensity_adapt.mode`: off|hist_match|clahe|
    # hist_match+clahe. Default off → identity, bit-for-bit backwards
    # compatible. See docs/.../2026-04-16-internal-alignment-closed-loop-plan.md.
    intensity_adapt_cfg = reg_cfg.get("intensity_adapt") or {}
    intensity_adapt_mode = str(intensity_adapt_cfg.get("mode", "off")).strip()
    if intensity_adapt_mode and intensity_adapt_mode != "off":
        try:
            from scripts.intensity_adapter import adapt_intensity
        except ImportError:
            from intensity_adapter import adapt_intensity

        _emit(
            "Intensity Adapt",
            2,
            20,
            f"Applying intensity adaptation (mode={intensity_adapt_mode})",
            {"mode": intensity_adapt_mode},
        )
        _moving_vol_img = nib.load(str(volume_meta["volume_path"]))
        _template_vol_img = nib.load(str(template_meta["template_path"]))
        _moving_arr = np.asarray(_moving_vol_img.dataobj)
        _template_arr = np.asarray(_template_vol_img.dataobj)
        _adapted = adapt_intensity(
            _moving_arr,
            _template_arr,
            mode=intensity_adapt_mode,
            clahe_kernel_size=int(intensity_adapt_cfg.get("clahe_kernel_size", 32)),
            clahe_clip_limit=float(intensity_adapt_cfg.get("clahe_clip_limit", 0.01)),
        )
        adapted_path = outputs_dir / "volume" / "input_volume_adapted.nii.gz"
        adapted_path.parent.mkdir(parents=True, exist_ok=True)
        nib.save(
            nib.Nifti1Image(_adapted, _moving_vol_img.affine, _moving_vol_img.header),
            str(adapted_path),
        )
        volume_meta["intensity_adapted_path"] = adapted_path
        volume_meta["volume_path"] = adapted_path
        print(f"[intensity-adapt] mode={intensity_adapt_mode} written to {adapted_path}")

    # Optional axis alignment stage — vendored from UCI-XuLab-RegTools.
    # Detects the longitudinal fissure in moving + template, fits plane
    # normals via SVD, and pre-rotates the moving volume so ANTs SyN does
    # not waste capacity on global roll/pitch correction. Off by default
    # to stay bit-for-bit compatible with older runs; opt in via
    # `registration.axis_alignment_enabled: true`.
    axis_alignment_enabled = bool(reg_cfg.get("axis_alignment_enabled", False))
    axis_align_dir = outputs_dir / "axis_alignment"
    if axis_alignment_enabled:
        from scipy.ndimage import affine_transform as _scipy_affine_transform

        try:
            from scripts.regtools_align import compute_longitudinal_fissure_alignment
        except ImportError:
            from regtools_align import compute_longitudinal_fissure_alignment

        _emit(
            "Axis Alignment",
            2,
            18,
            "Longitudinal-fissure axis alignment (vendored from RegTools)",
            {"axis_alignment_dir": str(axis_align_dir)},
        )
        axis_align_dir.mkdir(parents=True, exist_ok=True)
        try:
            _moving_vol_img = nib.load(str(volume_meta["volume_path"]))
            _template_vol_img = nib.load(str(template_meta["template_path"]))
            _moving_raw = np.asarray(_moving_vol_img.dataobj).astype(np.float32)
            _template_raw = np.asarray(_template_vol_img.dataobj).astype(np.float32)

            # RegTools' preprocess() clips at max_val=400 (tuned for CCF
            # intensity range). Brainfast volumes are 0–65535 uint16, so
            # rescale to the expected range before handing off.
            def _scale_to_align_range(vol: np.ndarray) -> np.ndarray:
                p99 = float(np.percentile(vol, 99))
                if p99 <= 0:
                    return vol.astype(np.float32)
                return np.clip(vol, 0, p99).astype(np.float32) / p99 * 400.0

            _moving_scaled = _scale_to_align_range(_moving_raw)
            _template_scaled = _scale_to_align_range(_template_raw)

            affine4x4, _mcoef, _tcoef, _mpts, _tpts = compute_longitudinal_fissure_alignment(
                _moving_scaled, _template_scaled
            )
            np.save(str(axis_align_dir / "axisAlignA.npy"), affine4x4)

            # Apply affine (rotation only, pivot at origin) to the original
            # uint16 moving volume so ANTs sees a globally-aligned input.
            rot3x3 = affine4x4[:3, :3]
            offset = affine4x4[:3, 3]
            aligned = _scipy_affine_transform(
                _moving_raw,
                matrix=rot3x3,
                offset=offset,
                order=1,
                mode="constant",
                cval=0.0,
            )
            aligned_u16 = np.clip(aligned, 0, 65535).astype(np.uint16)
            aligned_path = axis_align_dir / "input_volume_aligned.nii.gz"
            nib.save(
                nib.Nifti1Image(aligned_u16, _moving_vol_img.affine, _moving_vol_img.header),
                str(aligned_path),
            )
            volume_meta["axis_aligned_volume_path"] = aligned_path
            volume_meta["volume_path"] = aligned_path  # ANTs consumes the pre-aligned volume
            print(
                f"[axis-align] Applied fissure-based rotation (saved {aligned_path}). "
                f"Moving normal: {_mcoef}; template normal: {_tcoef}."
            )
        except Exception as _ax_err:  # noqa: BLE001 — axis alignment is best-effort
            print(
                f"[axis-align] Skipped: {_ax_err}. Continuing with unaligned volume. "
                "Set registration.axis_alignment_enabled=false to silence this attempt."
            )

    _emit(
        "ANTS Registration",
        3,
        45,
        "Running ANTS whole-brain registration",
        {"metrics_csv": str(ants_dir / "registration_metrics.csv")},
    )
    ants_meta = run_ants_registration(
        fixed_path=Path(template_meta["template_path"]),
        moving_path=Path(volume_meta["volume_path"]),
        out_dir=ants_dir,
        transform=ants_transform,
        random_seed=random_seed,
    )

    if skip_laplacian:
        _emit(
            "Laplacian Refinement",
            4,
            65,
            "Skipping Laplacian refinement (skip_laplacian_refinement=true)",
            {},
        )
        # Create a passthrough: copy ANTs result as "refined", zero displacement field
        refine_dir.mkdir(parents=True, exist_ok=True)
        import shutil as _shutil

        _passthrough_path = refine_dir / "final_registered.nii.gz"
        _shutil.copy2(str(ants_meta["registered_volume"]), str(_passthrough_path))
        _zero_field = np.zeros((3,) + nib.load(str(_passthrough_path)).shape, dtype=np.float32)
        _field_path = refine_dir / "laplacian_deformation_field.npy"
        np.save(str(_field_path), _zero_field)
        refine_meta = {
            "final_registered_path": _passthrough_path,
            "field_path": _field_path,
            "metrics_csv": refine_dir / "refinement_metrics.csv",
        }
        print("[laplacian] Skipped — using ANTs result directly")
    else:
        _emit(
            "Laplacian Refinement",
            4,
            65,
            "Applying Laplacian refinement to the registered volume",
            {"metrics_csv": str(refine_dir / "refinement_metrics.csv")},
        )
        refine_meta = refine_registered_volume(
            fixed_path=Path(template_meta["template_path"]),
            moving_path=Path(ants_meta["registered_volume"]),
            out_dir=refine_dir,
            iterations=laplacian_maxiter,
            lambda_=laplacian_lambda,
        )
    refined_annotation_path = _apply_refinement_field_to_annotation_volume(
        annotation_path=Path(template_meta["annotation_path"]),
        field_path=Path(refine_meta["field_path"]),
        output_path=refined_annotation_path,
    )

    _emit(
        "Truth Export",
        5,
        82,
        "Warping atlas annotations back into input space and exporting truth slices",
        {
            "truth_dir": str(truth_dir),
            "annotation_refined_path": str(refined_annotation_path),
            "annotation_registered_path": str(registered_annotation_path),
        },
    )
    annotation_registered_path = _warp_annotation_volume_to_input_space(
        annotation_path=refined_annotation_path,
        reference_volume_path=Path(volume_meta["volume_path"]),
        inverse_transforms=list(ants_meta.get("inverse_transforms", [])),
        output_path=registered_annotation_path,
        forward_transforms=list(ants_meta.get("forward_transforms", [])) or None,
        ants_result_path=Path(ants_meta.get("registered_volume", "")),
    )
    # Reverse the ML flip on annotation so it matches original tissue orientation
    if volume_meta.get("ml_flipped", False):
        _ann_img = nib.load(str(annotation_registered_path))
        _ann_data = np.asarray(_ann_img.dataobj)[:, :, ::-1].copy()
        nib.save(
            nib.Nifti1Image(_ann_data, _ann_img.affine, _ann_img.header),
            str(annotation_registered_path),
        )
        print(f"[warp] ML-flip reversed on annotation volume → {annotation_registered_path}")

    # --- Extrapolate annotation to fill tissue regions not covered by half-hemisphere ---
    # The half-hemisphere template is narrower than the tissue section, leaving a
    # ~100-140 pixel gap on the medial side.  We fill it by propagating the nearest
    # annotated label along the ML (X) axis wherever tissue exists.
    annotation_registered_path = _extrapolate_annotation_to_tissue(
        annotation_path=annotation_registered_path,
        input_volume_path=volume_path,  # original (unflipped) volume
        merged_slice_paths=merged_slice_paths,
    )

    atlas_hemisphere = str(reg_cfg.get("atlas_hemisphere", "")).lower().strip()
    # Task 2 — consume learned calibration (fit_mode / edge_smooth_iter /
    # warp_params). Callers inject these via cfg["truth_export"] after
    # resolving the shared tuned JSON; we also accept per-field registration
    # overrides so a config can still pin a choice explicitly.
    truth_cfg = dict(cfg.get("truth_export", {}) or {})
    te_warp_params = dict(truth_cfg.get("warp_params", {}) or {})
    te_fit_mode = str(truth_cfg.get("fit_mode", "cover")).strip() or "cover"
    te_edge = int(truth_cfg.get("edge_smooth_iter", 0) or 0)
    truth_rows = export_registered_truth_slices(
        real_slice_paths=list(merged_slice_paths),
        annotation_volume_path=annotation_registered_path,
        out_dir=truth_dir,
        pixel_size_um=pixel_size_um_xy,
        slicing_plane=slicing_plane,
        atlas_hemisphere=atlas_hemisphere,
        warp_params=te_warp_params,
        fit_mode=te_fit_mode,
        edge_smooth_iter=te_edge,
    )

    quant_meta = run_quantification_from_truth(
        cfg=cfg,
        quantify_fn=quantify_fn,
        input_dir=Path(input_dir),
        outputs_dir=outputs_dir,
        merged_slice_paths=list(merged_slice_paths),
        truth_rows=truth_rows,
        truth_source="3d_registered_volume",
        volume_meta=volume_meta,
        template_meta=template_meta,
        ants_meta=ants_meta,
        refine_meta=refine_meta,
    )
    _emit(
        "Quantification",
        6,
        100,
        "Quantification against exported truth slices completed",
        {"cells_mapped_csv": str(quant_meta.get("cells_mapped_csv", ""))},
    )

    return {
        "truth_source": "3d_registered_volume",
        "volume_meta": volume_meta,
        "template_meta": template_meta,
        "ants_meta": ants_meta,
        "refine_meta": refine_meta,
        "truth_rows": truth_rows,
        "quant_meta": quant_meta,
        "refined_annotation_path": refined_annotation_path,
        "annotation_registered_path": annotation_registered_path,
    }
