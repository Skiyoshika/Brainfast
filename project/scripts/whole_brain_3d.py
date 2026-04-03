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
except Exception:
    from laplacian_refine_3d import refine_registered_volume
    from pipeline_progress import write_stage_progress
    from registration_3d_ants import run_ants_registration
    from registration_3d_volume import build_volume_from_tiffs, prepare_half_template_inputs
    from truth_export_3d import export_registered_truth_slices


def run_quantification_from_truth(**kwargs) -> dict:
    return kwargs["quantify_fn"](**kwargs)


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


def _warp_annotation_volume_to_input_space(
    annotation_path: Path,
    reference_volume_path: Path,
    inverse_transforms: list[str],
    output_path: Path,
) -> Path:
    if not inverse_transforms:
        raise ValueError("inverse transforms are required to warp annotation volume into input space")

    ants = importlib.import_module("ants")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    annotation_img = ants.image_read(str(annotation_path))
    reference_img = ants.image_read(str(reference_volume_path))
    warped = ants.apply_transforms(
        fixed=reference_img,
        moving=annotation_img,
        transformlist=list(inverse_transforms),
        interpolator="genericLabel",
    )
    ants.image_write(warped, str(output_path))
    return output_path


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
    slice_spacing_um = float(input_cfg.get("slice_spacing_um", 25.0))
    slicing_plane = str(input_cfg.get("slicing_plane", "coronal")).lower()
    hemisphere = str(reg_cfg.get("atlas_hemisphere", "left"))
    template_path = Path(reg_cfg.get("template_path", "configs/allen_ref_cache/average_template_25.nii.gz"))
    if not template_path.is_absolute():
        template_path = project_root / template_path
    annotation_path = Path(reg_cfg.get("annotation_path", "annotation_25.nii.gz"))
    if not annotation_path.is_absolute():
        annotation_path = project_root / annotation_path
    ap_start = int(reg_cfg.get("ap_start", 0))
    ap_end = int(reg_cfg.get("ap_end", 528))
    ants_transform = str(reg_cfg.get("ants_transform", "SyN"))
    random_seed = int(reg_cfg.get("random_seed", 42))
    laplacian_lambda = float(reg_cfg.get("laplacian_lambda", 0.18))
    laplacian_maxiter = int(reg_cfg.get("laplacian_maxiter", 100))
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
    )

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
    )
    truth_rows = export_registered_truth_slices(
        real_slice_paths=list(merged_slice_paths),
        annotation_volume_path=annotation_registered_path,
        out_dir=truth_dir,
        pixel_size_um=pixel_size_um_xy,
        slicing_plane=slicing_plane,
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
