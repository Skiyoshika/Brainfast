"""Smoke test for the default runtime path: scope=whole, whole_brain_backend=miki_3d, primary_model=cpsam.

Exercises the full pipeline flow (volume build -> template prep -> registration ->
truth export -> quantification) using tiny synthetic 3D volumes and mocked ANTs
registration so that the test is lightweight enough for CI.
"""

from __future__ import annotations

import importlib
import sys
import tempfile
from pathlib import Path
from unittest import mock

import nibabel as nib
import numpy as np
import pandas as pd
import pytest
from tifffile import imwrite

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.asset_bootstrap import default_structure_source  # noqa: E402
from scripts.registration_3d_volume import (  # noqa: E402
    build_volume_from_tiffs,
    prepare_half_template_inputs,
)
from scripts.whole_brain_3d import run_whole_brain_3d  # noqa: E402

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NZ = 6  # number of synthetic Z slices
_HW = 32  # spatial height/width of each slice (tiny for speed)
_REGION_ID = 315  # a known Allen CCFv3 region (primary visual area)


def _make_synthetic_slices(slice_dir: Path, n: int = _NZ, hw: int = _HW) -> list[Path]:
    """Write tiny synthetic TIFF slices and return their paths."""
    slice_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(42)
    paths = []
    for i in range(n):
        arr = rng.integers(50, 200, size=(hw, hw), dtype=np.uint16)
        # Burn a bright "cell" blob in the centre
        arr[hw // 2 - 2 : hw // 2 + 3, hw // 2 - 2 : hw // 2 + 3] = 60000
        p = slice_dir / f"z{i:04d}.tif"
        imwrite(str(p), arr)
        paths.append(p)
    return paths


def _make_fake_atlas(atlas_dir: Path, shape: tuple[int, int, int]) -> tuple[Path, Path]:
    """Create minimal NIfTI template + annotation volumes."""
    atlas_dir.mkdir(parents=True, exist_ok=True)
    affine = np.diag([0.025, 0.025, 0.025, 1.0])

    # Template: uniform bright tissue
    tpl_data = np.full(shape, 30000, dtype=np.uint16)
    tpl_path = atlas_dir / "template.nii.gz"
    nib.save(nib.Nifti1Image(tpl_data, affine), str(tpl_path))

    # Annotation: fill the entire volume with a known region id
    ann_data = np.full(shape, _REGION_ID, dtype=np.int32)
    ann_path = atlas_dir / "annotation.nii.gz"
    nib.save(nib.Nifti1Image(ann_data, affine), str(ann_path))

    return tpl_path, ann_path


def _build_mock_ants_result(
    fixed_path: Path,
    moving_path: Path,
    out_dir: Path,
    **_kwargs,
) -> dict:
    """Fake ANTs registration that copies the moving volume as 'registered' output.

    Returns the same dict shape as ``run_ants_registration`` without importing ANTs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ANTs registration warps moving into fixed space, so the result must have
    # the fixed volume's shape.  We fill it with the fixed image data to mimic
    # a perfect registration.
    fixed_img = nib.load(str(fixed_path))
    registered_path = out_dir / "ants_result.nii.gz"
    nib.save(fixed_img, str(registered_path))

    # Create a dummy affine transform file (identity)
    fwd_mat = out_dir / "fwd_transform_0.mat"
    fwd_mat.write_text("identity", encoding="utf-8")
    inv_mat = out_dir / "inv_transform_0.mat"
    inv_mat.write_text("identity", encoding="utf-8")

    # Write a minimal metrics CSV
    metrics_csv = out_dir / "registration_metrics.csv"
    metrics_csv.write_text(
        "metric,value\nNCC,0.9\nNMI,1.5\nSSIM,0.85\nDice,0.8\nMSE,0.01\nPSNR,30.0\n",
        encoding="utf-8",
    )
    summary_txt = out_dir / "registration_summary.txt"
    summary_txt.write_text("mock registration\n", encoding="utf-8")

    return {
        "registered_volume": registered_path,
        "metrics_csv": metrics_csv,
        "summary_txt": summary_txt,
        "forward_transforms": [str(fwd_mat)],
        "inverse_transforms": [str(inv_mat)],
    }


def _fake_quantify(**kwargs) -> dict:
    """Minimal quantification stub that writes a cells_mapped.csv and QC files."""
    outputs_dir = Path(kwargs["outputs_dir"])
    truth_rows = kwargs.get("truth_rows", [])

    # Write a tiny cells_mapped CSV
    cells_csv = outputs_dir / "cells_mapped.csv"
    pd.DataFrame(
        [
            {
                "cell_id": 1,
                "slice_id": 0,
                "x": 16.0,
                "y": 16.0,
                "score": 10.0,
                "region_id": _REGION_ID,
                "region_name": "test_region",
                "mapping_status": "ok",
            }
        ]
    ).to_csv(cells_csv, index=False)

    # Write slice QC
    slice_qc = outputs_dir / "slice_registration_qc.csv"
    rows = []
    for tr in truth_rows:
        rows.append(
            {
                "slice_id": tr["slice_id"],
                "registration_method": "3d_truth_export",
                "score_type": "volume_truth_export",
                "score": 1.0,
            }
        )
    pd.DataFrame(rows).to_csv(slice_qc, index=False)

    # Write volume QC
    vol_qc = outputs_dir / "volume_registration_qc.csv"
    pd.DataFrame(
        [
            {
                "truth_source": "3d_registered_volume",
                "score_type": "volume_truth_export",
                "score": 1.0,
            }
        ]
    ).to_csv(vol_qc, index=False)

    return {
        "cells_mapped_csv": str(cells_csv),
        "truth_source": "3d_registered_volume",
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDefaultRuntimePath:
    """Validate the default miki_3d + cpsam shipped path end-to-end."""

    def test_default_runtime_path_volume_build(self, tmp_path: Path) -> None:
        """Volume build produces a valid NIfTI from tiny synthetic slices."""
        slice_dir = tmp_path / "slices"
        _make_synthetic_slices(slice_dir)
        out_path = tmp_path / "volume.nii.gz"

        meta = build_volume_from_tiffs(
            slice_dir=slice_dir,
            output_path=out_path,
            pixel_um_xy=25.0,
            z_spacing_um=25.0,
            glob_pattern="z*.tif",
        )

        assert out_path.exists()
        vol = nib.load(str(out_path))
        assert vol.shape[0] == _NZ
        assert meta["slice_count"] == _NZ

    def test_default_runtime_path_template_prep(self, tmp_path: Path) -> None:
        """Template preparation crops hemisphere and AP range correctly."""
        atlas_dir = tmp_path / "atlas"
        tpl_path, ann_path = _make_fake_atlas(atlas_dir, shape=(20, _HW, _HW))
        out_dir = tmp_path / "template_prep"

        meta = prepare_half_template_inputs(
            template_path=tpl_path,
            annotation_path=ann_path,
            hemisphere="left",
            ap_start=2,
            ap_end=18,
            out_dir=out_dir,
        )

        assert Path(meta["template_path"]).exists()
        assert Path(meta["annotation_path"]).exists()
        tpl_vol = nib.load(str(meta["template_path"]))
        # AP range 2..18 = 16 slices, left hemisphere = half width
        assert tpl_vol.shape[0] == 16
        assert tpl_vol.shape[2] == _HW // 2

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_default_runtime_path_full_pipeline(self, tmp_path: Path) -> None:
        """Full miki_3d pipeline with mocked ANTs: build -> prep -> reg -> export -> quantify."""
        slice_dir = tmp_path / "slices"
        slice_paths = _make_synthetic_slices(slice_dir)
        outputs_dir = tmp_path / "outputs"

        atlas_dir = tmp_path / "atlas"
        # Atlas shape must accommodate the input: at least _NZ slices in AP,
        # and at least _HW in Y/X so the half-hemisphere crop is valid.
        tpl_path, ann_path = _make_fake_atlas(atlas_dir, shape=(_NZ + 4, _HW, _HW))

        cfg = {
            "input": {
                "pixel_size_um_xy": 25.0,
                "slice_spacing_um": 25.0,
                "slicing_plane": "coronal",
            },
            "registration": {
                "template_path": str(tpl_path),
                "annotation_path": str(ann_path),
                "atlas_hemisphere": "left",
                "ap_start": 0,
                "ap_end": _NZ + 4,
                "ants_transform": "SyN",
                "random_seed": 42,
                "skip_laplacian_refinement": True,
            },
            "detection": {
                "primary_model": "cpsam",
            },
            "dedup": {
                "neighbor_slices": 1,
                "r_xy_um": 6.0,
            },
            "quantify_fn": _fake_quantify,
        }

        # Mock ANTs registration (the only heavy dependency).
        # We also need to prevent _warp_annotation_volume_to_input_space from
        # calling ants.apply_transforms.  We selectively block only "ants"
        # imports through importlib.import_module, letting all other modules
        # (e.g. skimage sub-modules via lazy_loader) pass through normally.
        _real_import_module = importlib.import_module

        def _selective_import(name, *args, **kwargs):
            if name == "ants":
                raise ImportError("mocked: ants not available")
            return _real_import_module(name, *args, **kwargs)

        with (
            mock.patch(
                "scripts.whole_brain_3d.run_ants_registration",
                side_effect=_build_mock_ants_result,
            ),
            mock.patch(
                "scripts.whole_brain_3d.importlib.import_module",
                side_effect=_selective_import,
            ),
        ):
            result = run_whole_brain_3d(
                cfg=cfg,
                input_dir=slice_dir,
                outputs_dir=outputs_dir,
                merged_slice_paths=slice_paths,
            )

        # --- Verify pipeline flow ---

        # 1. Volume build artefact
        vol_path = Path(result["volume_meta"]["volume_path"])
        assert vol_path.exists(), "Volume NIfTI must be written"
        vol = nib.load(str(vol_path))
        assert vol.shape[0] == _NZ

        # 2. Template prep artefact
        assert Path(result["template_meta"]["template_path"]).exists()
        assert Path(result["template_meta"]["annotation_path"]).exists()

        # 3. (Mocked) ANTs registration artefact
        assert Path(result["ants_meta"]["registered_volume"]).exists()

        # 4. Truth export produced per-slice rows
        truth_rows = result["truth_rows"]
        assert len(truth_rows) == _NZ, f"Expected {_NZ} truth rows, got {len(truth_rows)}"
        for row in truth_rows:
            assert Path(row["registered_label_path"]).exists()
            assert Path(row["overlay_path"]).exists()

        # 5. Quantification output
        assert result["truth_source"] == "3d_registered_volume"
        quant = result["quant_meta"]
        assert Path(quant["cells_mapped_csv"]).exists()

        # 6. Progress file written by _emit
        progress_file = outputs_dir / "pipeline_progress.json"
        assert progress_file.exists(), "pipeline_progress.json must be written"

    def test_default_runtime_path_detect_import(self) -> None:
        """detect.py resolves cpsam model type correctly (no actual model load)."""
        from scripts.detect import _is_cellpose_model, _resolve_model_type

        assert _resolve_model_type("cpsam") == "cpsam"
        assert _resolve_model_type("Cellpose-SAM") == "cpsam"
        assert _resolve_model_type("") == "cpsam"  # default
        assert _is_cellpose_model("cpsam")
        assert _is_cellpose_model("sam")

    def test_default_runtime_path_detect_fallback(self, tmp_path: Path) -> None:
        """detect_cells with fallback_model=threshold runs on synthetic data."""
        from scripts.detect import detect_cells

        slice_dir = tmp_path / "slices"
        paths = _make_synthetic_slices(slice_dir, n=1)

        cfg = {
            "input": {"pixel_size_um_xy": 25.0},
            "compute": {"device": "cpu"},
            "detection": {
                "primary_model": "disabled",
                "secondary_model": "disabled",
                "fallback_model": "threshold",
                "fallback_threshold": 50000.0,
                "fallback_min_distance": 4,
                "within_slice_dedup_px": 2.0,
            },
        }

        detections = detect_cells(paths[0], cfg)
        assert isinstance(detections, pd.DataFrame)
        assert len(detections) > 0, "Should detect the bright blob in synthetic data"
        assert "x" in detections.columns
        assert "y" in detections.columns
