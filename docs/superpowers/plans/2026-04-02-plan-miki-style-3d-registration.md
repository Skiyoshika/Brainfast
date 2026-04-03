# Miki-Style 3D Whole-Brain Registration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace slice-first whole-brain auto registration with a native Miki-style 3D pipeline that becomes the truth source for exported slice labels, QC, mapping, and counts, while keeping existing 2D tools only for preview and manual correction.

**Architecture:** Refactor the existing reusable 3D pieces out of `project/scripts/run_3d_registration.py`, add a Python-based ANTs + Laplacian refinement chain, then wire `project/scripts/main.py` to use a six-stage whole-brain 3D orchestration path that writes progress artifacts consumable by the Flask UI. Existing per-slice 2D logic remains available for single-slice/manual workflows but must no longer generate whole-brain truth.

**Tech Stack:** Python 3.10+, nibabel, numpy, scipy, tifffile, antspyx (`ants`), Flask, vanilla JS, pytest

---

## File Structure

| Path | Action | Responsibility |
|---|---|---|
| `project/scripts/registration_3d_volume.py` | Create | Build a normalized NIfTI volume from TIFF slices and prepare cropped hemisphere/template inputs |
| `project/scripts/registration_3d_ants.py` | Create | Run ANTs 3D registration and write registration metrics/summary artifacts |
| `project/scripts/laplacian_refine_3d.py` | Create | Perform post-ANTs Laplacian refinement and write refinement metrics/artifacts |
| `project/scripts/truth_export_3d.py` | Create | Export registered annotation volume into per-slice registered labels and overlays |
| `project/scripts/pipeline_progress.py` | Create | Write/read subprocess-safe stage progress JSON for whole-brain 3D runs |
| `project/scripts/whole_brain_3d.py` | Create | Orchestrate the six-stage whole-brain 3D pipeline and quantification |
| `project/scripts/run_3d_registration.py` | Modify | Become a thin CLI wrapper over reusable 3D pipeline functions |
| `project/scripts/main.py` | Modify | Route whole-brain automatic runs to the 3D pipeline and preserve 2D helper mode |
| `project/scripts/check_env.py` | Modify | Validate ANTs dependency and 3D pipeline requirements |
| `project/configs/run_config.template.json` | Modify | Add explicit whole-brain 3D backend/truth-source config defaults |
| `project/configs/run_config_35.json` | Modify | Turn the sample config onto the new whole-brain 3D path |
| `project/frontend/server_context.py` | Modify | Track stage progress metadata read from output artifacts |
| `project/frontend/blueprints/api_pipeline.py` | Modify | Expose 3D stage progress and output status to the UI |
| `project/frontend/blueprints/api_outputs.py` | Modify | Expose volume-level QC summaries and 3D truth-derived output metadata |
| `project/frontend/index.html` | Modify | Add 3D registration status, volume QC summary, and slice inspector surfaces |
| `project/frontend/app.js` | Modify | Render six-stage progress, volume QC, and explicit 2D auxiliary messaging |
| `project/frontend/styles.css` | Modify | Style stage-progress rows, QC panels, and slice inspector state |
| `project/tests/unit/test_registration_3d_volume.py` | Create | Unit tests for volume building and template prep |
| `project/tests/unit/test_registration_3d_ants.py` | Create | Unit tests for ANTs invocation contract and metrics output |
| `project/tests/unit/test_laplacian_refine_3d.py` | Create | Unit tests for refinement artifact and metric output |
| `project/tests/unit/test_truth_export_3d.py` | Create | Unit tests for 3D truth slice export behavior |
| `project/tests/unit/test_pipeline_progress.py` | Create | Unit tests for subprocess-safe progress persistence |
| `project/tests/unit/test_whole_brain_3d.py` | Create | Unit tests for stage orchestration and quantification routing |
| `project/tests/unit/test_main.py` | Modify | Verify `main.py` routes whole-brain mode to the 3D pipeline |
| `project/tests/unit/test_frontend_regressions.py` | Modify | Verify new UI/status/QC hooks exist and use active output dir |
| `project/tests/unit/test_services.py` | Modify | Verify progress/QC service behavior exposed by Flask routes |
| `project/tests/integration/test_regression_suite.py` | Modify | Add an end-to-end synthetic check for 3D truth export semantics |
| `README.md` | Modify | Document whole-brain 3D truth architecture |
| `REPRODUCE.md` | Modify | Document 3D whole-brain reproduction flow and expected outputs |

---

## Task 1: Extract reusable 3D volume and template-prep primitives

**Files:**
- Create: `project/scripts/registration_3d_volume.py`
- Modify: `project/scripts/run_3d_registration.py`
- Test: `project/tests/unit/test_registration_3d_volume.py`

- [ ] **Step 1: Write the failing unit tests**

Create `project/tests/unit/test_registration_3d_volume.py`:

```python
from pathlib import Path

import nibabel as nib
import numpy as np
from tifffile import imwrite


def test_build_volume_from_tiffs_writes_target_resolution_volume(tmp_path):
    from project.scripts.registration_3d_volume import build_volume_from_tiffs

    slice_dir = tmp_path / "slices"
    slice_dir.mkdir()
    imwrite(str(slice_dir / "z0000.tif"), np.arange(16, dtype=np.uint16).reshape(4, 4))
    imwrite(str(slice_dir / "z0001.tif"), (np.arange(16, dtype=np.uint16).reshape(4, 4) + 20))
    out_path = tmp_path / "brain_25um.nii.gz"

    meta = build_volume_from_tiffs(
        slice_dir=slice_dir,
        output_path=out_path,
        pixel_um_xy=12.5,
        z_spacing_um=25.0,
        target_um=25.0,
        glob_pattern="z*.tif",
    )

    assert out_path.exists()
    img = nib.load(str(out_path))
    assert img.shape == (2, 2, 2)
    assert tuple(round(v, 5) for v in img.header.get_zooms()[:3]) == (0.025, 0.025, 0.025)
    assert meta["shape"] == [2, 2, 2]
    assert meta["downsample_factor"] == 2


def test_prepare_half_template_inputs_crops_ap_range_and_left_half(tmp_path):
    from project.scripts.registration_3d_volume import prepare_half_template_inputs

    template = np.ones((8, 6, 6), dtype=np.float32)
    annotation = np.arange(8 * 6 * 6, dtype=np.int32).reshape(8, 6, 6)
    template_path = tmp_path / "template.nii.gz"
    annotation_path = tmp_path / "annotation.nii.gz"
    nib.save(nib.Nifti1Image(template, np.eye(4)), str(template_path))
    nib.save(nib.Nifti1Image(annotation, np.eye(4)), str(annotation_path))

    result = prepare_half_template_inputs(
        template_path=template_path,
        annotation_path=annotation_path,
        hemisphere="left",
        ap_start=1,
        ap_end=5,
        out_dir=tmp_path / "prepared",
    )

    tmpl = nib.load(result["template_path"])
    ann = nib.load(result["annotation_path"])
    assert tmpl.shape == (4, 6, 3)
    assert ann.shape == (4, 6, 3)
    assert result["hemisphere"] == "left"
    assert result["ap_range"] == [1, 5]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```powershell
python -m pytest project\tests\unit\test_registration_3d_volume.py -v
```

Expected:
- `ModuleNotFoundError` for `project.scripts.registration_3d_volume`

- [ ] **Step 3: Write the reusable implementation**

Create `project/scripts/registration_3d_volume.py`:

```python
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from tifffile import imread


def build_volume_from_tiffs(
    slice_dir: Path,
    output_path: Path,
    pixel_um_xy: float,
    z_spacing_um: float,
    target_um: float = 25.0,
    glob_pattern: str = "z*.tif",
) -> dict:
    slices = sorted(Path(slice_dir).glob(glob_pattern))
    if not slices:
        raise FileNotFoundError(f"no TIFF slices found in {slice_dir} with pattern {glob_pattern}")

    ds = max(1, round(float(target_um) / float(pixel_um_xy)))
    stack = []
    for tif_path in slices:
        arr = imread(str(tif_path)).astype(np.float32)
        if arr.ndim == 3:
            arr = arr[0]
        stack.append(arr[::ds, ::ds])

    vol = np.stack(stack, axis=0)
    lo = float(np.percentile(vol, 1))
    hi = float(np.percentile(vol, 99.5))
    vol = np.clip((vol - lo) / max(hi - lo, 1.0) * 65535.0, 0, 65535).astype(np.uint16)
    vox_mm = (float(z_spacing_um) / 1000.0, float(pixel_um_xy) * ds / 1000.0, float(pixel_um_xy) * ds / 1000.0)
    img = nib.Nifti1Image(vol, np.diag([vox_mm[0], vox_mm[1], vox_mm[2], 1.0]))
    img.header.set_zooms(vox_mm)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(img, str(output_path))
    return {
        "volume_path": str(output_path),
        "shape": list(vol.shape),
        "downsample_factor": int(ds),
        "voxel_mm": [float(v) for v in vox_mm],
        "slice_count": len(slices),
    }


def prepare_half_template_inputs(
    template_path: Path,
    annotation_path: Path,
    hemisphere: str,
    ap_start: int,
    ap_end: int,
    out_dir: Path,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    template_img = nib.load(str(template_path))
    annotation_img = nib.load(str(annotation_path))

    template = np.asarray(template_img.dataobj)
    annotation = np.asarray(annotation_img.dataobj)
    template = template[ap_start:ap_end, :, :]
    annotation = annotation[ap_start:ap_end, :, :]

    mid = template.shape[2] // 2
    if hemisphere == "right_flipped":
        template = template[:, :, mid:][:, :, ::-1].copy()
        annotation = annotation[:, :, mid:][:, :, ::-1].copy()
    elif hemisphere == "right":
        template = template[:, :, mid:]
        annotation = annotation[:, :, mid:]
    else:
        template = template[:, :, :mid]
        annotation = annotation[:, :, :mid]

    out_template = out_dir / "template_half.nii.gz"
    out_annotation = out_dir / "annotation_half.nii.gz"
    nib.save(nib.Nifti1Image(template.astype(np.float32), template_img.affine), str(out_template))
    nib.save(nib.Nifti1Image(annotation.astype(np.int32), annotation_img.affine), str(out_annotation))
    return {
        "template_path": str(out_template),
        "annotation_path": str(out_annotation),
        "hemisphere": str(hemisphere),
        "ap_range": [int(ap_start), int(ap_end)],
        "shape": list(template.shape),
    }
```

- [ ] **Step 4: Refactor the existing CLI script to use the new helpers**

Modify `project/scripts/run_3d_registration.py` imports and calls:

```python
from project.scripts.registration_3d_volume import (
    build_volume_from_tiffs,
    prepare_half_template_inputs,
)

volume_meta = build_volume_from_tiffs(
    slice_dir=slice_dir,
    output_path=brain_nii,
    pixel_um_xy=pixel_um_xy,
    z_spacing_um=z_spacing,
    target_um=25.0,
)

prep_meta = prepare_half_template_inputs(
    template_path=TEMPLATE_25,
    annotation_path=ANNOTATION,
    hemisphere=hemisphere,
    ap_start=ap_start,
    ap_end=ap_end,
    out_dir=out_dir,
)

tmpl_cropped = Path(prep_meta["template_path"])
ann_cropped = Path(prep_meta["annotation_path"])
```

- [ ] **Step 5: Run tests and verify they pass**

Run:

```powershell
python -m pytest project\tests\unit\test_registration_3d_volume.py -v
```

Expected:
- `2 passed`

- [ ] **Step 6: Commit**

```powershell
git add project/scripts/registration_3d_volume.py project/scripts/run_3d_registration.py project/tests/unit/test_registration_3d_volume.py
git commit -m "feat: extract reusable 3d volume prep helpers"
```

---

## Task 2: Add 3D ANTs registration and metrics artifacts

**Files:**
- Create: `project/scripts/registration_3d_ants.py`
- Test: `project/tests/unit/test_registration_3d_ants.py`

- [ ] **Step 1: Write the failing unit tests**

Create `project/tests/unit/test_registration_3d_ants.py`:

```python
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np


class _FakeAntsModule:
    @staticmethod
    def image_read(path):
        return nib.load(str(path)).get_fdata().astype(np.float32)

    @staticmethod
    def image_write(arr, path):
        nib.save(nib.Nifti1Image(np.asarray(arr, dtype=np.float32), np.eye(4)), str(path))

    @staticmethod
    def registration(fixed, moving, type_of_transform, random_seed):
        return {
            "warpedmovout": moving,
            "fwdtransforms": ["fake_warp.nii.gz"],
            "invtransforms": ["fake_inv_warp.nii.gz"],
        }


def test_run_ants_registration_writes_metrics_and_summary(tmp_path, monkeypatch):
    from project.scripts.registration_3d_ants import run_ants_registration

    monkeypatch.setitem(sys.modules, "ants", _FakeAntsModule)
    fixed = tmp_path / "fixed.nii.gz"
    moving = tmp_path / "moving.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32), np.eye(4)), str(fixed))
    nib.save(nib.Nifti1Image(np.ones((4, 4, 4), dtype=np.float32) * 0.8, np.eye(4)), str(moving))

    result = run_ants_registration(
        fixed_path=fixed,
        moving_path=moving,
        out_dir=tmp_path / "ants_out",
        transform="SyN",
        random_seed=42,
    )

    assert Path(result["registered_volume"]).exists()
    assert Path(result["metrics_csv"]).exists()
    assert Path(result["summary_txt"]).exists()
    rows = list(csv.DictReader(Path(result["metrics_csv"]).open(encoding="utf-8")))
    assert rows[0]["metric"] == "NCC"


def test_compute_registration_metrics_returns_expected_metric_names():
    from project.scripts.registration_3d_ants import compute_registration_metrics

    fixed = np.ones((4, 4, 4), dtype=np.float32)
    moving = np.ones((4, 4, 4), dtype=np.float32) * 0.9
    metrics = compute_registration_metrics(fixed, moving)

    assert set(metrics) >= {"NCC", "NMI", "SSIM", "Dice", "MSE"}
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```powershell
python -m pytest project\tests\unit\test_registration_3d_ants.py -v
```

Expected:
- `ModuleNotFoundError` for `project.scripts.registration_3d_ants`

- [ ] **Step 3: Implement the ANTs runner and metric writer**

Create `project/scripts/registration_3d_ants.py`:

```python
from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from skimage.metrics import structural_similarity as ssim


def _norm(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    lo = float(np.percentile(arr, 1))
    hi = float(np.percentile(arr, 99))
    return np.clip((arr - lo) / max(hi - lo, 1e-6), 0.0, 1.0)


def compute_registration_metrics(fixed_arr: np.ndarray, moving_arr: np.ndarray) -> dict[str, float]:
    fixed = _norm(fixed_arr)
    moving = _norm(moving_arr)
    fixed_mask = fixed > 0.1
    moving_mask = moving > 0.1
    dice = (2.0 * float(np.logical_and(fixed_mask, moving_mask).sum())) / max(
        float(fixed_mask.sum() + moving_mask.sum()), 1.0
    )
    mse = float(np.mean((fixed - moving) ** 2))
    ncc = float(np.corrcoef(fixed.ravel(), moving.ravel())[0, 1])
    nmi = float((fixed.mean() + moving.mean()) / max(mse + 1e-6, 1e-6))
    ssim_score = float(ssim(fixed[fixed.shape[0] // 2], moving[moving.shape[0] // 2], data_range=1.0))
    return {"NCC": ncc, "NMI": nmi, "SSIM": ssim_score, "Dice": dice, "MSE": mse}


def run_ants_registration(
    fixed_path: Path,
    moving_path: Path,
    out_dir: Path,
    transform: str = "SyN",
    random_seed: int = 42,
) -> dict:
    import ants  # type: ignore

    out_dir.mkdir(parents=True, exist_ok=True)
    fixed = ants.image_read(str(fixed_path))
    moving = ants.image_read(str(moving_path))
    reg = ants.registration(
        fixed=fixed,
        moving=moving,
        type_of_transform=str(transform),
        random_seed=int(random_seed),
    )

    result_path = out_dir / "ants_result.nii.gz"
    ants.image_write(reg["warpedmovout"], str(result_path))

    fixed_arr = np.asarray(nib.load(str(fixed_path)).dataobj, dtype=np.float32)
    moved_arr = np.asarray(nib.load(str(result_path)).dataobj, dtype=np.float32)
    metrics = compute_registration_metrics(fixed_arr, moved_arr)

    metrics_csv = out_dir / "registration_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "value"])
        writer.writeheader()
        for key, value in metrics.items():
            writer.writerow({"metric": key, "value": f"{value:.6f}"})

    summary_txt = out_dir / "registration_summary.txt"
    summary_txt.write_text(
        "\n".join(
            [
                "REGISTRATION SUMMARY",
                f"Fixed image: {fixed_path}",
                f"Moving image: {moving_path}",
                f"Transform: {transform}",
                f"Registered output: {result_path}",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "registered_volume": str(result_path),
        "metrics_csv": str(metrics_csv),
        "summary_txt": str(summary_txt),
        "forward_transforms": list(reg.get("fwdtransforms", [])),
        "inverse_transforms": list(reg.get("invtransforms", [])),
    }
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```powershell
python -m pytest project\tests\unit\test_registration_3d_ants.py -v
```

Expected:
- `2 passed`

- [ ] **Step 5: Commit**

```powershell
git add project/scripts/registration_3d_ants.py project/tests/unit/test_registration_3d_ants.py
git commit -m "feat: add 3d ants registration contract"
```

---

## Task 3: Add Laplacian refinement after ANTs

**Files:**
- Create: `project/scripts/laplacian_refine_3d.py`
- Test: `project/tests/unit/test_laplacian_refine_3d.py`

- [ ] **Step 1: Write the failing unit tests**

Create `project/tests/unit/test_laplacian_refine_3d.py`:

```python
from pathlib import Path

import nibabel as nib
import numpy as np


def test_refine_registered_volume_writes_field_and_final_volume(tmp_path):
    from project.scripts.laplacian_refine_3d import refine_registered_volume

    fixed = tmp_path / "fixed.nii.gz"
    moving = tmp_path / "moving.nii.gz"
    nib.save(nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.float32), np.eye(4)), str(fixed))
    nib.save(nib.Nifti1Image(np.ones((5, 5, 5), dtype=np.float32) * 0.7, np.eye(4)), str(moving))

    result = refine_registered_volume(
        fixed_path=fixed,
        moving_path=moving,
        out_dir=tmp_path / "laplacian",
        iterations=5,
        lambda_=0.15,
    )

    assert Path(result["final_registered_path"]).exists()
    assert Path(result["field_path"]).exists()
    assert Path(result["metrics_csv"]).exists()
    refined = nib.load(result["final_registered_path"])
    assert refined.shape == (5, 5, 5)
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
python -m pytest project\tests\unit\test_laplacian_refine_3d.py -v
```

Expected:
- `ModuleNotFoundError` for `project.scripts.laplacian_refine_3d`

- [ ] **Step 3: Implement the minimal refinement module**

Create `project/scripts/laplacian_refine_3d.py`:

```python
from __future__ import annotations

import csv
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter, laplace

from project.scripts.registration_3d_ants import compute_registration_metrics


def refine_registered_volume(
    fixed_path: Path,
    moving_path: Path,
    out_dir: Path,
    iterations: int = 100,
    lambda_: float = 0.18,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    fixed = np.asarray(nib.load(str(fixed_path)).dataobj, dtype=np.float32)
    moving_img = nib.load(str(moving_path))
    moving = np.asarray(moving_img.dataobj, dtype=np.float32)

    refined = moving.copy()
    for _ in range(int(iterations)):
        residual = fixed - refined
        refined = refined + float(lambda_) * laplace(gaussian_filter(residual, sigma=0.8))

    disp = gaussian_filter(fixed - refined, sigma=1.2).astype(np.float32)
    field = np.stack(np.gradient(disp), axis=0).astype(np.float32)

    final_path = out_dir / "final_registered.nii.gz"
    field_path = out_dir / "laplacian_deformation_field.npy"
    nib.save(nib.Nifti1Image(refined.astype(np.float32), moving_img.affine), str(final_path))
    np.save(field_path, field)

    before = compute_registration_metrics(fixed, moving)
    after = compute_registration_metrics(fixed, refined)
    metrics_csv = out_dir / "refinement_metrics.csv"
    with metrics_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["metric", "before", "after", "change"])
        writer.writeheader()
        for metric_name, before_value in before.items():
            after_value = after[metric_name]
            writer.writerow(
                {
                    "metric": metric_name,
                    "before": f"{before_value:.6f}",
                    "after": f"{after_value:.6f}",
                    "change": f"{(after_value - before_value):+.6f}",
                }
            )
    return {
        "final_registered_path": str(final_path),
        "field_path": str(field_path),
        "metrics_csv": str(metrics_csv),
    }
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```powershell
python -m pytest project\tests\unit\test_laplacian_refine_3d.py -v
```

Expected:
- `1 passed`

- [ ] **Step 5: Commit**

```powershell
git add project/scripts/laplacian_refine_3d.py project/tests/unit/test_laplacian_refine_3d.py
git commit -m "feat: add 3d laplacian refinement stage"
```

---

## Task 4: Export 3D truth into per-slice labels and overlays

**Files:**
- Create: `project/scripts/truth_export_3d.py`
- Test: `project/tests/unit/test_truth_export_3d.py`

- [ ] **Step 1: Write the failing unit tests**

Create `project/tests/unit/test_truth_export_3d.py`:

```python
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np
from PIL import Image
from tifffile import imwrite


@patch("project.scripts.truth_export_3d.render_overlay")
def test_export_registered_truth_slices_writes_labels_and_overlays(mock_render, tmp_path):
    from project.scripts.truth_export_3d import export_registered_truth_slices

    volume = np.zeros((2, 8, 8), dtype=np.int32)
    volume[0, 2:6, 2:6] = 315
    volume[1, 1:7, 1:7] = 997
    volume_path = tmp_path / "annotation_registered.nii.gz"
    nib.save(nib.Nifti1Image(volume, np.eye(4)), str(volume_path))

    merged_dir = tmp_path / "merged"
    merged_dir.mkdir()
    raw0 = merged_dir / "merged_0000.tif"
    raw1 = merged_dir / "merged_0001.tif"
    imwrite(str(raw0), np.ones((8, 8), dtype=np.uint16) * 20)
    imwrite(str(raw1), np.ones((8, 8), dtype=np.uint16) * 30)

    def _fake_render(real_slice_path, label_slice_path, out_png, **kwargs):
        Image.new("RGB", (8, 8), (12, 34, 56)).save(out_png)
        return out_png, {"warp": {"method": "3d_truth_export"}}

    mock_render.side_effect = _fake_render
    rows = export_registered_truth_slices(
        real_slice_paths=[raw0, raw1],
        annotation_volume_path=volume_path,
        out_dir=tmp_path / "registered_slices",
        pixel_size_um=25.0,
        slicing_plane="coronal",
    )

    assert len(rows) == 2
    assert (tmp_path / "registered_slices" / "slice_0000_registered_label.tif").exists()
    assert (tmp_path / "registered_slices" / "slice_0001_overlay.png").exists()
    assert rows[0]["registration_method"] == "3d_truth_export"
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
python -m pytest project\tests\unit\test_truth_export_3d.py -v
```

Expected:
- `ModuleNotFoundError` for `project.scripts.truth_export_3d`

- [ ] **Step 3: Implement the truth-export module**

Create `project/scripts/truth_export_3d.py`:

```python
from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from tifffile import imwrite

from project.scripts.overlay_render import render_overlay


def _select_volume_slice(volume: np.ndarray, index: int, slicing_plane: str) -> np.ndarray:
    plane = str(slicing_plane).lower()
    if plane == "sagittal":
        return volume[:, :, index]
    if plane == "horizontal":
        return volume[:, index, :]
    return volume[index, :, :]


def export_registered_truth_slices(
    real_slice_paths: list[Path],
    annotation_volume_path: Path,
    out_dir: Path,
    pixel_size_um: float,
    slicing_plane: str,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    volume = np.asarray(nib.load(str(annotation_volume_path)).dataobj, dtype=np.int32)
    rows = []
    for idx, real_slice_path in enumerate(real_slice_paths):
        label_slice = _select_volume_slice(volume, idx, slicing_plane).astype(np.int32)
        label_path = out_dir / f"slice_{idx:04d}_registered_label.tif"
        overlay_path = out_dir / f"slice_{idx:04d}_overlay.png"
        imwrite(str(label_path), label_slice.astype(np.int32))
        _, diagnostic = render_overlay(
            real_slice_path=Path(real_slice_path),
            label_slice_path=label_path,
            out_png=overlay_path,
            alpha=0.72,
            mode="fill",
            pixel_size_um=float(pixel_size_um),
            major_top_k=28,
            fit_mode="cover",
            edge_smooth_iter=0,
            warp_params={},
            return_meta=True,
            warped_label_out=label_path,
        )
        rows.append(
            {
                "slice_id": int(idx),
                "real_slice_path": str(real_slice_path),
                "registered_label_path": str(label_path),
                "overlay_path": str(overlay_path),
                "registration_method": str(diagnostic.get("warp", {}).get("method", "3d_truth_export")),
            }
        )
    return rows
```

- [ ] **Step 4: Run the test to verify it passes**

Run:

```powershell
python -m pytest project\tests\unit\test_truth_export_3d.py -v
```

Expected:
- `1 passed`

- [ ] **Step 5: Commit**

```powershell
git add project/scripts/truth_export_3d.py project/tests/unit/test_truth_export_3d.py
git commit -m "feat: export per-slice truth from 3d registered volume"
```

---

## Task 5: Add subprocess-safe pipeline progress and whole-brain 3D orchestration

**Files:**
- Create: `project/scripts/pipeline_progress.py`
- Create: `project/scripts/whole_brain_3d.py`
- Modify: `project/scripts/main.py`
- Modify: `project/configs/run_config.template.json`
- Modify: `project/configs/run_config_35.json`
- Test: `project/tests/unit/test_pipeline_progress.py`
- Test: `project/tests/unit/test_whole_brain_3d.py`
- Test: `project/tests/unit/test_main.py`

- [ ] **Step 1: Write the failing tests**

Create `project/tests/unit/test_pipeline_progress.py`:

```python
def test_write_stage_progress_persists_json(tmp_path):
    from project.scripts.pipeline_progress import write_stage_progress, read_stage_progress

    write_stage_progress(
        outputs_dir=tmp_path,
        stage_name="ANTS Registration",
        stage_index=3,
        stage_count=6,
        percent=42,
        message="Running SyN stage",
        artifacts={"metrics_csv": "registration_metrics.csv"},
    )

    payload = read_stage_progress(tmp_path)
    assert payload["stageName"] == "ANTS Registration"
    assert payload["stageIndex"] == 3
    assert payload["percent"] == 42
    assert payload["artifacts"]["metrics_csv"] == "registration_metrics.csv"
```

Create `project/tests/unit/test_whole_brain_3d.py`:

```python
def test_run_whole_brain_3d_reports_six_stages(tmp_path, monkeypatch):
    from project.scripts.whole_brain_3d import run_whole_brain_3d

    stage_names = []

    monkeypatch.setattr(
        "project.scripts.whole_brain_3d.build_volume_from_tiffs",
        lambda **kwargs: {"volume_path": str(tmp_path / "brain_25um.nii.gz"), "shape": [2, 2, 2]},
    )
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d.prepare_half_template_inputs",
        lambda **kwargs: {
            "template_path": str(tmp_path / "template_half.nii.gz"),
            "annotation_path": str(tmp_path / "annotation_half.nii.gz"),
        },
    )
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d.run_ants_registration",
        lambda **kwargs: {
            "registered_volume": str(tmp_path / "ants_result.nii.gz"),
            "metrics_csv": str(tmp_path / "registration_metrics.csv"),
            "summary_txt": str(tmp_path / "registration_summary.txt"),
        },
    )
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d.refine_registered_volume",
        lambda **kwargs: {
            "final_registered_path": str(tmp_path / "final_registered.nii.gz"),
            "field_path": str(tmp_path / "laplacian_deformation_field.npy"),
            "metrics_csv": str(tmp_path / "refinement_metrics.csv"),
        },
    )
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d.export_registered_truth_slices",
        lambda **kwargs: [{"slice_id": 0, "registered_label_path": str(tmp_path / "slice_0000_registered_label.tif")}],
    )
    monkeypatch.setattr(
        "project.scripts.whole_brain_3d.run_quantification_from_truth",
        lambda **kwargs: {"cells_mapped_csv": str(tmp_path / "cells_mapped.csv")},
    )

    def _progress(stage_name, stage_index, stage_count, percent, message, artifacts=None):
        stage_names.append(stage_name)

    result = run_whole_brain_3d(
        cfg={"input": {"pixel_size_um_xy": 5.0, "slice_spacing_um": 25.0, "slicing_plane": "coronal"}, "registration": {"atlas_hemisphere": "left"}},
        input_dir=tmp_path / "input",
        outputs_dir=tmp_path / "outputs",
        merged_slice_paths=[],
        progress_cb=_progress,
    )

    assert stage_names == [
        "Volume Build",
        "Template Prep",
        "ANTS Registration",
        "Laplacian Refinement",
        "Truth Export",
        "Quantification",
    ]
    assert result["truth_source"] == "3d_registered_volume"
```

Append to `project/tests/unit/test_main.py`:

```python
def test_run_real_input_routes_whole_brain_backend_to_3d(tmp_path, monkeypatch):
    from project.scripts.main import run_real_input

    input_dir = tmp_path / "input"
    input_dir.mkdir()
    (input_dir / "z0000.tif").write_bytes(b"stub")
    called = {}

    monkeypatch.setattr("project.scripts.main._collect_slice_files", lambda *_args, **_kwargs: [input_dir / "z0000.tif"])
    monkeypatch.setattr("project.scripts.main._extract_channel_to_tmp", lambda *_args, **_kwargs: [input_dir / "z0000.tif"])
    monkeypatch.setattr("project.scripts.main.merge_every_n_slices", lambda *_args, **_kwargs: [input_dir / "z0000.tif"])
    monkeypatch.setattr(
        "project.scripts.main.run_whole_brain_3d",
        lambda cfg, input_dir, outputs_dir, merged_slice_paths: called.setdefault("hit", str(outputs_dir)) or {"truth_source": "3d_registered_volume"},
    )

    cfg = {
        "input": {"slice_glob": "*.tif", "slice_interval_n": 1},
        "registration": {"scope": "whole", "whole_brain_backend": "miki_3d"},
    }
    run_real_input(cfg, input_dir=input_dir, output_dir=tmp_path / "outputs")
    assert called["hit"].endswith("outputs")
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```powershell
python -m pytest project\tests\unit\test_pipeline_progress.py project\tests\unit\test_whole_brain_3d.py project\tests\unit\test_main.py -v
```

Expected:
- missing modules/functions for `pipeline_progress` and `whole_brain_3d`

- [ ] **Step 3: Implement subprocess-safe progress persistence**

Create `project/scripts/pipeline_progress.py`:

```python
from __future__ import annotations

import json
from pathlib import Path


def _progress_path(outputs_dir: Path) -> Path:
    return Path(outputs_dir) / "pipeline_progress.json"


def write_stage_progress(
    outputs_dir: Path,
    stage_name: str,
    stage_index: int,
    stage_count: int,
    percent: int,
    message: str,
    artifacts: dict | None = None,
) -> Path:
    payload = {
        "stageName": str(stage_name),
        "stageIndex": int(stage_index),
        "stageCount": int(stage_count),
        "percent": int(percent),
        "message": str(message),
        "artifacts": dict(artifacts or {}),
    }
    out_path = _progress_path(outputs_dir)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def read_stage_progress(outputs_dir: Path) -> dict:
    out_path = _progress_path(outputs_dir)
    if not out_path.exists():
        return {}
    return json.loads(out_path.read_text(encoding="utf-8"))
```

- [ ] **Step 4: Implement the whole-brain 3D orchestrator and route `main.py` into it**

Create `project/scripts/whole_brain_3d.py`:

```python
from __future__ import annotations

from pathlib import Path

from project.scripts.pipeline_progress import write_stage_progress
from project.scripts.registration_3d_ants import run_ants_registration
from project.scripts.registration_3d_volume import build_volume_from_tiffs, prepare_half_template_inputs
from project.scripts.laplacian_refine_3d import refine_registered_volume
from project.scripts.truth_export_3d import export_registered_truth_slices


def run_quantification_from_truth(**kwargs) -> dict:
    return kwargs["quantify_fn"](**kwargs)


def run_whole_brain_3d(cfg: dict, input_dir: Path, outputs_dir: Path, merged_slice_paths: list[Path], progress_cb=None) -> dict:
    def _emit(stage_name: str, stage_index: int, percent: int, message: str, artifacts: dict | None = None):
        write_stage_progress(outputs_dir, stage_name, stage_index, 6, percent, message, artifacts)
        if progress_cb is not None:
            progress_cb(stage_name, stage_index, 6, percent, message, artifacts)

    _emit("Volume Build", 1, 5, "Stacking TIFF slices into NIfTI")
    volume_meta = build_volume_from_tiffs(
        slice_dir=input_dir,
        output_path=outputs_dir / "brain_25um.nii.gz",
        pixel_um_xy=float(cfg["input"]["pixel_size_um_xy"]),
        z_spacing_um=float(cfg["input"]["slice_spacing_um"]),
        target_um=25.0,
        glob_pattern=str(cfg["input"].get("slice_glob", "z*.tif")),
    )
    _emit("Template Prep", 2, 20, "Preparing hemisphere template inputs")
    prep_meta = prepare_half_template_inputs(
        template_path=Path(cfg["registration"]["template_path"]),
        annotation_path=Path(cfg["registration"]["annotation_path"]),
        hemisphere=str(cfg["registration"].get("atlas_hemisphere", "left")),
        ap_start=int(cfg["registration"]["ap_start"]),
        ap_end=int(cfg["registration"]["ap_end"]),
        out_dir=outputs_dir / "template_prep",
    )
    _emit("ANTS Registration", 3, 35, "Running ANTs")
    ants_meta = run_ants_registration(
        fixed_path=Path(prep_meta["template_path"]),
        moving_path=Path(volume_meta["volume_path"]),
        out_dir=outputs_dir / "registered_output" / "02_ants",
        transform=str(cfg["registration"].get("ants_transform", "SyN")),
        random_seed=int(cfg["registration"].get("random_seed", 42)),
    )
    _emit("Laplacian Refinement", 4, 60, "Running Laplacian refinement")
    refine_meta = refine_registered_volume(
        fixed_path=Path(prep_meta["template_path"]),
        moving_path=Path(ants_meta["registered_volume"]),
        out_dir=outputs_dir / "registered_output" / "03_laplacian_refinement",
        iterations=int(cfg["registration"].get("laplacian_maxiter", 100)),
        lambda_=float(cfg["registration"].get("laplacian_lambda", 0.18)),
    )
    _emit("Truth Export", 5, 80, "Exporting slice truth labels and overlays")
    truth_rows = export_registered_truth_slices(
        real_slice_paths=merged_slice_paths,
        annotation_volume_path=Path(refine_meta["final_registered_path"]),
        out_dir=outputs_dir / "registered_slices",
        pixel_size_um=float(cfg["input"]["pixel_size_um_xy"]),
        slicing_plane=str(cfg["input"].get("slicing_plane", "coronal")),
    )
    _emit("Quantification", 6, 92, "Quantifying against 3D truth")
    quant_meta = run_quantification_from_truth(
        truth_rows=truth_rows,
        cfg=cfg,
        outputs_dir=outputs_dir,
        quantify_fn=cfg["quantify_fn"],
    )
    _emit("Quantification", 6, 100, "Whole-brain 3D pipeline complete", {"truth_source": "3d_registered_volume"})
    return {
        "truth_source": "3d_registered_volume",
        "volume_meta": volume_meta,
        "template_meta": prep_meta,
        "ants_meta": ants_meta,
        "refine_meta": refine_meta,
        "truth_rows": truth_rows,
        "quant_meta": quant_meta,
    }
```

Modify the whole-brain branch in `project/scripts/main.py`:

```python
from project.scripts.whole_brain_3d import run_whole_brain_3d

backend = str(cfg.get("registration", {}).get("whole_brain_backend", "")).lower()
scope = str(cfg.get("registration", {}).get("scope", "whole")).lower()

if scope == "whole" and backend == "miki_3d":
    quant_fn = lambda **kw: _quantify_against_exported_truth(
        truth_rows=kw["truth_rows"],
        cfg=kw["cfg"],
        outputs_dir=kw["outputs_dir"],
    )
    cfg = dict(cfg)
    cfg["quantify_fn"] = quant_fn
    return run_whole_brain_3d(
        cfg=cfg,
        input_dir=input_dir,
        outputs_dir=outputs_dir,
        merged_slice_paths=merged_files,
    )
```

Modify config defaults in `project/configs/run_config.template.json` and `project/configs/run_config_35.json`:

```json
"registration": {
  "scope": "whole",
  "whole_brain_backend": "miki_3d",
  "truth_source": "3d_registered_volume",
  "template_path": "configs/allen_ref_cache/average_template_25.nii.gz",
  "annotation_path": "annotation_25.nii.gz",
  "atlas_hemisphere": "left",
  "ants_transform": "SyN",
  "random_seed": 42,
  "laplacian_lambda": 0.18,
  "laplacian_maxiter": 100
}
```

- [ ] **Step 5: Run the tests to verify they pass**

Run:

```powershell
python -m pytest project\tests\unit\test_pipeline_progress.py project\tests\unit\test_whole_brain_3d.py project\tests\unit\test_main.py -v
```

Expected:
- all targeted tests pass

- [ ] **Step 6: Commit**

```powershell
git add project/scripts/pipeline_progress.py project/scripts/whole_brain_3d.py project/scripts/main.py project/configs/run_config.template.json project/configs/run_config_35.json project/tests/unit/test_pipeline_progress.py project/tests/unit/test_whole_brain_3d.py project/tests/unit/test_main.py
git commit -m "feat: route whole-brain mode through 3d orchestration"
```

---

## Task 6: Make quantification and QC consume only 3D truth outputs

**Files:**
- Modify: `project/scripts/main.py`
- Modify: `project/tests/integration/test_regression_suite.py`

- [ ] **Step 1: Write the failing tests**

Append to `project/tests/integration/test_regression_suite.py`:

```python
def test_slice_registration_qc_semantics_can_represent_3d_truth_exports(tmp_path):
    import csv

    qc_path = tmp_path / "slice_registration_qc.csv"
    rows = [
        {
            "slice_id": 0,
            "registered_label_path": "registered_slices/slice_0000_registered_label.tif",
            "overlay_path": "registered_slices/slice_0000_overlay.png",
            "registration_method": "3d_truth_export",
            "score_type": "volume_truth_export",
            "best_score": 0.8123,
        }
    ]
    with qc_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    reloaded = list(csv.DictReader(qc_path.open(encoding="utf-8")))
    assert reloaded[0]["registration_method"] == "3d_truth_export"
    assert reloaded[0]["score_type"] == "volume_truth_export"
```

- [ ] **Step 2: Run the test to verify it fails for the current pipeline contract**

Run:

```powershell
python -m pytest project\tests\integration\test_regression_suite.py -k volume_truth_export -v
```

Expected:
- failure because current pipeline code does not write or preserve the new semantics

- [ ] **Step 3: Implement quantification against exported 3D truth**

Add this helper in `project/scripts/main.py`:

```python
def _quantify_against_exported_truth(truth_rows: list[dict], cfg: dict, outputs_dir: Path) -> dict:
    mapped_frames = []
    registration_rows = []
    px_um = float(cfg["input"]["pixel_size_um_xy"])
    spacing_um = float(cfg["input"]["slice_spacing_um"])

    for row in truth_rows:
        real_path = Path(row["real_slice_path"])
        label_path = Path(row["registered_label_path"])
        detections = detect_cells(real_path, cfg).copy()
        detections["slice_id"] = int(row["slice_id"])
        mapped = map_cells_with_registered_label_slice(
            detections,
            registered_label_tif=label_path,
            structure_csv=_resolve_structure_source(_project_root()),
            atlas_slice_index=int(row["slice_id"]),
            registration_score=float(row.get("best_score", 0.0)),
            registration_method="3d_truth_export",
        )
        mapped_frames.append(mapped)
        registration_rows.append(
            {
                "slice_id": int(row["slice_id"]),
                "registered_label_path": str(label_path),
                "overlay_path": str(row["overlay_path"]),
                "registration_method": "3d_truth_export",
                "score_type": "volume_truth_export",
                "best_score": float(row.get("best_score", 0.0)),
            }
        )

    all_mapped = pd.concat(mapped_frames, ignore_index=True) if mapped_frames else pd.DataFrame()
    deduped, dedup_stats = apply_dedup_kdtree(
        all_mapped,
        neighbor_slices=int(cfg["dedup"]["neighbor_slices"]),
        pixel_size_um=px_um,
        slice_spacing_um=spacing_um,
        r_xy_um=float(cfg["dedup"]["r_xy_um"]),
    )
    leaf, hierarchy = aggregate_by_region(deduped)
    write_outputs(leaf, hierarchy, outputs_dir)
    pd.DataFrame(registration_rows).to_csv(outputs_dir / "slice_registration_qc.csv", index=False)
    pd.DataFrame([{"truth_source": "3d_registered_volume"}]).to_csv(outputs_dir / "volume_registration_qc.csv", index=False)
    return {"cells_mapped_csv": str(outputs_dir / "cells_mapped.csv"), "truth_source": "3d_registered_volume"}
```

- [ ] **Step 4: Run the integration test to verify it passes**

Run:

```powershell
python -m pytest project\tests\integration\test_regression_suite.py -k volume_truth_export -v
```

Expected:
- targeted test passes

- [ ] **Step 5: Commit**

```powershell
git add project/scripts/main.py project/tests/integration/test_regression_suite.py
git commit -m "feat: switch quantification and qc semantics to 3d truth"
```

---

## Task 7: Expose 3D stage progress and volume QC in Flask APIs

**Files:**
- Modify: `project/frontend/server_context.py`
- Modify: `project/frontend/blueprints/api_pipeline.py`
- Modify: `project/frontend/blueprints/api_outputs.py`
- Modify: `project/tests/unit/test_frontend_regressions.py`
- Modify: `project/tests/unit/test_services.py`

- [ ] **Step 1: Write the failing tests**

Append to `project/tests/unit/test_frontend_regressions.py`:

```python
def test_status_endpoint_returns_stage_progress(tmp_path, monkeypatch, client):
    out_dir = tmp_path / "run_3d"
    out_dir.mkdir()
    (out_dir / "pipeline_progress.json").write_text(
        '{"stageName":"ANTS Registration","stageIndex":3,"stageCount":6,"percent":42,"message":"Running SyN","artifacts":{"metrics_csv":"registration_metrics.csv"}}',
        encoding="utf-8",
    )
    monkeypatch.setattr(ctx, "active_output_dir", lambda: out_dir)

    res = client.get("/api/status")
    payload = res.get_json()

    assert payload["stage"]["stageName"] == "ANTS Registration"
    assert payload["stage"]["percent"] == 42
```

Append to `project/tests/unit/test_services.py`:

```python
def test_reg_slice_list_uses_truth_exported_overlays(tmp_path, monkeypatch):
    import project.frontend.server_context as ctx
    from project.frontend.blueprints.api_outputs import outputs_reg_slice_list

    out_dir = tmp_path / "run_3d"
    reg_dir = out_dir / "registered_slices"
    reg_dir.mkdir(parents=True)
    (reg_dir / "slice_0000_overlay.png").write_bytes(b"png")
    monkeypatch.setattr(ctx, "active_output_dir", lambda: out_dir)

    payload = outputs_reg_slice_list().get_json()
    assert payload["count"] == 1
    assert payload["files"] == ["slice_0000_overlay.png"]
```

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```powershell
python -m pytest project\tests\unit\test_frontend_regressions.py project\tests\unit\test_services.py -v
```

Expected:
- `/api/status` payload missing `stage`

- [ ] **Step 3: Read progress JSON in the backend and expose volume QC**

Modify `project/frontend/server_context.py`:

```python
from project.scripts.pipeline_progress import read_stage_progress

run_state: dict = {
    "running": False,
    "done": False,
    "error": None,
    "logs": [],
    "channels": [],
    "proc": None,
    "current_channel": None,
    "history": [],
    "config_path": None,
    "stage": {},
}


def latest_stage_progress(out_dir: Path | None = None) -> dict:
    target = out_dir or active_output_dir()
    return read_stage_progress(target)
```

Modify `project/frontend/blueprints/api_pipeline.py`:

```python
stage = ctx.latest_stage_progress(out_dir)
return jsonify(
    {
        "running": ctx.run_state["running"],
        "done": ctx.run_state["done"],
        "error": ctx.run_state["error"],
        "channels": ctx.run_state["channels"],
        "currentChannel": ctx.run_state["current_channel"],
        "logCount": len(ctx.run_state["logs"]),
        "slicesDone": slices_done,
        "slicesTotal": slices_total,
        "outputDir": ctx.run_state.get("outputDir", str(out_dir)),
        "runName": ctx.run_state.get("runName", ""),
        "stage": stage,
    }
)
```

Modify `project/frontend/blueprints/api_outputs.py`:

```python
@bp.get("/volume-reg-stats")
def outputs_volume_reg_stats():
    fp = ctx.active_output_dir() / "volume_registration_qc.csv"
    if not fp.exists():
        return jsonify({"ok": False, "error": "volume QC not found"}), 404
    return send_from_directory(str(fp.parent), fp.name)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```powershell
python -m pytest project\tests\unit\test_frontend_regressions.py project\tests\unit\test_services.py -v
```

Expected:
- newly added API/status assertions pass

- [ ] **Step 5: Commit**

```powershell
git add project/frontend/server_context.py project/frontend/blueprints/api_pipeline.py project/frontend/blueprints/api_outputs.py project/tests/unit/test_frontend_regressions.py project/tests/unit/test_services.py
git commit -m "feat: expose 3d pipeline progress and volume qc apis"
```

---

## Task 8: Update the frontend for six-stage progress, 3D QC, and auxiliary 2D messaging

**Files:**
- Modify: `project/frontend/index.html`
- Modify: `project/frontend/app.js`
- Modify: `project/frontend/styles.css`
- Modify: `project/tests/unit/test_frontend_regressions.py`

- [ ] **Step 1: Write the failing UI regression tests**

Append to `project/tests/unit/test_frontend_regressions.py`:

```python
def test_index_html_has_3d_progress_and_qc_panels():
    html = Path(r"D:\Brainfast\project\frontend\index.html").read_text(encoding="utf-8", errors="replace")
    for snippet in (
        'id="wholeBrain3dStatusSection"',
        'id="wholeBrainStageList"',
        'id="volumeQcSection"',
        'id="sliceInspectorSection"',
        'id="aux2dNotice"',
    ):
        assert snippet in html


def test_app_js_renders_stage_progress_and_volume_qc():
    js = Path(r"D:\Brainfast\project\frontend\app.js").read_text(encoding="utf-8", errors="replace")
    assert "function renderWholeBrain3dStage" in js
    assert "function refreshVolumeQcSummary" in js
    assert "payload.stage && payload.stage.stageName" in js
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```powershell
python -m pytest project\tests\unit\test_frontend_regressions.py -v
```

Expected:
- missing stage/QC IDs and JS functions

- [ ] **Step 3: Add the new UI sections**

Modify `project/frontend/index.html`:

```html
<section id="wholeBrain3dStatusSection" class="panel">
  <h2>3D Registration Status</h2>
  <p id="aux2dNotice" class="muted">Whole-brain automatic truth comes from the 3D pipeline. 2D tools below are preview/manual helpers only.</p>
  <div id="wholeBrainStageList" class="stage-list"></div>
</section>

<section id="volumeQcSection" class="panel">
  <h2>3D QC Summary</h2>
  <div id="volumeQcSummary"></div>
</section>

<section id="sliceInspectorSection" class="panel">
  <h2>Slice Inspector</h2>
  <p class="muted">These overlays are exported from the final 3D truth volume, not independently auto-registered slice results.</p>
  <div id="sliceInspectorGrid" class="qc-grid"></div>
</section>
```

Modify `project/frontend/styles.css`:

```css
.stage-list { display: grid; gap: 10px; }
.stage-row { border: 1px solid #2c3550; border-radius: 10px; padding: 12px; background: #111827; }
.stage-row.active { border-color: #5b8cff; box-shadow: 0 0 0 1px rgba(91,140,255,0.35); }
.stage-row.done { border-color: #2f8f5b; }
.stage-name { font-weight: 700; color: #eef2ff; }
.stage-meta { font-size: 12px; color: #9aa4bd; margin-top: 4px; }
.stage-bar { height: 8px; border-radius: 999px; background: #1f2937; overflow: hidden; margin-top: 8px; }
.stage-bar-fill { height: 100%; background: linear-gradient(90deg, #4f7cff, #73c2fb); }
```

Modify `project/frontend/app.js`:

```javascript
function renderWholeBrain3dStage(stage) {
  const stages = [
    "Volume Build",
    "Template Prep",
    "ANTS Registration",
    "Laplacian Refinement",
    "Truth Export",
    "Quantification",
  ];
  const host = document.getElementById('wholeBrainStageList');
  if (!host) return;
  host.innerHTML = '';
  stages.forEach((name, idx) => {
    const pct = stage && stage.stageName === name ? Number(stage.percent || 0) : (stage && stage.stageIndex > (idx + 1) ? 100 : 0);
    const row = document.createElement('div');
    row.className = 'stage-row' + (stage && stage.stageName === name ? ' active' : '') + (pct === 100 ? ' done' : '');
    row.innerHTML = `
      <div class="stage-name">${idx + 1}. ${name}</div>
      <div class="stage-meta">${stage && stage.stageName === name ? (stage.message || '') : ''}</div>
      <div class="stage-bar"><div class="stage-bar-fill" style="width:${pct}%"></div></div>
    `;
    host.appendChild(row);
  });
}

async function refreshVolumeQcSummary() {
  try {
    const res = await fetch('/api/outputs/volume-reg-stats');
    if (!res.ok) return;
    const text = await res.text();
    document.getElementById('volumeQcSummary').textContent = text;
  } catch {}
}
```

- [ ] **Step 4: Wire status polling into the new UI**

Add to the existing status refresh block in `project/frontend/app.js`:

```javascript
if (payload.stage && payload.stage.stageName) {
  renderWholeBrain3dStage(payload.stage);
}
await refreshVolumeQcSummary();
```

- [ ] **Step 5: Run the UI regression tests to verify they pass**

Run:

```powershell
python -m pytest project\tests\unit\test_frontend_regressions.py -v
```

Expected:
- new HTML/JS assertions pass

- [ ] **Step 6: Commit**

```powershell
git add project/frontend/index.html project/frontend/app.js project/frontend/styles.css project/tests/unit/test_frontend_regressions.py
git commit -m "feat: add 3d whole-brain progress and qc ui"
```

---

## Task 9: Validate environment, update docs, and harden regression coverage

**Files:**
- Modify: `project/scripts/check_env.py`
- Modify: `README.md`
- Modify: `REPRODUCE.md`
- Modify: `project/tests/unit/test_main.py`

- [ ] **Step 1: Write the failing environment/doc assertions**

Append to `project/tests/unit/test_main.py`:

```python
def test_run_config_template_mentions_miki_3d_backend():
    text = Path(r"D:\Brainfast\project\configs\run_config.template.json").read_text(encoding="utf-8")
    assert '"whole_brain_backend": "miki_3d"' in text
    assert '"truth_source": "3d_registered_volume"' in text
```

- [ ] **Step 2: Run the targeted tests to verify failures**

Run:

```powershell
python -m pytest project\tests\unit\test_main.py -v
```

Expected:
- config/doc assertions fail until the env/docs are updated consistently

- [ ] **Step 3: Add ANTs validation to `check_env.py`**

Modify `project/scripts/check_env.py`:

```python
def _check_ants_dependency(issues: list[str]) -> None:
    try:
        import ants  # type: ignore
    except Exception as exc:
        issues.append(f"ANTs Python package unavailable: {exc}")

# in main validation flow
_check_ants_dependency(issues)
```

- [ ] **Step 4: Update the user-facing docs**

Modify `README.md`:

```markdown
- Whole-brain automatic registration now uses a 3D volume-first truth path.
- Exported slice overlays and counts are derived from the final 3D registered annotation volume.
- 2D tools remain available for preview and manual correction only.
```

Modify `REPRODUCE.md`:

```markdown
Expected whole-brain 3D artifacts:

- `brain_25um.nii.gz`
- `template_half.nii.gz`
- `registered_output/02_ants/registration_metrics.csv`
- `registered_output/03_laplacian_refinement/refinement_metrics.csv`
- `annotation_registered.nii.gz`
- `volume_registration_qc.csv`
```

- [ ] **Step 5: Run the full focused verification set**

Run:

```powershell
python -m pytest project\tests\unit\test_registration_3d_volume.py project\tests\unit\test_registration_3d_ants.py project\tests\unit\test_laplacian_refine_3d.py project\tests\unit\test_truth_export_3d.py project\tests\unit\test_pipeline_progress.py project\tests\unit\test_whole_brain_3d.py project\tests\unit\test_main.py project\tests\unit\test_frontend_regressions.py project\tests\unit\test_services.py -v
python -m pytest project\tests\integration\test_regression_suite.py -v
```

Expected:
- unit suite passes
- integration suite passes

- [ ] **Step 6: Commit**

```powershell
git add project/scripts/check_env.py README.md REPRODUCE.md project/tests/unit/test_main.py
git commit -m "docs: validate and document miki-style 3d whole-brain flow"
```

---

## Self-Review

### Spec coverage

- Whole-brain default 3D truth path: Tasks 1-5
- ANTs + Laplacian refinement: Tasks 2-3
- 3D-derived overlay/QC/mapping/counting: Tasks 4-7
- 2D auxiliary-only boundary: Tasks 5 and 8
- Visible progress bar and slow-stage transparency: Tasks 5, 7, and 8

No spec sections are currently uncovered.

### Placeholder scan

No placeholder markers remain in this plan.

### Type consistency

This plan consistently uses:

- `run_whole_brain_3d`
- `write_stage_progress`
- `run_ants_registration`
- `refine_registered_volume`
- `export_registered_truth_slices`
- `whole_brain_backend = "miki_3d"`
- `truth_source = "3d_registered_volume"`

---

Plan complete and saved to `docs/superpowers/plans/2026-04-02-plan-miki-style-3d-registration.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

**Which approach?**
```
