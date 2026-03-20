from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
from PIL import Image

from scripts.make_registration_overview import make_registration_overview


def test_make_registration_overview_masks_background_outside_annotation(tmp_path: Path) -> None:
    brain = np.full((1, 10, 10), 250.0, dtype=np.float32)
    brain[0, 2:8, 2:8] = 1000.0
    annotation = np.zeros((1, 10, 10), dtype=np.int32)
    annotation[0, 2:8, 2:8] = 42

    brain_path = tmp_path / "brain.nii.gz"
    annotation_path = tmp_path / "annotation.nii.gz"
    out_path = tmp_path / "overview.png"
    nib.save(nib.Nifti1Image(brain, np.eye(4)), str(brain_path))
    nib.save(nib.Nifti1Image(annotation, np.eye(4)), str(annotation_path))

    make_registration_overview(brain_path, annotation_path, out_path, slices=[0])

    img = np.asarray(Image.open(out_path))
    brain_panel = img[34:, :360, :]
    assert tuple(int(v) for v in brain_panel[20, 20]) == (0, 0, 0)
    assert int(brain_panel[180, 180, 0]) > 0
