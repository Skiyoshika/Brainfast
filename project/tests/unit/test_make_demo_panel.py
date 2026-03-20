from __future__ import annotations

from pathlib import Path

import numpy as np
from tifffile import imwrite

from scripts.make_demo_panel import (
    _apply_tissue_alpha,
    _clip_to_tissue,
    _crop_to_brain,
    _family_color,
    _meaningful_region_label,
    _select_region_annotations,
    _tissue_support_from_raw,
)


def test_crop_to_brain_prefers_explicit_mask() -> None:
    img = np.zeros((20, 20, 3), dtype=np.uint8)
    img[0, 0] = (255, 255, 255)
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:15, 6:14] = True

    cropped = _crop_to_brain(img, pad=1, mask=mask)

    assert cropped.shape == (12, 10, 3)


def test_clip_to_tissue_feathers_outer_edge() -> None:
    img = np.full((15, 15, 3), 200, dtype=np.uint8)
    mask = np.zeros((15, 15), dtype=bool)
    mask[3:12, 3:12] = True

    clipped = _clip_to_tissue(img, mask, feather_px=2)

    assert tuple(int(v) for v in clipped[0, 0]) == (0, 0, 0)
    assert 0 < int(clipped[3, 3, 0]) < 200
    assert int(clipped[7, 7, 0]) >= 190


def test_tissue_support_from_raw_builds_soft_alpha(tmp_path: Path) -> None:
    raw = np.zeros((40, 40), dtype=np.uint16)
    raw[8:32, 10:30] = 900
    raw[6:34, 18:22] = 1000
    path = tmp_path / "raw.tif"
    imwrite(str(path), raw)

    mask, alpha = _tissue_support_from_raw(path)

    assert mask is not None
    assert alpha is not None
    assert mask.shape == raw.shape
    assert alpha.shape == raw.shape
    assert bool(mask[20, 20])
    assert float(alpha[20, 20]) > 0.95
    assert float(alpha[0, 0]) < 0.05


def test_apply_tissue_alpha_uses_soft_matte() -> None:
    img = np.full((5, 5, 3), 200, dtype=np.uint8)
    alpha = np.zeros((5, 5), dtype=np.float32)
    alpha[2, 2] = 1.0
    alpha[2, 1] = 0.5

    out = _apply_tissue_alpha(img, alpha)

    assert tuple(int(v) for v in out[0, 0]) == (0, 0, 0)
    assert int(out[2, 1, 0]) == 100
    assert int(out[2, 2, 0]) == 200


def test_meaningful_region_label_filters_generic_and_numeric_entries() -> None:
    lookup = {
        1158: {"acronym": "BRAIN", "name": "Brain", "color": "000000"},
        1282: {"acronym": "1282", "name": "1282", "color": "CCCCCC"},
        662: {"acronym": "GU6b", "name": "Gustatory areas, layer 6b", "color": "009C75"},
    }

    assert _meaningful_region_label(1158, lookup) is None
    assert _meaningful_region_label(1282, lookup) is None
    assert _meaningful_region_label(662, lookup) == ("GU6b", "Gustatory areas, layer 6b")


def test_select_region_annotations_skips_invalid_numeric_regions() -> None:
    label = np.zeros((20, 20), dtype=np.int32)
    label[1:10, 1:10] = 1158
    label[10:19, 1:10] = 1282
    label[3:17, 10:19] = 662
    lookup = {
        1158: {"acronym": "BRAIN", "name": "Brain", "color": "000000"},
        662: {"acronym": "GU6b", "name": "Gustatory areas, layer 6b", "color": "009C75"},
    }

    selected = _select_region_annotations(label, structure_lookup=lookup, top_n=5, min_pixels=20)

    assert len(selected) == 1
    assert selected[0]["acro"] == "GU6b"


def test_family_color_keeps_related_regions_in_same_color_family() -> None:
    lookup = {
        500: {"acronym": "MO", "name": "Somatomotor areas", "color": "1F9D5A"},
        985: {"acronym": "MOp", "name": "Primary motor area", "color": "1F9D5A"},
        453: {"acronym": "SS", "name": "Somatosensory areas", "color": "188064"},
    }

    mo = np.array(_family_color(500, lookup))
    mop = np.array(_family_color(985, lookup))
    ss = np.array(_family_color(453, lookup))

    assert np.abs(mo - mop).sum() < np.abs(mo - ss).sum()
