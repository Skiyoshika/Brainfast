"""
Shared pytest fixtures for Brainfast tests.

Session-scoped fixtures are created once per test run and reused across all
test files — this avoids redundant atlas/TIFF loading in integration tests.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from tifffile import imwrite

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ── Tiny synthetic images (unit tests — no atlas required) ────────────────────

@pytest.fixture(scope="session")
def tiny_label_tif(tmp_path_factory):
    """20x20 int32 label TIFF with one non-zero region."""
    p = tmp_path_factory.mktemp("fixtures") / "label.tif"
    arr = np.zeros((20, 20), dtype=np.int32)
    arr[2:8, 2:8] = 12345
    imwrite(str(p), arr)
    return p


@pytest.fixture(scope="session")
def tiny_real_tif(tmp_path_factory):
    """20x20 uint16 grayscale TIFF with random noise."""
    p = tmp_path_factory.mktemp("fixtures") / "real.tif"
    rng = np.random.default_rng(42)
    arr = (rng.random((20, 20)) * 65535).astype(np.uint16)
    imwrite(str(p), arr)
    return p


@pytest.fixture(scope="session")
def sparse_real_array():
    """Simulated sparse cleared-brain slice: mostly zeros, a few bright cells."""
    rng = np.random.default_rng(7)
    arr = np.zeros((256, 256), dtype=np.uint16)
    arr += rng.integers(50, 150, arr.shape, dtype=np.uint16)
    for _ in range(30):
        y, x = rng.integers(10, 246, size=2)
        arr[y - 2:y + 3, x - 2:x + 3] = rng.integers(40000, 65000)
    return arr


# ── Atlas fixture (integration tests only) ────────────────────────────────────

@pytest.fixture(scope="session")
def atlas_nii_path():
    """Path to annotation_25.nii.gz — skips test if not present."""
    p = PROJECT_ROOT / "annotation_25.nii.gz"
    if not p.exists():
        pytest.skip("annotation_25.nii.gz not found — skipping integration test")
    return p


@pytest.fixture(scope="session")
def structure_csv_path():
    """Path to Allen structure graph CSV."""
    p = PROJECT_ROOT / "configs" / "allen_mouse_structure_graph.csv"
    if not p.exists():
        pytest.skip("allen_mouse_structure_graph.csv not found")
    return p
