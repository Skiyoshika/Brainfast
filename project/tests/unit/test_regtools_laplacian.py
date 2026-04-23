"""Unit tests for vendored RegTools Laplacian solver.

These exercise the vendored CG + Jacobi-preconditioned Laplace solver
directly (not through the Brainfast refinement wrapper) to verify the
vendored module behaves as specified.
"""

from __future__ import annotations

import numpy as np
import pytest


def test_solver_module_importable():
    """Vendored module must be importable as a sibling package."""
    from project.scripts.regtools_laplacian import solveLaplacianFromCorrespondences

    assert callable(solveLaplacianFromCorrespondences)


def test_solver_zero_correspondences_returns_zero_field():
    """With no correspondence points the field is trivially zero."""
    from project.scripts.regtools_laplacian import solveLaplacianFromCorrespondences

    field = solveLaplacianFromCorrespondences(
        vol_shape=(4, 5, 6),
        source_pts=np.empty((0, 3)),
        target_pts=np.empty((0, 3)),
        axes=(0, 1, 2),
    )
    assert field.shape == (3, 4, 5, 6)
    assert np.allclose(field, 0.0)


def test_solver_single_axis_translation_recovered():
    """A uniform rigid shift along axis 2 should solve to an approximately
    uniform displacement of the same magnitude at every tissue voxel."""
    from project.scripts.regtools_laplacian import solveLaplacianFromCorrespondences

    shape = (6, 8, 8)
    # Seed 6 boundary points around the cube with a +2 shift in axis 2 (x).
    corners = np.array(
        [
            [0, 0, 0], [0, 0, 7], [0, 7, 0], [0, 7, 7],
            [5, 0, 0], [5, 0, 7], [5, 7, 0], [5, 7, 7],
        ],
        dtype=float,
    )
    source_pts = corners.copy()
    target_pts = corners.copy()
    source_pts[:, 2] += 2.0  # moving is 2 voxels to the right of target

    field = solveLaplacianFromCorrespondences(
        vol_shape=shape,
        source_pts=source_pts,
        target_pts=target_pts,
        axes=(0, 1, 2),
        rtol=1e-3,
        maxiter=200,
    )

    # Axis 2 displacement at interior voxels should recover the +2 shift.
    axis2_interior = field[2, 2:4, 2:6, 2:6]
    assert np.mean(axis2_interior) == pytest.approx(2.0, abs=0.5), (
        f"uniform +2 shift should propagate; mean interior axis-2 disp = "
        f"{np.mean(axis2_interior):.3f}"
    )

    # Other axes should stay near zero (no cross-talk for pure axis-2 shift).
    axis0_interior = field[0, 2:4, 2:6, 2:6]
    axis1_interior = field[1, 2:4, 2:6, 2:6]
    assert abs(np.mean(axis0_interior)) < 0.5
    assert abs(np.mean(axis1_interior)) < 0.5


def test_solver_anisotropic_spacing_accepted():
    """spacing kwarg must flow through without error."""
    from project.scripts.regtools_laplacian import solveLaplacianFromCorrespondences

    src = np.array([[0, 0, 0], [0, 0, 3]], dtype=float)
    tgt = np.array([[0, 0, 0], [0, 0, 3]], dtype=float)
    src[:, 2] += 1.0

    field = solveLaplacianFromCorrespondences(
        vol_shape=(3, 4, 4),
        source_pts=src,
        target_pts=tgt,
        axes=(0, 1, 2),
        spacing=(0.025, 0.005, 0.005),
        rtol=1e-2,
        maxiter=100,
    )
    assert field.shape == (3, 3, 4, 4)


def test_slice_to_slice_laplacian_module_importable():
    """Xu Lab slice-to-slice Laplacian must be exposed from the vendored package."""
    from project.scripts.regtools_laplacian import sliceToSlice3DLaplacian

    assert callable(sliceToSlice3DLaplacian)


def test_slice_to_slice_laplacian_empty_volumes_returns_zero_field():
    """Empty/blank volumes produce no correspondences; field is zero."""
    from project.scripts.regtools_laplacian import sliceToSlice3DLaplacian

    fixed = np.zeros((4, 6, 6), dtype=np.float32)
    moving = np.zeros((4, 6, 6), dtype=np.float32)
    field = sliceToSlice3DLaplacian(
        fixedImage=fixed,
        movingImage=moving,
        axis=0,
        rtol=1e-2,
        maxiter=10,
    )
    assert field.shape == (3, 4, 6, 6)
    assert np.allclose(field, 0.0)
