"""Vendored Laplacian solver from UCI-XuLab-RegTools.

Source: UCI-XuLab/UCI-XuLab-RegTools
Original author: Atchuth Naveen, UC Irvine
Vendored on: 2026-04-16 for Brainfast
Why: Brainfast's inline _build_laplacian_3d_fast / _solve_laplace_displacement
     OOM'd on 89M-voxel template-space refinement (see
     docs/superpowers/plans/2026-04-16-registration-quality-investigation.md
     Bug #4). RegTools' implementation uses Dirichlet BC with boundary-column
     exclusion and CG + Jacobi preconditioner, giving a smaller and better-
     conditioned system.

Usage:
    from project.scripts.regtools_laplacian import solveLaplacianFromCorrespondences

    field = solveLaplacianFromCorrespondences(
        vol_shape=(D, H, W),
        source_pts=moving_voxel_coords,          # shape (N, 3)
        target_pts=template_voxel_coords,        # shape (N, 3)
        axes=(0, 1, 2),                          # solve all 3 axes
        spacing=(0.025, 0.025, 0.025),           # mm per voxel
    )
"""

from .solver import solveLaplacianFromCorrespondences
from .utils import laplacianA3D, propagate_dirichlet_rhs

__all__ = [
    "solveLaplacianFromCorrespondences",
    "laplacianA3D",
    "propagate_dirichlet_rhs",
]
