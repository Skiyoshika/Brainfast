"""Vendored Laplacian solver from UCI-XuLab-RegTools.

Source: UCI-XuLab/UCI-XuLab-RegTools
Original author: Atchuth Naveen, UC Irvine
Vendored on: 2026-04-16 (solver/utils), extended 2026-04-22 (correspondence).
Why: Brainfast's inline _build_laplacian_3d_fast / _solve_laplace_displacement
     OOM'd on 89M-voxel template-space refinement (see
     docs/superpowers/plans/2026-04-16-registration-quality-investigation.md
     Bug #4). RegTools' implementation uses Dirichlet BC with boundary-column
     exclusion and CG + Jacobi preconditioner, giving a smaller and better-
     conditioned system. The 2D-per-slice correspondence matcher
     ``sliceToSlice3DLaplacian`` was added 2026-04-22 to bring Laplacian
     refinement to parity with Xu Lab's canonical pipeline (Canny contours +
     PCA normals + angle-constrained NN), replacing Brainfast's generic 3D
     surface+KDTree matcher.

Usage:
    from project.scripts.regtools_laplacian import solveLaplacianFromCorrespondences

    field = solveLaplacianFromCorrespondences(
        vol_shape=(D, H, W),
        source_pts=moving_voxel_coords,          # shape (N, 3)
        target_pts=template_voxel_coords,        # shape (N, 3)
        axes=(0, 1, 2),                          # solve all 3 axes
        spacing=(0.025, 0.025, 0.025),           # mm per voxel
    )

    # or the end-to-end slice-by-slice path (Xu Lab canonical):
    from project.scripts.regtools_laplacian import sliceToSlice3DLaplacian
    field = sliceToSlice3DLaplacian(fixed_arr, moving_arr, axis=0, spacing=spacing)
"""

from .solver import solveLaplacianFromCorrespondences
from .utils import laplacianA3D, propagate_dirichlet_rhs

__all__ = [
    "solveLaplacianFromCorrespondences",
    "sliceToSlice3DLaplacian",
    "laplacianA3D",
    "propagate_dirichlet_rhs",
]


def __getattr__(name):
    """Lazy-load ``sliceToSlice3DLaplacian`` so the package is importable
    even when its heavier transitive deps (joblib, tqdm, skimage.feature)
    aren't installed — the CI Py 3.10 lane doesn't ship them.
    Importing the symbol via ``from regtools_laplacian import sliceToSlice3DLaplacian``
    or attribute access still works the same way for callers; they just
    pay the joblib import cost at first access instead of at module load.
    """
    if name == "sliceToSlice3DLaplacian":
        from .correspondence import sliceToSlice3DLaplacian as _impl

        return _impl
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
