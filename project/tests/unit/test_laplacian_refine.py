from __future__ import annotations

import numpy as np
from scipy.sparse.linalg import cg

from scripts.laplacian_refine import WeightedDirichletLaplacian, apply_deformation_field


def _explicit_matrix(shape: tuple[int, int, int], boundary_mask: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    size = int(np.prod(shape))
    weights = tuple(1.0 / (s * s) for s in spacing)
    boundary_mask = np.asarray(boundary_mask, dtype=bool)
    A = np.zeros((size, size), dtype=np.float32)

    def _idx(coord: tuple[int, int, int]) -> int:
        return np.ravel_multi_index(coord, shape)

    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                coord = (x, y, z)
                row = _idx(coord)
                if boundary_mask[coord]:
                    A[row, row] = 1.0
                    continue

                for axis, weight in enumerate(weights):
                    for delta in (-1, 1):
                        nbr = [x, y, z]
                        nbr[axis] += delta
                        if nbr[axis] < 0 or nbr[axis] >= shape[axis]:
                            continue
                        nbr_t = tuple(nbr)
                        A[row, row] += weight
                        if not boundary_mask[nbr_t]:
                            col = _idx(nbr_t)
                            A[row, col] -= weight
    return A


def _explicit_rhs(shape: tuple[int, int, int], boundary_mask: np.ndarray, spacing: tuple[float, float, float], boundary_values: np.ndarray) -> np.ndarray:
    size = int(np.prod(shape))
    weights = tuple(1.0 / (s * s) for s in spacing)
    rhs = np.zeros(size, dtype=np.float32)

    def _idx(coord: tuple[int, int, int]) -> int:
        return np.ravel_multi_index(coord, shape)

    for x in range(shape[0]):
        for y in range(shape[1]):
            for z in range(shape[2]):
                coord = (x, y, z)
                row = _idx(coord)
                if boundary_mask[coord]:
                    rhs[row] = float(boundary_values[coord])
                    continue
                total = 0.0
                for axis, weight in enumerate(weights):
                    for delta in (-1, 1):
                        nbr = [x, y, z]
                        nbr[axis] += delta
                        if nbr[axis] < 0 or nbr[axis] >= shape[axis]:
                            continue
                        nbr_t = tuple(nbr)
                        if boundary_mask[nbr_t]:
                            total += weight * float(boundary_values[nbr_t])
                rhs[row] = total
    return rhs


def test_weighted_dirichlet_laplacian_matches_explicit_matrix() -> None:
    shape = (2, 3, 3)
    spacing = (1.0, 2.0, 0.5)
    boundary_mask = np.zeros(shape, dtype=bool)
    boundary_mask[0, 0, 0] = True
    boundary_mask[1, 2, 2] = True

    lap = WeightedDirichletLaplacian(shape, boundary_mask, spacing)
    A = _explicit_matrix(shape, boundary_mask, spacing)

    x = np.arange(np.prod(shape), dtype=np.float32)
    np.testing.assert_allclose(lap.operator.matvec(x), A @ x, rtol=1e-6, atol=1e-6)

    boundary_values = np.zeros(shape, dtype=np.float32)
    boundary_values[0, 0, 0] = 3.0
    boundary_values[1, 2, 2] = -2.0
    rhs_expected = _explicit_rhs(shape, boundary_mask, spacing, boundary_values)
    rhs_actual = lap.build_rhs(boundary_values)
    np.testing.assert_allclose(rhs_actual, rhs_expected, rtol=1e-6, atol=1e-6)

    solved, info = cg(
        lap.operator,
        rhs_actual,
        x0=boundary_values.ravel(),
        rtol=1e-8,
        atol=0.0,
        M=lap.preconditioner,
        maxiter=200,
    )
    assert info == 0
    np.testing.assert_allclose(solved, np.linalg.solve(A, rhs_expected), rtol=1e-4, atol=1e-4)


def test_apply_deformation_field_uses_output_to_input_mapping() -> None:
    volume = np.zeros((3, 3, 3), dtype=np.float32)
    volume[1, 1, 1] = 7.0
    field = np.zeros((3, 3, 3, 3), dtype=np.float32)
    field[2] = 1.0

    warped = apply_deformation_field(volume, field, chunk_slices=2)
    assert float(warped[1, 1, 0]) == 7.0
    assert float(warped[1, 1, 1]) == 0.0
