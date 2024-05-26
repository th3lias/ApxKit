from typing import Callable, Union, List, Tuple

import numpy as np
from scipy.linalg import lu

from grid.grid import Grid
from grid.grid_provider import GridProvider, GridType
from interpolate.basis_types import BasisType
from interpolate.interpolator import Interpolator


# most of the content is adapted from https://github.com/EconForge/Smolyak, which implemented the Smolyak algorithm
# based on the paper:

# Smolyak method for solving dynamic economic models:
# Lagrange interpolation, anisotropic grid and adaptive domain

# from Kenneth L. Judd, Lilia Maliar, Serguei Maliar and Rafael Valero


class SmolyakInterpolator(Interpolator):
    def __init__(self, grid: Grid, basis_type: BasisType = BasisType.CHEBYSHEV):
        if grid is None:
            raise ValueError("Grid must not be None, but of type Grid!")

        super().__init__(grid)
        self.basis_type = basis_type

    def interpolate(self, f: Union[Callable, List[Callable]]) -> Callable:
        if self.basis is None:
            self.basis = self._build_basis()
            self.L, self.U = lu(self.basis, permute_l=True)[-2:]
        n_samples = self.grid.grid.shape[0]
        if isinstance(f, list):
            y = np.empty(shape=(n_samples, len(f)), dtype=np.float64)
            for i, func in enumerate(f):
                if not isinstance(func, Callable):
                    raise ValueError(f"One element of the list is not a function but from the type {type(func)}")
                y[:, i] = func(self.grid.grid)
        else:
            y = f(self.grid.grid)
        coeff = np.linalg.solve(self.U, np.linalg.solve(self.L, y))

        def f_hat(data: np.ndarray) -> np.ndarray:
            data_transformed = self._build_basis(grid=data, b_idx=self._b_idx)
            y_hat = data_transformed @ coeff
            if y_hat.ndim > 1:
                return y_hat.T
            return y_hat

        return f_hat

    def _build_basis(self, basis_type: Union[BasisType, None] = None, grid: Union[None, np.ndarray] = None,
                     b_idx: Union[List[Tuple[int]], None] = None):

        if basis_type is None:
            basis_type = self.basis_type

        if not basis_type == BasisType.CHEBYSHEV:
            raise ValueError("Smolyak Algorithm must not run with another basis")

        return self._build_poly_basis(grid, b_idx)
