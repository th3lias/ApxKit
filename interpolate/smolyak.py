from typing import Union, List, Tuple

import numpy as np
from TasmanianSG import TasmanianSparseGrid
from scipy.linalg import lu

from fit import BasisType
from function import Function
from grid.grid.grid import Grid
from fit.method.interpolation_method import InterpolationMethod
from interpolate.interpolator import Interpolator


# most of the content (some parts of it in the >>Interpolator<< class) is adapted from
# https://github.com/EconForge/Smolyak, which implemented the Smolyak algorithm based on the paper:

# Smolyak method for solving dynamic economic models:
# Lagrange interpolation, anisotropic grid and adaptive domain

# from Kenneth L. Judd, Lilia Maliar, Serguei Maliar and Rafael Valero

class SmolyakInterpolator(Interpolator):
    def __init__(self, grid: Grid, method: InterpolationMethod, basis_type: BasisType = BasisType.CHEBYSHEV,
                 store_indices: bool = True):
        if grid is None:
            raise ValueError("Grid must not be None, but of type Grid!")

        super().__init__(grid, store_indices)
        self.method = method
        self.basis_type = basis_type

    def set_method(self, method: InterpolationMethod):
        self.method = method

    def fit(self, f: Union[Function, List[Function]]):

        if self.basis is None:
            self.basis = self._build_basis()
            self.L, self.U = lu(self.basis, permute_l=True)[-2:]

        # calculate y
        y = self._calculate_y(f)

        coeff = np.linalg.solve(self.U, np.linalg.solve(self.L, y))

        self.coeff = coeff

    def interpolate(self, grid: Union[Grid, np.ndarray]) -> np.ndarray:
        if isinstance(grid, Grid):
            grid = grid.grid

        if isinstance(grid, TasmanianSparseGrid):
            # transform to numpy array
            grid = grid.getPoints()

        if self.method == InterpolationMethod.STANDARD:
            data_transformed = self._build_basis(grid=grid, b_idx=self._b_idx)
            y_hat = data_transformed @ self.coeff
            if y_hat.ndim > 1:
                return y_hat.T
            return y_hat
        else:
            raise ValueError(f"Method {self.method} is not supported!")

    def _build_basis(self, basis_type: Union[BasisType, None] = None, grid: Union[None, np.ndarray] = None,
                     b_idx: Union[List[Tuple[int]], None] = None):

        if basis_type is None:
            basis_type = self.basis_type

        if not basis_type == BasisType.CHEBYSHEV:
            raise ValueError("Smolyak Algorithm must not run with another basis")

        return self._build_poly_basis(grid, b_idx)
