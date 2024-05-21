from typing import Callable, Union, List, Tuple

import numpy as np
from scipy.linalg import lu

from grid.grid_provider import GridProvider, GridType
from interpolate.basis_types import BasisType
from interpolate.interpolator import Interpolator


# most of the content is adapted from https://github.com/EconForge/Smolyak, which implemented the Smolyak algorithm
# based on the paper:

# Smolyak method for solving dynamic economic models:
# Lagrange interpolation, anisotropic grid and adaptive domain

# from Kenneth L. Judd, Lilia Maliar, Serguei Maliar and Rafael Valero


class SmolyakInterpolator(Interpolator):
    def __init__(self, dimension: int, scale: int, basis_type: BasisType = BasisType.CHEBYSHEV, seed: int = None):
        self.dim = dimension
        self.scale = scale
        self.basis_type = basis_type
        self.gp = GridProvider(self.dim, seed=seed)
        super().__init__(self.gp.generate(GridType.CHEBYSHEV, self.scale))

    def interpolate(self, f: Callable) -> Callable:
        if self.basis is None:
            self.basis = self._build_basis()
            self.l, self.u = lu(self.basis, permute_l=True)[-2:]
        y = f(self.grid.grid)
        coeff = np.linalg.solve(self.u, np.linalg.solve(self.l, y))

        def f_hat(data: np.ndarray) -> np.ndarray:
            data_transformed = self._build_basis(grid=data, b_idx=self._b_idx)
            return data_transformed @ coeff

        return f_hat

    def _build_basis(self, basis_type: Union[BasisType, None]=None, grid: Union[None, np.ndarray] = None,
                     b_idx: Union[List[Tuple[int]], None] = None):

        if basis_type is None:
            basis_type = self.basis_type

        if not basis_type == BasisType.CHEBYSHEV:
            raise ValueError("Smolyak Algorithm must not run with another basis")

        return self._build_poly_basis(grid, b_idx)
