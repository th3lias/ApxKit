from typing import Union, List, Tuple

import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from scipy.linalg import lstsq, lu

from fit import BasisType
from function import Function
from grid.grid.grid import Grid
from grid.rule.random_grid_rule import RandomGridRule
from interpolate.interpolator import Interpolator
from fit.method.least_squares_method import LeastSquaresMethod
from utils.utils import find_degree


class LeastSquaresInterpolator(Interpolator):
    def __init__(self, include_bias: bool, basis_type: BasisType, method: LeastSquaresMethod = LeastSquaresMethod.EXACT,
                 grid: Union[Grid, None] = None, store_indices: bool = True):
        super().__init__(grid, store_indices)
        self.include_bias = include_bias
        self.method = method
        self.basis_type = basis_type

    def set_method(self, method: LeastSquaresMethod):
        self.method = method

    def fit(self, f: Union[Function, List[Function]]):

        assert self.grid is not None, "Grid needs to be set before interpolation"

        if self.basis is None:
            self.basis = self._build_basis()

        y = self._calculate_y(f)

        if self.method == LeastSquaresMethod.EXACT:
            self._approximate_exact(y)
        elif self.method == LeastSquaresMethod.SCIPY_LSTSQ_GELSD:
            self._approximate_scipy_lstsq(y, 'gelsd')
        elif self.method == LeastSquaresMethod.SCIPY_LSTSQ_GELSS:
            self._approximate_scipy_lstsq(y, 'gelss')
        elif self.method == LeastSquaresMethod.SCIPY_LSTSQ_GELSY:
            self._approximate_scipy_lstsq(y, 'gelsy')
        else:
            raise ValueError(f'The method {self.method.name} is not supported')

    def interpolate(self, grid: Union[Grid, np.ndarray]):
        """
        Applies the fitted function on the given data and returns the calculates values
        :param grid: Grid on which the fitted function should be tested
        """

        if isinstance(grid, Grid):
            grid = grid.grid

        data_pol = self._build_basis(basis_type=None, grid=grid, b_idx=self._b_idx)
        y_hat = data_pol @ self.coeff
        if y_hat.ndim > 1:
            return y_hat.T
        return y_hat

    def _build_basis(self, basis_type: Union[BasisType, None] = None, grid: Union[None, np.ndarray] = None,
                     b_idx: Union[List[Tuple[int]], None] = None):

        if basis_type is None:
            basis_type = self.basis_type

        if not basis_type == BasisType.CHEBYSHEV and not basis_type == BasisType.REGULAR:
            raise ValueError(f"Unsupported Basis-Type. Expected Chebyshev or Regular Basis but got {basis_type}")

        if grid is None:
            grid = self.grid.grid

        if basis_type == BasisType.CHEBYSHEV:
            return self._build_poly_basis(grid, b_idx)

        elif basis_type == BasisType.REGULAR:
            degree = find_degree(self.scale, self.dim)

            poly = PolynomialFeatures(degree=degree, include_bias=self.include_bias)
            return poly.fit_transform(grid)

        else:
            raise ValueError(f"Unexpected basis type {basis_type}")

    def _approximate_exact(self, y: np.ndarray):
        """
        Approximates a (or multiple) function(s) with polynomials by least squares.
        :param y: function values
        """
        if not self.include_bias:
            print("Please be aware that the result may become significantly worse when using no intercepts (bias)")

        self._self_implementation(y)

    def _self_implementation(self, y: np.ndarray):

        weight = self._get_weights_for_weighted_ls()

        print("Warning: The following is very unstable, since we calculate A.T@A")

        x_poly = (weight * self.basis.T).T
        self.basis = None
        y_prime = x_poly.T @ (weight * y.T).T
        x2 = x_poly.T @ x_poly
        del x_poly
        self.L, self.U = lu(x2, permute_l=True)[-2:]
        coeff = np.linalg.solve(self.U, np.linalg.solve(self.L, y_prime))
        self.coeff = coeff

    def _approximate_scipy_lstsq(self, y: np.ndarray, lapack_driver: str = 'gelsy'):

        if not self.include_bias:
            print("Please be aware that the result may become significantly worse when using no intercept (bias)")

        weight = self._get_weights_for_weighted_ls()

        x_poly = (weight * self.basis.T).T
        self.basis = None
        y_prime = (weight * y.T).T

        sol = lstsq(x_poly, y_prime, lapack_driver=lapack_driver)
        coeff = sol[0]

        self.coeff = coeff

    def _get_weights_for_weighted_ls(self):
        """
        Calculates the weights for the weighted least squares (vectorized method).
        """
        points = self.grid.grid
        if self.grid.rule == RandomGridRule.CHEBYSHEV:
            if self.grid.lower_bound == 0.0 and self.grid.upper_bound == 1.0:
                points = 2 * points - 1
            elif self.grid.lower_bound != -1.0 or self.grid.upper_bound != 1.0:
                raise ValueError("The Chebyshev rule only supports the range [-1, 1] or [0, 1]")
            weight = np.sqrt(np.prod(np.polynomial.chebyshev.chebweight(points), axis=1))
        elif self.grid.rule == RandomGridRule.UNIFORM:
            weight = np.ones(shape=(self.grid.get_num_points()), dtype=np.float64)
        else:
            raise ValueError(f"Unsupported grid type {self.grid.rule}")
        return weight
