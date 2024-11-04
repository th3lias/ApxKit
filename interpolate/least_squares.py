from typing import Callable, Union, List, Tuple

import numpy as np
import scipy
from sklearn.preprocessing import PolynomialFeatures

from grid.grid.grid import Grid
from grid.rule.random_grid_rule import RandomGridRule
from interpolate.basis_types import BasisType
from interpolate.interpolator import Interpolator
from fit.method.least_squares_method import LeastSquaresMethod
from utils.utils import find_degree

from scipy.linalg import lu


class LeastSquaresInterpolator(Interpolator):
    def __init__(self, include_bias: bool, basis_type: BasisType, method: LeastSquaresMethod = LeastSquaresMethod.EXACT,
                 grid: Union[Grid, None] = None):
        super().__init__(grid)
        self.include_bias = include_bias
        self.method = method
        self.basis_type = basis_type

    def set_method(self, method: LeastSquaresMethod):
        self.method = method

    def fit(self, f: Union[Callable, List[Callable]]):

        assert self.grid is not None, "Grid needs to be set before interpolation"

        if self.basis is None:
            self.basis = self._build_basis()

        if self.method == LeastSquaresMethod.EXACT:
            self._approximate_exact(f)
        elif self.method == LeastSquaresMethod.NUMPY_LSTSQ:
            self._approximate_numpy_lstsq(f)
        elif self.method == LeastSquaresMethod.SCIPY_LSTSQ_GELSD:
            self._approximate_scipy_lstsq(f, 'gelsd')
        elif self.method == LeastSquaresMethod.SCIPY_LSTSQ_GELSS:
            self._approximate_scipy_lstsq(f, 'gelss')
        elif self.method == LeastSquaresMethod.SCIPY_LSTSQ_GELSY:
            self._approximate_scipy_lstsq(f, 'gelsy')
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
            print(DeprecationWarning("This will not be supported in the near future"))
            degree = find_degree(self.scale, self.dim)

            poly = PolynomialFeatures(degree=degree, include_bias=self.include_bias)
            return poly.fit_transform(grid)

    def _approximate_exact(self, f: Union[Callable, List[Callable]]):
        """
        Approximates a (or multiple) function(s) with polynomials by least squares.
        :param f: function or list of functions that need to be approximated on the same points
        :return: fitted function(s)
        """
        grid = self.grid.grid
        if not self.include_bias:
            print("Please be aware that the result may become significantly worse when using no intercepts (bias)")
        if not (isinstance(f, list) or isinstance(f, Callable)):
            raise ValueError(f"f needs to be a function or a list of functions but is {type(f)}")
        n_samples = grid.shape[0]
        if isinstance(f, list):
            y = np.empty(shape=(n_samples, len(f)), dtype=np.float64)
            for i, func in enumerate(f):
                if not isinstance(func, Callable):
                    raise ValueError(f"One element of the list is not a function but from the type {type(func)}")
                y[:, i] = func(grid)
        else:
            y = f(grid)

        self._self_implementation(y)

    def _self_implementation(self, y: np.ndarray):

        if self.grid.grid.grid_type == RandomGridRule.CHEBYSHEV:  # TODO: Check was previously just a self.grid.grid_type
            weight = np.empty(shape=(self.grid.get_num_points()))
            for i, row in enumerate(self.grid.grid):
                weight[i] = np.sqrt(np.prod(np.polynomial.chebyshev.chebweight(row) / np.pi))

        elif self.grid.grid.grid_type == RandomGridRule.UNIFORM:  # TODO: Check was previously just a self.grid.grid_type
            weight = np.ones(shape=(self.grid.get_num_points()), dtype=np.float64)
        else:
            raise ValueError(
                f"Unsupported grid type {self.grid.grid.grid_type}")  # TODO: Check was previously just a self.grid.grid_type

        print("Warning: The following is very unstable, since we calculate A.T@A")

        x_poly = (weight * self.basis.T).T
        self.basis = None
        y_prime = x_poly.T @ (weight * y.T).T
        x2 = x_poly.T @ x_poly
        del x_poly
        self.L, self.U = lu(x2, permute_l=True)[-2:]
        coeff = np.linalg.solve(self.U, np.linalg.solve(self.L, y_prime))
        self.coeff = coeff

    def _approximate_numpy_lstsq(self, f: Union[Callable, list[Callable]]):

        grid = self.grid.grid
        if not self.include_bias:
            print("Please be aware that the result may become significantly worse when using no intercept (bias)")
        n_samples = grid.shape[0]
        if isinstance(f, list):
            y = np.empty(shape=(n_samples, len(f)), dtype=np.float64)
            for i, func in enumerate(f):
                if not isinstance(func, Callable):
                    raise ValueError(f"One element of the list is not a function but from the type {type(func)}")
                y[:, i] = func(grid)
        else:
            y = f(grid)

        # weighted least squares
        if self.grid.rule == RandomGridRule.CHEBYSHEV:
            weight = np.empty(shape=(self.grid.get_num_points()))
            for i, row in enumerate(self.grid.grid):
                weight[i] = np.sqrt(np.prod(np.polynomial.chebyshev.chebweight(row)))

        elif self.grid.rule == RandomGridRule.UNIFORM:
            weight = np.ones(shape=(self.grid.get_num_points()), dtype=np.float64)
        else:
            raise ValueError(f"Unsupported grid type {self.grid.rule}")

        x_poly = (weight * self.basis.T).T
        del self.basis
        y_prime = (weight * y.T).T

        sol = np.linalg.lstsq(x_poly, y_prime, rcond=None)
        coeff = sol[0]

        self.coeff = coeff

    def _approximate_scipy_gelsd(self, f: Union[Callable, list[Callable]], lapack_driver: str = 'gelsd'):

        grid = self.grid.grid
        if not self.include_bias:
            print("Please be aware that the result may become significantly worse when using no intercept (bias)")
        n_samples = grid.shape[0]
        if isinstance(f, list):
            y = np.empty(shape=(n_samples, len(f)), dtype=np.float64)
            for i, func in enumerate(f):
                if not isinstance(func, Callable):
                    raise ValueError(f"One element of the list is not a function but from the type {type(func)}")
                y[:, i] = func(grid)
        else:
            y = f(grid)

        # weighted least squares
        if self.grid.rule == RandomGridRule.CHEBYSHEV:
            weight = np.empty(shape=(self.grid.get_num_points()))
            for i, row in enumerate(self.grid.grid):
                weight[i] = np.sqrt(np.prod(np.polynomial.chebyshev.chebweight(row)))

        elif self.grid.rule == RandomGridRule.UNIFORM:
            weight = np.ones(shape=(self.grid.get_num_points()), dtype=np.float64)
        else:
            raise ValueError(f"Unsupported grid type {self.grid.rule}")

        x_poly = (weight * self.basis.T).T
        del self.basis
        y_prime = (weight * y.T).T

        sol = scipy.linalg.lstsq(x_poly, y_prime, lapack_driver=lapack_driver)
        coeff = sol[0]

        self.coeff = coeff
