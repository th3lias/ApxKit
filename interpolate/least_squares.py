from typing import Callable, Union, List, Tuple

import numpy as np
import torch
from scipy.sparse.linalg import lsmr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from grid.grid import Grid
from grid.grid_type import GridType
from interpolate.basis_types import BasisType
from interpolate.interpolator import Interpolator
from interpolate.least_squares_method import LeastSquaresMethod
from utils.utils import find_degree


class LeastSquaresInterpolator(Interpolator):
    def __init__(self, include_bias: bool, basis_type: BasisType, method: LeastSquaresMethod = LeastSquaresMethod.EXACT,
                 grid: Union[Grid, None] = None):
        super().__init__(grid)
        self.include_bias = include_bias
        self.method = method
        self.basis_type = basis_type

    def set_method(self, method: LeastSquaresMethod):
        self.method = method

    def interpolate(self, f: Union[Callable, List[Callable]]) -> Callable:
        assert self.grid is not None, "Grid needs to be set before interpolation"

        if self.basis is None:
            self.basis = self._build_basis()
            # TODO: Also LU-decomposition here?

        if self.method == LeastSquaresMethod.ITERATIVE_LSMR:
            return self._approximate_lsmr(f)
        elif self.method == LeastSquaresMethod.EXACT or self.method == LeastSquaresMethod.SKLEARN:
            return self._approximate(f)
        elif self.method == LeastSquaresMethod.PYTORCH:
            return self._approximate_pt(f)
        else:
            raise ValueError(f'The method {self.method.name} is not supported')

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

    def _approximate(self, f: Union[Callable, List[Callable]]) -> Callable:
        """
        Approximates a (or multiple) function(s) with polynomials by least squares.
        :param f: function or list of functions that need to be approximated on the same points
        :return: fitted function(s)
        """
        grid = self.grid.get_grid()
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
        if self.method == LeastSquaresMethod.EXACT:
            return self._self_implementation(y)
        elif self.method == LeastSquaresMethod.SKLEARN:
            return self._sklearn(y)
        else:
            raise ValueError(f"The method {self.method.name} is not supported")

    def _self_implementation(self, y: np.ndarray) -> Callable:
        """
        Approximation of a function with a polynomial by least squares (self-implemented).
        :param y: calculated function values
        :return: fitted function
        """

        if self.grid.grid_type == GridType.RANDOM_CHEBYSHEV:
            weight = np.empty(shape=(self.grid.get_num_points()))
            for i, row in enumerate(self.grid.get_grid()):
                weight[i] = np.prod(np.polynomial.chebyshev.chebweight(row))

            weight = np.sqrt(np.diag(weight))
        else:
            weight = np.eye(N=self.grid.get_num_points())

        x_poly = weight @ self.basis
        y_prime = x_poly.T @ weight @ y
        x2 = x_poly.T @ x_poly
        coeff = np.linalg.solve(x2, y_prime)

        def f_hat(data: np.ndarray) -> np.ndarray:
            data_pol = self._build_basis(basis_type=None, grid=data, b_idx=self._b_idx)
            y_hat = data_pol @ coeff
            if y_hat.ndim > 1:
                return y_hat.T
            return y_hat

        return f_hat

    def _sklearn(self, y: np.ndarray) -> Callable:
        """
        Approximation of a function with a polynomial by least squares (using the sklearn library).
        :param y: calculated function values
        :return: fitted function
        """

        x_poly = self.basis
        model = LinearRegression()
        model.fit(x_poly, y)

        def f_hat(data: np.ndarray) -> np.ndarray:
            data_pol = self._build_basis(basis_type=None, grid=data, b_idx=self._b_idx)
            return model.predict(data_pol).T

        return f_hat

    def _approximate_lsmr(self, f: Callable) -> Callable:
        """
        Approximation of a function with a polynomial by least squares iterative approach (using the lsmr algorithm).
        :param f: function that needs to be approximated
        :return: fitted function
        """

        grid = self.grid.grid
        if not self.include_bias:  # TODO: Maybe remove this statement (everywhere)
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

        x_poly = self.basis
        if y.ndim == 1:
            res = lsmr(x_poly, y)
            coeffs = res[0]
        else:
            for i in range(y.shape[1]):
                coeffs = np.empty((self.basis.shape[0], y.shape[1]))
                res = lsmr(x_poly, y[:, i])
                coeffs[:, i] = res[0]

        def f_hat(data: np.ndarray) -> np.ndarray:
            data_pol = self._build_basis(basis_type=None, grid=data, b_idx=self._b_idx)
            return (data_pol @ coeffs).T

        return f_hat

    def _approximate_pt(self, f: Callable, driver="gelss"):
        """
        Approximation of a function with a polynomial by least squares (using the implementation from pytorch)
        :param f: function that needs to be approximated
        :param driver: method of approximation
        :return: fitted function
        """

        grid = self.grid.grid
        if not self.include_bias:
            print("Please be aware that the result may become significantly worse when using no intercept (bias)")
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
        y = torch.tensor(y)
        x_poly = torch.tensor(self.basis)

        sol = torch.linalg.lstsq(x_poly, y, rcond=None, driver=driver)
        coeff = sol[0]

        def f_hat(data: np.ndarray) -> np.ndarray:
            data_pol = torch.tensor(self._build_basis(basis_type=None, grid=data, b_idx=self._b_idx))
            y_hat = data_pol @ coeff
            if y_hat.ndim > 1:
                return y_hat.T.numpy()
            return y_hat.numpy()

        return f_hat
