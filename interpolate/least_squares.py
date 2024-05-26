from typing import Callable, Union, List, Tuple

import numpy as np
from scipy.sparse.linalg import lsmr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from grid.grid import Grid
from interpolate.basis_types import BasisType
from interpolate.interpolator import Interpolator


class LeastSquaresInterpolator(Interpolator):
    def __init__(self, include_bias: bool, basis_type: BasisType, grid: Union[Grid,None] = None,
                 self_implemented: bool = True,
                 iterative: bool = False):
        super().__init__(grid)
        self.include_bias = include_bias
        self.self_implemented = self_implemented
        self.iterative = iterative
        self.basis_type = basis_type

    def set_self_implemented(self, self_implemented: bool):
        self.self_implemented = self_implemented

    def set_iterative(self, iterative: bool):
        self.iterative = iterative

    def interpolate(self, f: Union[Callable, List[Callable]]) -> Callable:
        assert self.grid is not None, "Grid needs to be set before interpolation"

        if self.basis is None:
            self.basis = self._build_basis()
            # TODO: Also LU-decomposition here?

        if self.iterative:
            return self._approximate_iterative(f)
        return self._approximate(f)

    def _build_basis(self, basis_type: Union[BasisType, None] = None, grid: Union[None, np.ndarray] = None,
                     b_idx: Union[List[Tuple[int]], None] = None, degree: Union[int, None] = None):

        if basis_type is None:
            basis_type = self.basis_type

        if not basis_type == BasisType.CHEBYSHEV and not basis_type == BasisType.REGULAR:
            raise ValueError(f"Unsupported Basis-Type. Expected Chebyshev or Regular Basis but got {basis_type}")

        if grid is None:
            grid = self.grid.grid

        if basis_type == BasisType.CHEBYSHEV:
            return self._build_poly_basis(grid, b_idx)

        if degree is None:
            raise ValueError(
                f"Please provide a degree which specifies the max total degree that can occur in the basis")
        poly = PolynomialFeatures(degree=degree, include_bias=self.include_bias)
        return poly.fit_transform(grid)

    def _approximate(self, f: Union[Callable, List[Callable]]) -> Callable:
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
        if self.self_implemented:
            return self._self_implementation(y)
        else:
            return self._sklearn(y)

    def _self_implementation(self, y: np.ndarray) -> Callable:
        """
        Approximation of a function with a polynomial by least squares (self-implemented).
        :param y: calculated function values
        :return: fitted function
        """

        x_poly = self.basis
        y_prime = x_poly.T @ y
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
            return model.predict(data_pol)

        return f_hat

    def _approximate_iterative(self, f: Callable) -> Callable:
        """
        Approximation of a function with a polynomial by least squares iterative approach (using the lsmr algorithm).
        :param f: function that needs to be approximated
        :return: fitted function
        """

        y = f(self.grid.grid)
        x_poly = self.basis
        res = lsmr(x_poly, y)
        coeff = res[0]

        def f_hat(data: np.ndarray) -> np.ndarray:
            data_pol = self._build_basis(basis_type=None, grid=data, b_idx=self._b_idx)
            return data_pol @ coeff

        return f_hat
