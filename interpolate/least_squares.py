from typing import Callable, Union, List

import numpy as np
from scipy.sparse.linalg import lsmr
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

from grid.grid import Grid
from interpolate.interpolator import Interpolator


class LeastSquaresInterpolator(Interpolator):
    def __init__(self, degree: int, include_bias: bool, grid: Grid = None, self_implemented: bool = True,
                 iterative: bool = False):
        super().__init__(grid)
        self.include_bias = include_bias
        self.degree = degree
        self.self_implemented = self_implemented
        self.iterative = iterative

    def set_self_implemented(self, self_implemented: bool):
        self.self_implemented = self_implemented

    def set_iterative(self, iterative: bool):
        self.iterative = iterative

    def interpolate(self, f: Union[Callable, List[Callable]]) -> Callable:
        assert self.grid is not None, "Grid needs to be set before interpolation"
        if self.iterative:
            return self._approximate_by_polynomial_with_least_squares_iterative(f)
        return self._approximate_by_polynomial_with_least_squares(f)

    def _approximate_by_polynomial_with_least_squares(self, f: Union[Callable, List[Callable]]) -> Callable:
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
        poly = PolynomialFeatures(degree=self.degree, include_bias=self.include_bias)
        x_poly = poly.fit_transform(grid)
        if self.self_implemented:

            y_prime = x_poly.T @ y
            del y
            x2 = x_poly.T @ x_poly
            del x_poly
            coeff = np.linalg.solve(x2, y_prime)

            def f_hat(x):
                pol = PolynomialFeatures(degree=self.degree, include_bias=self.include_bias)
                x_pol = pol.fit_transform(x)
                pred = x_pol @ coeff
                if pred.ndim == 2:
                    return pred.T
                return pred

        else:
            model = LinearRegression()
            model.fit(x_poly, y)

            def f_hat(x):
                pol = PolynomialFeatures(degree=self.degree, include_bias=self.include_bias)
                x_pol = pol.fit_transform(x)
                return model.predict(x_pol)

        return f_hat

    def _approximate_by_polynomial_with_least_squares_iterative(self, f: Callable) -> Callable:
        """
        Approximation of a function with a polynomial by least squares iterative approach (using the lsmr algorithm).
        :param f: function that needs to be approximated
        :return: fitted function
        """
        grid = self.grid.grid
        y = f(grid)
        poly = PolynomialFeatures(degree=self.degree, include_bias=self.include_bias)
        x_poly = poly.fit_transform(grid)
        res = lsmr(x_poly, y)
        coeff = res[0]

        def f_hat(x):
            pol = PolynomialFeatures(degree=self.degree, include_bias=self.include_bias)
            x_pol = pol.fit_transform(x)
            return x_pol @ coeff

        return f_hat
