import unittest
from typing import Callable

import numpy as np

from genz.genz_functions import get_genz_function, GenzFunctionType
from grid.grid import Grid
from grid.grid_provider import GridProvider
from grid.grid_type import GridType
from interpolate.least_squares import approximate_by_polynomial_with_least_squares
from interpolate.least_squares import approximate_by_polynomial_with_least_squares_iterative
from utils.utils import sample


class LeastSquaresTests(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(LeastSquaresTests, self).__init__(*args, **kwargs)
        self.degree = 3
        self.dimension = 20
        self.n_samples = 10_000
        self.n_test_samples = 100
        self.gp = GridProvider(dimension=self.dimension)
        self.grid = self.gp.generate(grid_type=GridType.RANDOM, scale=self.n_samples)
        self.test_grid = self.gp.generate(grid_type=GridType.RANDOM, scale=self.n_test_samples).grid

    def test_parallel_oscillatory(self):
        f_1 = get_genz_function(GenzFunctionType.OSCILLATORY, c=sample(dim=self.dimension),
                                w=sample(dim=self.dimension), d=self.dimension)
        f_2 = get_genz_function(GenzFunctionType.OSCILLATORY, c=sample(dim=self.dimension),
                                w=sample(dim=self.dimension), d=self.dimension)
        f_hat_1 = approximate_by_polynomial_with_least_squares(points=self.grid, f=f_1, degree=self.degree,
                                                               include_bias=True, dim=self.dimension,
                                                               self_implemented=True)
        f_hat_2 = approximate_by_polynomial_with_least_squares(points=self.grid, f=f_2, degree=self.degree,
                                                               include_bias=True, dim=self.dimension,
                                                               self_implemented=True)
        f_hat_both = approximate_by_polynomial_with_least_squares(f=[f_1, f_2], dim=self.dimension, degree=self.degree,
                                                                  points=self.grid, include_bias=True,
                                                                  self_implemented=True)

        y_hat = f_hat_both(self.test_grid)

        y_hat_1_combined = y_hat[0, :]
        y_hat_2_combined = y_hat[1, :]

        y_hat_1 = f_hat_1(self.test_grid)
        y_hat_2 = f_hat_2(self.test_grid)

        self.assertTrue(np.isclose(y_hat_1, y_hat_1_combined).all())
        self.assertTrue(np.isclose(y_hat_2, y_hat_2_combined).all())

    def test_parallel_product_peak(self):
        f_1 = get_genz_function(GenzFunctionType.PRODUCT_PEAK, c=sample(self.dimension),
                                w=sample(self.dimension), d=self.dimension)
        f_2 = get_genz_function(GenzFunctionType.PRODUCT_PEAK, c=sample(self.dimension),
                                w=sample(self.dimension), d=self.dimension)
        f_hat_1 = approximate_by_polynomial_with_least_squares(points=self.grid, f=f_1, degree=self.degree,
                                                               include_bias=True, dim=self.dimension,
                                                               self_implemented=True)
        f_hat_2 = approximate_by_polynomial_with_least_squares(points=self.grid, f=f_2, degree=self.degree,
                                                               include_bias=True, dim=self.dimension,
                                                               self_implemented=True)
        f_hat_both = approximate_by_polynomial_with_least_squares(f=[f_1, f_2], dim=self.dimension, degree=self.degree,
                                                                  points=self.grid, include_bias=True,
                                                                  self_implemented=True)

        y_hat = f_hat_both(self.test_grid)

        y_hat_1_combined = y_hat[0, :]
        y_hat_2_combined = y_hat[1, :]

        y_hat_1 = f_hat_1(self.test_grid)
        y_hat_2 = f_hat_2(self.test_grid)

        self.assertTrue(np.isclose(y_hat_1, y_hat_1_combined).all())
        self.assertTrue(np.isclose(y_hat_2, y_hat_2_combined).all())

    def test_parallel_corner_peak(self):
        f_1 = get_genz_function(GenzFunctionType.CORNER_PEAK, c=sample(self.dimension),
                                w=sample(self.dimension), d=self.dimension)
        f_2 = get_genz_function(GenzFunctionType.CORNER_PEAK, c=sample(self.dimension),
                                w=sample(self.dimension), d=self.dimension)

        f_hat_1 = approximate_by_polynomial_with_least_squares(points=self.grid, f=f_1, degree=self.degree,
                                                               include_bias=True, dim=self.dimension,
                                                               self_implemented=True)
        f_hat_2 = approximate_by_polynomial_with_least_squares(points=self.grid, f=f_2, degree=self.degree,
                                                               include_bias=True, dim=self.dimension,
                                                               self_implemented=True)
        f_hat_both = approximate_by_polynomial_with_least_squares(f=[f_1, f_2], dim=self.dimension, degree=self.degree,
                                                                  points=self.grid, include_bias=True,
                                                                  self_implemented=True)

        y_hat = f_hat_both(self.test_grid)

        y_hat_1_combined = y_hat[0, :]
        y_hat_2_combined = y_hat[1, :]

        y_hat_1 = f_hat_1(self.test_grid)
        y_hat_2 = f_hat_2(self.test_grid)

        self.assertTrue(np.isclose(y_hat_1, y_hat_1_combined).all())
        self.assertTrue(np.isclose(y_hat_2, y_hat_2_combined).all())

    def test_parallel_gaussian(self):
        f_1 = get_genz_function(GenzFunctionType.GAUSSIAN, c=sample(self.dimension),
                                w=sample(self.dimension), d=self.dimension)
        f_2 = get_genz_function(GenzFunctionType.GAUSSIAN, c=sample(self.dimension),
                                w=sample(self.dimension), d=self.dimension)
        f_hat_1 = approximate_by_polynomial_with_least_squares(points=self.grid, f=f_1, degree=self.degree,
                                                               include_bias=True, dim=self.dimension,
                                                               self_implemented=True)
        f_hat_2 = approximate_by_polynomial_with_least_squares(points=self.grid, f=f_2, degree=self.degree,
                                                               include_bias=True, dim=self.dimension,
                                                               self_implemented=True)
        f_hat_both = approximate_by_polynomial_with_least_squares(f=[f_1, f_2], dim=self.dimension, degree=self.degree,
                                                                  points=self.grid, include_bias=True,
                                                                  self_implemented=True)

        y_hat = f_hat_both(self.test_grid)

        y_hat_1_combined = y_hat[0, :]
        y_hat_2_combined = y_hat[1, :]

        y_hat_1 = f_hat_1(self.test_grid)
        y_hat_2 = f_hat_2(self.test_grid)

        self.assertTrue(np.isclose(y_hat_1, y_hat_1_combined).all())
        self.assertTrue(np.isclose(y_hat_2, y_hat_2_combined).all())

    def test_parallel_continuous(self):
        f_1 = get_genz_function(GenzFunctionType.CONTINUOUS, c=sample(self.dimension),
                                w=sample(self.dimension), d=self.dimension)
        f_2 = get_genz_function(GenzFunctionType.CONTINUOUS, c=sample(self.dimension),
                                w=sample(self.dimension), d=self.dimension)
        f_hat_1 = approximate_by_polynomial_with_least_squares(points=self.grid, f=f_1, degree=self.degree,
                                                               include_bias=True, dim=self.dimension,
                                                               self_implemented=True)
        f_hat_2 = approximate_by_polynomial_with_least_squares(points=self.grid, f=f_2, degree=self.degree,
                                                               include_bias=True, dim=self.dimension,
                                                               self_implemented=True)
        f_hat_both = approximate_by_polynomial_with_least_squares(f=[f_1, f_2], dim=self.dimension, degree=self.degree,
                                                                  points=self.grid, include_bias=True,
                                                                  self_implemented=True)

        y_hat = f_hat_both(self.test_grid)

        y_hat_1_combined = y_hat[0, :]
        y_hat_2_combined = y_hat[1, :]

        y_hat_1 = f_hat_1(self.test_grid)
        y_hat_2 = f_hat_2(self.test_grid)

        self.assertTrue(np.isclose(y_hat_1, y_hat_1_combined).all())
        self.assertTrue(np.isclose(y_hat_2, y_hat_2_combined).all())

    def test_parallel_discontinuous(self):
        f_1 = get_genz_function(GenzFunctionType.DISCONTINUOUS, c=sample(self.dimension),
                                w=sample(self.dimension), d=self.dimension)
        f_2 = get_genz_function(GenzFunctionType.DISCONTINUOUS, c=sample(self.dimension),
                                w=sample(self.dimension), d=self.dimension)
        f_hat_1 = approximate_by_polynomial_with_least_squares(points=self.grid, f=f_1, degree=self.degree,
                                                               include_bias=True, dim=self.dimension,
                                                               self_implemented=True)
        f_hat_2 = approximate_by_polynomial_with_least_squares(points=self.grid, f=f_2, degree=self.degree,
                                                               include_bias=True, dim=self.dimension,
                                                               self_implemented=True)
        f_hat_both = approximate_by_polynomial_with_least_squares(f=[f_1, f_2], dim=self.dimension, degree=self.degree,
                                                                  points=self.grid, include_bias=True,
                                                                  self_implemented=True)

        y_hat = f_hat_both(self.test_grid)

        y_hat_1_combined = y_hat[0, :]
        y_hat_2_combined = y_hat[1, :]

        y_hat_1 = f_hat_1(self.test_grid)
        y_hat_2 = f_hat_2(self.test_grid)

        self.assertTrue(np.isclose(y_hat_1, y_hat_1_combined).all())
        self.assertTrue(np.isclose(y_hat_2, y_hat_2_combined).all())

    def test_self_implemented_oscillatory(self):
        f = get_genz_function(GenzFunctionType.OSCILLATORY, c=sample(self.dimension),
                              w=sample(self.dimension), d=self.dimension)
        f_hat_self, f_hat_sklearn, f_hat_iterative = self._approximate(f, self.grid, True, self.degree, self.dimension)

        y_hat_self = f_hat_self(self.test_grid)
        y_hat_sklearn = f_hat_sklearn(self.test_grid)
        y_hat_iterative = f_hat_iterative(self.test_grid)

        self.assertTrue(np.isclose(y_hat_self, y_hat_sklearn, atol=1e-3).all())
        self.assertTrue(np.isclose(y_hat_self, y_hat_iterative, atol=1e-2).all())

    def test_self_implemented_product_peak(self):
        f = get_genz_function(GenzFunctionType.PRODUCT_PEAK, c=sample(self.dimension),
                              w=sample(self.dimension), d=self.dimension)
        f_hat_self, f_hat_sklearn, f_hat_iterative = self._approximate(f, self.grid, True, self.degree, self.dimension)

        y_hat_self = f_hat_self(self.test_grid)
        y_hat_sklearn = f_hat_sklearn(self.test_grid)
        y_hat_iterative = f_hat_iterative(self.test_grid)

        self.assertTrue(np.isclose(y_hat_self, y_hat_sklearn).all())
        self.assertTrue(np.isclose(y_hat_self, y_hat_iterative).all())

    def test_self_implemented_corner_peak(self):
        f = get_genz_function(GenzFunctionType.CORNER_PEAK, c=sample(self.dimension),
                              w=sample(self.dimension), d=self.dimension)
        f_hat_self, f_hat_sklearn, f_hat_iterative = self._approximate(f, self.grid, True, self.degree, self.dimension)

        y_hat_self = f_hat_self(self.test_grid)
        y_hat_sklearn = f_hat_sklearn(self.test_grid)
        y_hat_iterative = f_hat_iterative(self.test_grid)

        self.assertTrue(np.isclose(y_hat_self, y_hat_sklearn).all())
        self.assertTrue(np.isclose(y_hat_self, y_hat_iterative).all())

    def test_self_implemented_gaussian(self):
        f = get_genz_function(GenzFunctionType.GAUSSIAN, c=sample(self.dimension),
                              w=sample(self.dimension), d=self.dimension)
        f_hat_self, f_hat_sklearn, f_hat_iterative = self._approximate(f, self.grid, True, self.degree, self.dimension)

        y_hat_self = f_hat_self(self.test_grid)
        y_hat_sklearn = f_hat_sklearn(self.test_grid)
        y_hat_iterative = f_hat_iterative(self.test_grid)

        self.assertTrue(np.isclose(y_hat_self, y_hat_sklearn, atol=1e-3).all())
        self.assertTrue(np.isclose(y_hat_self, y_hat_iterative, atol=1e-3).all())

    def test_self_implemented_continuous(self):
        f = get_genz_function(GenzFunctionType.CONTINUOUS, c=sample(self.dimension),
                              w=sample(self.dimension), d=self.dimension)
        f_hat_self, f_hat_sklearn, f_hat_iterative = self._approximate(f, self.grid, True, self.degree, self.dimension)

        y_hat_self = f_hat_self(self.test_grid)
        y_hat_sklearn = f_hat_sklearn(self.test_grid)
        y_hat_iterative = f_hat_iterative(self.test_grid)

        self.assertTrue(np.isclose(y_hat_self, y_hat_sklearn, atol=1e-3).all())
        self.assertTrue(np.isclose(y_hat_self, y_hat_iterative, atol=1e-3).all())

    def test_self_implemented_discontinuous(self):
        f = get_genz_function(GenzFunctionType.DISCONTINUOUS, c=sample(self.dimension),
                              w=sample(self.dimension), d=self.dimension)
        f_hat_self, f_hat_sklearn, f_hat_iterative = self._approximate(f, self.grid, True, self.degree, self.dimension)

        y_hat_self = f_hat_self(self.test_grid)
        y_hat_sklearn = f_hat_sklearn(self.test_grid)
        y_hat_iterative = f_hat_iterative(self.test_grid)

        self.assertTrue(np.isclose(y_hat_self, y_hat_sklearn, atol=1).all())
        self.assertTrue(np.isclose(y_hat_self, y_hat_iterative, atol=4).all())

    @staticmethod
    def _approximate(f: Callable, grid: Grid, include_bias: bool, degree: int, dim: int):
        f_hat_self = approximate_by_polynomial_with_least_squares(points=grid, f=f, degree=degree,
                                                                  include_bias=include_bias, dim=dim,
                                                                  self_implemented=True)
        f_hat_sklearn = approximate_by_polynomial_with_least_squares(points=grid, f=f, degree=degree,
                                                                     include_bias=include_bias, dim=dim,
                                                                     self_implemented=False)
        f_hat_iterative = approximate_by_polynomial_with_least_squares_iterative(f=f, dim=dim, degree=degree,
                                                                                 grid=grid.grid,
                                                                                 include_bias=include_bias)
        return f_hat_self, f_hat_sklearn, f_hat_iterative


if __name__ == '__main__':
    unittest.main()
