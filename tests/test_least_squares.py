import unittest
from typing import Callable

import numpy as np

from genz.genz_functions import get_genz_function, GenzFunctionType
from grid.grid_provider import GridProvider
from grid.grid_type import GridType
from interpolate.least_squares import LeastSquaresInterpolator
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
        self.lsq = LeastSquaresInterpolator(self.degree, True, self.grid)

    def test_parallel_oscillatory(self):
        f_1 = get_genz_function(GenzFunctionType.OSCILLATORY, c=sample(dim=self.dimension),
                                w=sample(dim=self.dimension), d=self.dimension)
        f_2 = get_genz_function(GenzFunctionType.OSCILLATORY, c=sample(dim=self.dimension),
                                w=sample(dim=self.dimension), d=self.dimension)

        f_hat_1 = self.lsq.interpolate(f_1)
        f_hat_2 = self.lsq.interpolate(f_2)
        f_hat_both = self.lsq.interpolate([f_1, f_2])

        y_hat = f_hat_both(self.test_grid)

        y_hat_1_combined = y_hat[0, :]
        y_hat_2_combined = y_hat[1, :]

        y_hat_1 = f_hat_1(self.test_grid)
        y_hat_2 = f_hat_2(self.test_grid)

        self.assertTrue(np.isclose(y_hat_1, y_hat_1_combined).all())
        self.assertTrue(np.isclose(y_hat_2, y_hat_2_combined).all())

    def test_parallel_product_peak(self):
        f_1 = get_genz_function(GenzFunctionType.PRODUCT_PEAK, c=sample(self.dimension), w=sample(self.dimension),
                                d=self.dimension)
        f_2 = get_genz_function(GenzFunctionType.PRODUCT_PEAK, c=sample(self.dimension), w=sample(self.dimension),
                                d=self.dimension)

        f_hat_1 = self.lsq.interpolate(f_1)
        f_hat_2 = self.lsq.interpolate(f_2)
        f_hat_both = self.lsq.interpolate([f_1, f_2])

        y_hat = f_hat_both(self.test_grid)

        y_hat_1_combined = y_hat[0, :]
        y_hat_2_combined = y_hat[1, :]

        y_hat_1 = f_hat_1(self.test_grid)
        y_hat_2 = f_hat_2(self.test_grid)

        self.assertTrue(np.isclose(y_hat_1, y_hat_1_combined).all())
        self.assertTrue(np.isclose(y_hat_2, y_hat_2_combined).all())

    def test_parallel_corner_peak(self):
        f_1 = get_genz_function(GenzFunctionType.CORNER_PEAK, c=sample(self.dimension), w=sample(self.dimension),
                                d=self.dimension)
        f_2 = get_genz_function(GenzFunctionType.CORNER_PEAK, c=sample(self.dimension), w=sample(self.dimension),
                                d=self.dimension)

        f_hat_1 = self.lsq.interpolate(f_1)
        f_hat_2 = self.lsq.interpolate(f_2)
        f_hat_both = self.lsq.interpolate([f_1, f_2])

        y_hat = f_hat_both(self.test_grid)

        y_hat_1_combined = y_hat[0, :]
        y_hat_2_combined = y_hat[1, :]

        y_hat_1 = f_hat_1(self.test_grid)
        y_hat_2 = f_hat_2(self.test_grid)

        self.assertTrue(np.isclose(y_hat_1, y_hat_1_combined).all())
        self.assertTrue(np.isclose(y_hat_2, y_hat_2_combined).all())

    def test_parallel_gaussian(self):
        f_1 = get_genz_function(GenzFunctionType.GAUSSIAN, c=sample(self.dimension), w=sample(self.dimension),
                                d=self.dimension)
        f_2 = get_genz_function(GenzFunctionType.GAUSSIAN, c=sample(self.dimension), w=sample(self.dimension),
                                d=self.dimension)

        f_hat_1 = self.lsq.interpolate(f_1)
        f_hat_2 = self.lsq.interpolate(f_2)
        f_hat_both = self.lsq.interpolate([f_1, f_2])

        y_hat = f_hat_both(self.test_grid)

        y_hat_1_combined = y_hat[0, :]
        y_hat_2_combined = y_hat[1, :]

        y_hat_1 = f_hat_1(self.test_grid)
        y_hat_2 = f_hat_2(self.test_grid)

        self.assertTrue(np.isclose(y_hat_1, y_hat_1_combined).all())
        self.assertTrue(np.isclose(y_hat_2, y_hat_2_combined).all())

    def test_parallel_continuous(self):
        f_1 = get_genz_function(GenzFunctionType.CONTINUOUS, c=sample(self.dimension), w=sample(self.dimension),
                                d=self.dimension)
        f_2 = get_genz_function(GenzFunctionType.CONTINUOUS, c=sample(self.dimension), w=sample(self.dimension),
                                d=self.dimension)

        f_hat_1 = self.lsq.interpolate(f_1)
        f_hat_2 = self.lsq.interpolate(f_2)
        f_hat_both = self.lsq.interpolate([f_1, f_2])

        y_hat = f_hat_both(self.test_grid)

        y_hat_1_combined = y_hat[0, :]
        y_hat_2_combined = y_hat[1, :]

        y_hat_1 = f_hat_1(self.test_grid)
        y_hat_2 = f_hat_2(self.test_grid)

        self.assertTrue(np.isclose(y_hat_1, y_hat_1_combined).all())
        self.assertTrue(np.isclose(y_hat_2, y_hat_2_combined).all())

    def test_parallel_discontinuous(self):
        f_1 = get_genz_function(GenzFunctionType.DISCONTINUOUS, c=sample(self.dimension), w=sample(self.dimension),
                                d=self.dimension)
        f_2 = get_genz_function(GenzFunctionType.DISCONTINUOUS, c=sample(self.dimension), w=sample(self.dimension),
                                d=self.dimension)

        f_hat_1 = self.lsq.interpolate(f_1)
        f_hat_2 = self.lsq.interpolate(f_2)
        f_hat_both = self.lsq.interpolate([f_1, f_2])

        y_hat = f_hat_both(self.test_grid)

        y_hat_1_combined = y_hat[0, :]
        y_hat_2_combined = y_hat[1, :]

        y_hat_1 = f_hat_1(self.test_grid)
        y_hat_2 = f_hat_2(self.test_grid)

        self.assertTrue(np.isclose(y_hat_1, y_hat_1_combined).all())
        self.assertTrue(np.isclose(y_hat_2, y_hat_2_combined).all())

    def test_self_implemented_oscillatory(self):
        f = get_genz_function(GenzFunctionType.OSCILLATORY, c=sample(self.dimension), w=sample(self.dimension),
                              d=self.dimension)
        f_hat_self, f_hat_sklearn, f_hat_iterative = self._approximate(f)

        y_hat_self = f_hat_self(self.test_grid)
        y_hat_sklearn = f_hat_sklearn(self.test_grid)
        y_hat_iterative = f_hat_iterative(self.test_grid)

        self.assertTrue(np.isclose(y_hat_self, y_hat_sklearn, atol=1e-3).all())
        self.assertTrue(np.isclose(y_hat_self, y_hat_iterative, atol=1e-2).all())

    def test_self_implemented_product_peak(self):
        f = get_genz_function(GenzFunctionType.PRODUCT_PEAK, c=sample(self.dimension), w=sample(self.dimension),
                              d=self.dimension)
        f_hat_self, f_hat_sklearn, f_hat_iterative = self._approximate(f)

        y_hat_self = f_hat_self(self.test_grid)
        y_hat_sklearn = f_hat_sklearn(self.test_grid)
        y_hat_iterative = f_hat_iterative(self.test_grid)

        self.assertTrue(np.isclose(y_hat_self, y_hat_sklearn).all())
        self.assertTrue(np.isclose(y_hat_self, y_hat_iterative).all())

    def test_self_implemented_corner_peak(self):
        f = get_genz_function(GenzFunctionType.CORNER_PEAK, c=sample(self.dimension), w=sample(self.dimension),
                              d=self.dimension)
        f_hat_self, f_hat_sklearn, f_hat_iterative = self._approximate(f)

        y_hat_self = f_hat_self(self.test_grid)
        y_hat_sklearn = f_hat_sklearn(self.test_grid)
        y_hat_iterative = f_hat_iterative(self.test_grid)

        self.assertTrue(np.isclose(y_hat_self, y_hat_sklearn).all())
        self.assertTrue(np.isclose(y_hat_self, y_hat_iterative).all())

    def test_self_implemented_gaussian(self):
        f = get_genz_function(GenzFunctionType.GAUSSIAN, c=sample(self.dimension), w=sample(self.dimension),
                              d=self.dimension)
        f_hat_self, f_hat_sklearn, f_hat_iterative = self._approximate(f)

        y_hat_self = f_hat_self(self.test_grid)
        y_hat_sklearn = f_hat_sklearn(self.test_grid)
        y_hat_iterative = f_hat_iterative(self.test_grid)

        self.assertTrue(np.isclose(y_hat_self, y_hat_sklearn, atol=1e-3).all())
        self.assertTrue(np.isclose(y_hat_self, y_hat_iterative, atol=1e-3).all())

    def test_self_implemented_continuous(self):
        f = get_genz_function(GenzFunctionType.CONTINUOUS, c=sample(self.dimension), w=sample(self.dimension),
                              d=self.dimension)
        f_hat_self, f_hat_sklearn, f_hat_iterative = self._approximate(f)

        y_hat_self = f_hat_self(self.test_grid)
        y_hat_sklearn = f_hat_sklearn(self.test_grid)
        y_hat_iterative = f_hat_iterative(self.test_grid)

        self.assertTrue(np.isclose(y_hat_self, y_hat_sklearn, atol=1e-3).all())
        self.assertTrue(np.isclose(y_hat_self, y_hat_iterative, atol=1e-3).all())

    def test_self_implemented_discontinuous(self):
        f = get_genz_function(GenzFunctionType.DISCONTINUOUS, c=sample(self.dimension), w=sample(self.dimension),
                              d=self.dimension)
        f_hat_self, f_hat_sklearn, f_hat_iterative = self._approximate(f)

        y_hat_self = f_hat_self(self.test_grid)
        y_hat_sklearn = f_hat_sklearn(self.test_grid)
        y_hat_iterative = f_hat_iterative(self.test_grid)

        self.assertTrue(np.isclose(y_hat_self, y_hat_sklearn, atol=1).all())
        self.assertTrue(np.isclose(y_hat_self, y_hat_iterative, atol=4).all())

    def _approximate(self, f: Callable):
        self.lsq.set_self_implemented(True)
        f_hat_self = self.lsq.interpolate(f)
        self.lsq.set_self_implemented(False)
        f_hat_sklearn = self.lsq.interpolate(f)
        self.lsq.set_iterative(True)
        f_hat_iterative = self.lsq.interpolate(f)
        return f_hat_self, f_hat_sklearn, f_hat_iterative


if __name__ == '__main__':
    unittest.main()
