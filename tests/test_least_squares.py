import unittest
from typing import Callable

import numpy as np

from genz.genz_functions import get_genz_function, GenzFunctionType
from grid.grid_provider import GridProvider
from grid.grid_type import GridType
from interpolate.least_squares import LeastSquaresInterpolator
from interpolate.interpolation_methods import LeastSquaresMethod
from utils.utils import sample
from interpolate.basis_types import BasisType


class LeastSquaresTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(LeastSquaresTests, self).__init__(*args, **kwargs)
        self.scale = 3
        self.dimension = 10
        self.n_test_samples = 100
        self.lb = 0.0
        self.ub = 1.0
        self.gp = GridProvider(dimension=self.dimension)
        self.grid = self.gp.generate(grid_type=GridType.RANDOM_CHEBYSHEV, scale=self.scale)
        self.test_grid = np.random.uniform(low=self.lb, high=self.ub, size=(self.n_test_samples, self.dimension))
        self.lsq = LeastSquaresInterpolator(True, basis_type=BasisType.CHEBYSHEV, grid=self.grid)

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
        f_hat_self, f_hat_sklearn, f_hat_iterative_lsmr, f_hat_pytorch = self._approximate(f)

        y_hat_self = f_hat_self(self.test_grid)
        y_hat_sklearn = f_hat_sklearn(self.test_grid)
        y_hat_iterative = f_hat_iterative_lsmr(self.test_grid)
        y_hat_pytorch = f_hat_pytorch(self.test_grid)

        self.assertTrue(np.isclose(y_hat_self, y_hat_sklearn, atol=1).all())
        self.assertTrue(np.isclose(y_hat_self, y_hat_iterative, atol=4).all())
        self.assertTrue(np.isclose(y_hat_self, y_hat_pytorch, atol=4).all())

    def test_self_implemented_product_peak(self):
        f = get_genz_function(GenzFunctionType.PRODUCT_PEAK, c=sample(self.dimension), w=sample(self.dimension),
                              d=self.dimension)
        f_hat_self, f_hat_sklearn, f_hat_iterative_lsmr, f_hat_pytorch = self._approximate(f)

        y_hat_self = f_hat_self(self.test_grid)
        y_hat_sklearn = f_hat_sklearn(self.test_grid)
        y_hat_iterative = f_hat_iterative_lsmr(self.test_grid)
        y_hat_pytorch = f_hat_pytorch(self.test_grid)

        self.assertTrue(np.isclose(y_hat_self, y_hat_sklearn, atol=1).all())
        self.assertTrue(np.isclose(y_hat_self, y_hat_iterative, atol=4).all())
        self.assertTrue(np.isclose(y_hat_self, y_hat_pytorch, atol=4).all())

    def test_self_implemented_corner_peak(self):
        f = get_genz_function(GenzFunctionType.CORNER_PEAK, c=sample(self.dimension), w=sample(self.dimension),
                              d=self.dimension)
        f_hat_self, f_hat_sklearn, f_hat_iterative_lsmr, f_hat_pytorch = self._approximate(f)

        y_hat_self = f_hat_self(self.test_grid)
        y_hat_sklearn = f_hat_sklearn(self.test_grid)
        y_hat_iterative = f_hat_iterative_lsmr(self.test_grid)
        y_hat_pytorch = f_hat_pytorch(self.test_grid)

        self.assertTrue(np.isclose(y_hat_self, y_hat_sklearn, atol=1).all())
        self.assertTrue(np.isclose(y_hat_self, y_hat_iterative, atol=4).all())
        self.assertTrue(np.isclose(y_hat_self, y_hat_pytorch, atol=4).all())

    def test_self_implemented_gaussian(self):
        f = get_genz_function(GenzFunctionType.GAUSSIAN, c=sample(self.dimension), w=sample(self.dimension),
                              d=self.dimension)
        f_hat_self, f_hat_sklearn, f_hat_iterative_lsmr, f_hat_pytorch = self._approximate(f)

        y_hat_self = f_hat_self(self.test_grid)
        y_hat_sklearn = f_hat_sklearn(self.test_grid)
        y_hat_iterative = f_hat_iterative_lsmr(self.test_grid)
        y_hat_pytorch = f_hat_pytorch(self.test_grid)

        self.assertTrue(np.isclose(y_hat_self, y_hat_sklearn, atol=1).all())
        self.assertTrue(np.isclose(y_hat_self, y_hat_iterative, atol=4).all())
        self.assertTrue(np.isclose(y_hat_self, y_hat_pytorch, atol=4).all())

    def test_self_implemented_continuous(self):
        f = get_genz_function(GenzFunctionType.CONTINUOUS, c=sample(self.dimension), w=sample(self.dimension),
                              d=self.dimension)
        f_hat_self, f_hat_sklearn, f_hat_iterative_lsmr, f_hat_pytorch = self._approximate(f)

        y_hat_self = f_hat_self(self.test_grid)
        y_hat_sklearn = f_hat_sklearn(self.test_grid)
        y_hat_iterative = f_hat_iterative_lsmr(self.test_grid)
        y_hat_pytorch = f_hat_pytorch(self.test_grid)

        self.assertTrue(np.isclose(y_hat_self, y_hat_sklearn, atol=1).all())
        self.assertTrue(np.isclose(y_hat_self, y_hat_iterative, atol=4).all())
        self.assertTrue(np.isclose(y_hat_self, y_hat_pytorch, atol=4).all())

    def test_self_implemented_discontinuous(self):
        f = get_genz_function(GenzFunctionType.DISCONTINUOUS, c=sample(self.dimension), w=sample(self.dimension),
                              d=self.dimension)
        f_hat_self, f_hat_sklearn, f_hat_iterative_lsmr, f_hat_pytorch = self._approximate(f)

        y_hat_self = f_hat_self(self.test_grid)
        y_hat_sklearn = f_hat_sklearn(self.test_grid)
        y_hat_iterative = f_hat_iterative_lsmr(self.test_grid)
        y_hat_pytorch = f_hat_pytorch(self.test_grid)

        self.assertTrue(np.isclose(y_hat_self, y_hat_sklearn, atol=1).all())
        self.assertTrue(np.isclose(y_hat_self, y_hat_iterative, atol=4).all())
        self.assertTrue(np.isclose(y_hat_self, y_hat_pytorch, atol=4).all())

    def _approximate(self, f: Callable):
        self.lsq.set_method(LeastSquaresMethod.EXACT)
        f_hat_self = self.lsq.interpolate(f)
        self.lsq.set_method(LeastSquaresMethod.SKLEARN)
        f_hat_sklearn = self.lsq.interpolate(f)
        self.lsq.set_method(LeastSquaresMethod.ITERATIVE_LSMR)
        f_hat_lsmr = self.lsq.interpolate(f)
        self.lsq.set_method(LeastSquaresMethod.PYTORCH)
        f_hat_pytorch = self.lsq.interpolate(f)
        return f_hat_self, f_hat_sklearn, f_hat_lsmr, f_hat_pytorch


if __name__ == '__main__':
    unittest.main()
