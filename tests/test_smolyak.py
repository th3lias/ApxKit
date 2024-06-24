import unittest

import numpy as np

from genz.genz_functions import get_genz_function, GenzFunctionType
from grid.grid_provider import GridProvider
from grid.grid_type import GridType
from interpolate.interpolation_methods import SmolyakMethod
from utils.utils import sample
from interpolate.smolyak import SmolyakInterpolator


class SmolyakTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(SmolyakTests, self).__init__(*args, **kwargs)
        self.scale = 3
        self.dimension = 10
        self.n_test_samples = 100
        self.lb = 0.0
        self.ub = 1.0
        self.gp = GridProvider(dimension=self.dimension, multiplier=1.0)
        self.grid = self.gp.generate(grid_type=GridType.CHEBYSHEV, scale=self.scale)
        self.test_grid = np.random.uniform(low=self.lb, high=self.ub, size=(self.n_test_samples, self.dimension))

    def test_parallel_standard(self):
        f_1 = get_genz_function(GenzFunctionType.OSCILLATORY, c=sample(dim=self.dimension),
                                w=sample(dim=self.dimension), d=self.dimension)
        f_2 = get_genz_function(GenzFunctionType.PRODUCT_PEAK, c=sample(dim=self.dimension),
                                w=sample(dim=self.dimension), d=self.dimension)
        f_3 = get_genz_function(GenzFunctionType.CONTINUOUS, c=sample(dim=self.dimension),
                                w=sample(dim=self.dimension), d=self.dimension)
        f_4 = get_genz_function(GenzFunctionType.GAUSSIAN, c=sample(dim=self.dimension),
                                w=sample(dim=self.dimension), d=self.dimension)
        f_5 = get_genz_function(GenzFunctionType.CONTINUOUS, c=sample(dim=self.dimension),
                                w=sample(dim=self.dimension), d=self.dimension)
        f_6 = get_genz_function(GenzFunctionType.DISCONTINUOUS, c=sample(dim=self.dimension),
                                w=sample(dim=self.dimension), d=self.dimension)

        f_hat_collected = [f_1, f_2, f_3, f_4, f_5, f_6]

        y_hat_individual = list()

        smolyak_interpolator = SmolyakInterpolator(grid=self.grid, method=SmolyakMethod.STANDARD)

        smolyak_interpolator.fit(f_1)
        y_hat_1 = smolyak_interpolator.interpolate(self.test_grid)
        y_hat_individual.append(y_hat_1)

        smolyak_interpolator.fit(f_2)
        y_hat_2 = smolyak_interpolator.interpolate(self.test_grid)
        y_hat_individual.append(y_hat_2)

        smolyak_interpolator.fit(f_3)
        y_hat_3 = smolyak_interpolator.interpolate(self.test_grid)
        y_hat_individual.append(y_hat_3)

        smolyak_interpolator.fit(f_4)
        y_hat_4 = smolyak_interpolator.interpolate(self.test_grid)
        y_hat_individual.append(y_hat_4)

        smolyak_interpolator.fit(f_5)
        y_hat_5 = smolyak_interpolator.interpolate(self.test_grid)
        y_hat_individual.append(y_hat_5)

        smolyak_interpolator.fit(f_6)
        y_hat_6 = smolyak_interpolator.interpolate(self.test_grid)
        y_hat_individual.append(y_hat_6)

        smolyak_interpolator.fit(f_hat_collected)
        y_hat_collected = smolyak_interpolator.interpolate(self.test_grid)

        for i in range(6):
            self.assertTrue(np.isclose(y_hat_individual[i], y_hat_collected[i]).all(), f"Not close for index {i}")

    def test_singleton_lagrange(self):
        # Only needs to be tested if there is no possibility to have multiple functions in parallel
        self.assertEqual(True, False)

    def test_parallel_largrange(self):
        # Only makes sense if it works with multiple functions in parallel (probably not)

        f_1 = get_genz_function(GenzFunctionType.OSCILLATORY, c=sample(dim=self.dimension),
                                w=sample(dim=self.dimension), d=self.dimension)
        f_2 = get_genz_function(GenzFunctionType.PRODUCT_PEAK, c=sample(dim=self.dimension),
                                w=sample(dim=self.dimension), d=self.dimension)
        f_3 = get_genz_function(GenzFunctionType.CONTINUOUS, c=sample(dim=self.dimension),
                                w=sample(dim=self.dimension), d=self.dimension)
        f_4 = get_genz_function(GenzFunctionType.GAUSSIAN, c=sample(dim=self.dimension),
                                w=sample(dim=self.dimension), d=self.dimension)
        f_5 = get_genz_function(GenzFunctionType.CONTINUOUS, c=sample(dim=self.dimension),
                                w=sample(dim=self.dimension), d=self.dimension)
        f_6 = get_genz_function(GenzFunctionType.DISCONTINUOUS, c=sample(dim=self.dimension),
                                w=sample(dim=self.dimension), d=self.dimension)

        f_hat_collected = [f_1, f_2, f_3, f_4, f_5, f_6]

        y_hat_individual = list

        smolyak_interpolator = SmolyakInterpolator(grid=self.grid, method=SmolyakMethod.LAGRANGE)

        self.assertRaises(NotImplementedError, smolyak_interpolator.fit, f_1)
        self.assertRaises(NotImplementedError, smolyak_interpolator.fit, f_2)
        self.assertRaises(NotImplementedError, smolyak_interpolator.fit, f_3)
        self.assertRaises(NotImplementedError, smolyak_interpolator.fit, f_4)
        self.assertRaises(NotImplementedError, smolyak_interpolator.fit, f_5)
        self.assertRaises(NotImplementedError, smolyak_interpolator.fit, f_6)

        # smolyak_interpolator.fit(f_1)
        # y_hat_1 = smolyak_interpolator.interpolate(self.test_grid)
        # y_hat_individual.append(y_hat_1)
        #
        # smolyak_interpolator.fit(f_2)
        # y_hat_2 = smolyak_interpolator.interpolate(self.test_grid)
        # y_hat_individual.append(y_hat_2)
        #
        # smolyak_interpolator.fit(f_3)
        # y_hat_3 = smolyak_interpolator.interpolate(self.test_grid)
        # y_hat_individual.append(y_hat_3)
        #
        # smolyak_interpolator.fit(f_4)
        # y_hat_4 = smolyak_interpolator.interpolate(self.test_grid)
        # y_hat_individual.append(y_hat_4)
        #
        # smolyak_interpolator.fit(f_5)
        # y_hat_5 = smolyak_interpolator.interpolate(self.test_grid)
        # y_hat_individual.append(y_hat_5)
        #
        # smolyak_interpolator.fit(f_6)
        # y_hat_6 = smolyak_interpolator.interpolate(self.test_grid)
        # y_hat_individual.append(y_hat_6)
        #
        # smolyak_interpolator.fit(f_hat_collected)
        # y_hat_collected = smolyak_interpolator.interpolate(self.test_grid)
        #
        # for i in range(6):
        #     self.assertTrue(np.isclose(y_hat_individual[i], y_hat_collected[i]).all(), f"Not close for index {i}")
        #
        # self.assertEqual(True, False)


if __name__ == '__main__':
    unittest.main()
