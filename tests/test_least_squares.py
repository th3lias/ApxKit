import unittest

import numpy as np

from test_functions.functions import get_test_function, FunctionType
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
        self.dimension = 5
        self.n_test_samples = 100
        self.lb = 0.0
        self.ub = 1.0
        self.gp = GridProvider(dimension=self.dimension, multiplier_fun=lambda x: x)
        self.grid = self.gp.generate(grid_type=GridType.RANDOM_CHEBYSHEV, scale=self.scale)
        self.test_grid = np.random.uniform(low=self.lb, high=self.ub, size=(self.n_test_samples, self.dimension))

    def test_parallel_exact(self):
        f_1 = get_test_function(FunctionType.OSCILLATORY, c=sample(dim=self.dimension),
                                w=sample(dim=self.dimension), d=self.dimension)
        f_2 = get_test_function(FunctionType.PRODUCT_PEAK, c=sample(dim=self.dimension),
                                w=sample(dim=self.dimension), d=self.dimension)
        f_3 = get_test_function(FunctionType.CONTINUOUS, c=sample(dim=self.dimension),
                                w=sample(dim=self.dimension), d=self.dimension)
        f_4 = get_test_function(FunctionType.GAUSSIAN, c=sample(dim=self.dimension),
                                w=sample(dim=self.dimension), d=self.dimension)
        f_5 = get_test_function(FunctionType.CONTINUOUS, c=sample(dim=self.dimension),
                                w=sample(dim=self.dimension), d=self.dimension)
        f_6 = get_test_function(FunctionType.DISCONTINUOUS, c=sample(dim=self.dimension),
                                w=sample(dim=self.dimension), d=self.dimension)

        f_hat_collected = [f_1, f_2, f_3, f_4, f_5, f_6]

        y_hat_individual = list()

        lsq = LeastSquaresInterpolator(include_bias=True, basis_type=BasisType.CHEBYSHEV, grid=self.grid,
                                       method=LeastSquaresMethod.EXACT)

        lsq.fit(f_1)
        y_hat_1 = lsq.interpolate(self.test_grid)
        y_hat_individual.append(y_hat_1)

        lsq.fit(f_2)
        y_hat_2 = lsq.interpolate(self.test_grid)
        y_hat_individual.append(y_hat_2)

        lsq.fit(f_3)
        y_hat_3 = lsq.interpolate(self.test_grid)
        y_hat_individual.append(y_hat_3)

        lsq.fit(f_4)
        y_hat_4 = lsq.interpolate(self.test_grid)
        y_hat_individual.append(y_hat_4)

        lsq.fit(f_5)
        y_hat_5 = lsq.interpolate(self.test_grid)
        y_hat_individual.append(y_hat_5)

        lsq.fit(f_6)
        y_hat_6 = lsq.interpolate(self.test_grid)
        y_hat_individual.append(y_hat_6)

        lsq.fit(f_hat_collected)
        y_hat_collected = lsq.interpolate(self.test_grid)

        for i in range(6):
            self.assertTrue(np.isclose(y_hat_individual[i], y_hat_collected[i], rtol=1e-3).all(),
                            f"Not close for index {i}")

if __name__ == '__main__':
    unittest.main()
