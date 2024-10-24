import unittest

import numpy as np

from grid.provider.random_grid_provider import RandomGridProvider
from grid.rule.random_grid_rule import RandomGridRule
from interpolate.basis_types import BasisType
from fit.method.least_squares_method import LeastSquaresMethod
from interpolate.least_squares import LeastSquaresInterpolator
from test_functions.functions import get_test_function, FunctionType
from utils.utils import sample


class LeastSquaresTests(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(LeastSquaresTests, self).__init__(*args, **kwargs)
        self.scale = 3
        self.dimension = 5
        self.n_test_samples = 100
        self.lb = 0.0
        self.ub = 1.0
        self.gp = RandomGridProvider(input_dim=self.dimension, lower_bound=self.lb, upper_bound=self.ub,
                                     rule=RandomGridRule.CHEBYSHEV)
        self.grid = self.gp.generate(scale=self.scale)
        self.test_grid = np.random.uniform(low=self.lb, high=self.ub, size=(self.n_test_samples, self.dimension))

    def test_parallel_exact(self):
        f_1 = get_test_function(FunctionType.OSCILLATORY, c=sample(dim=self.dimension), w=sample(dim=self.dimension),
                                d=self.dimension)
        f_2 = get_test_function(FunctionType.PRODUCT_PEAK, c=sample(dim=self.dimension), w=sample(dim=self.dimension),
                                d=self.dimension)
        f_3 = get_test_function(FunctionType.CONTINUOUS, c=sample(dim=self.dimension), w=sample(dim=self.dimension),
                                d=self.dimension)
        f_4 = get_test_function(FunctionType.GAUSSIAN, c=sample(dim=self.dimension), w=sample(dim=self.dimension),
                                d=self.dimension)
        f_5 = get_test_function(FunctionType.CONTINUOUS, c=sample(dim=self.dimension), w=sample(dim=self.dimension),
                                d=self.dimension)
        f_6 = get_test_function(FunctionType.DISCONTINUOUS, c=sample(dim=self.dimension), w=sample(dim=self.dimension),
                                d=self.dimension)

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

    def test_parallel_iterative_lsmr(self):
        f_1 = get_test_function(FunctionType.OSCILLATORY, c=sample(dim=self.dimension), w=sample(dim=self.dimension),
                                d=self.dimension)
        f_2 = get_test_function(FunctionType.PRODUCT_PEAK, c=sample(dim=self.dimension), w=sample(dim=self.dimension),
                                d=self.dimension)
        f_3 = get_test_function(FunctionType.CONTINUOUS, c=sample(dim=self.dimension), w=sample(dim=self.dimension),
                                d=self.dimension)
        f_4 = get_test_function(FunctionType.GAUSSIAN, c=sample(dim=self.dimension), w=sample(dim=self.dimension),
                                d=self.dimension)
        f_5 = get_test_function(FunctionType.CONTINUOUS, c=sample(dim=self.dimension), w=sample(dim=self.dimension),
                                d=self.dimension)
        f_6 = get_test_function(FunctionType.DISCONTINUOUS, c=sample(dim=self.dimension), w=sample(dim=self.dimension),
                                d=self.dimension)

        f_hat_collected = [f_1, f_2, f_3, f_4, f_5, f_6]

        y_hat_individual = list()

        lsq = LeastSquaresInterpolator(include_bias=True, basis_type=BasisType.CHEBYSHEV, grid=self.grid,
                                       method=LeastSquaresMethod.ITERATIVE_LSMR)

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
            self.assertTrue(np.isclose(y_hat_individual[i], y_hat_collected[i]).all(), f"Not close for index {i}")

    def test_parallel_sklearn(self):
        f_1 = get_test_function(FunctionType.OSCILLATORY, c=sample(dim=self.dimension), w=sample(dim=self.dimension),
                                d=self.dimension)
        f_2 = get_test_function(FunctionType.PRODUCT_PEAK, c=sample(dim=self.dimension), w=sample(dim=self.dimension),
                                d=self.dimension)
        f_3 = get_test_function(FunctionType.CONTINUOUS, c=sample(dim=self.dimension), w=sample(dim=self.dimension),
                                d=self.dimension)
        f_4 = get_test_function(FunctionType.GAUSSIAN, c=sample(dim=self.dimension), w=sample(dim=self.dimension),
                                d=self.dimension)
        f_5 = get_test_function(FunctionType.CONTINUOUS, c=sample(dim=self.dimension), w=sample(dim=self.dimension),
                                d=self.dimension)
        f_6 = get_test_function(FunctionType.DISCONTINUOUS, c=sample(dim=self.dimension), w=sample(dim=self.dimension),
                                d=self.dimension)

        f_hat_collected = [f_1, f_2, f_3, f_4, f_5, f_6]

        y_hat_individual = list()

        lsq = LeastSquaresInterpolator(include_bias=True, basis_type=BasisType.CHEBYSHEV, grid=self.grid,
                                       method=LeastSquaresMethod.SKLEARN)

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
            self.assertTrue(np.isclose(y_hat_individual[i], y_hat_collected[i]).all(), f"Not close for index {i}")

    def test_parallel_pytorch(self):
        f_1 = get_test_function(FunctionType.OSCILLATORY, c=sample(dim=self.dimension), w=sample(dim=self.dimension),
                                d=self.dimension)
        f_2 = get_test_function(FunctionType.PRODUCT_PEAK, c=sample(dim=self.dimension), w=sample(dim=self.dimension),
                                d=self.dimension)
        f_3 = get_test_function(FunctionType.CONTINUOUS, c=sample(dim=self.dimension), w=sample(dim=self.dimension),
                                d=self.dimension)
        f_4 = get_test_function(FunctionType.GAUSSIAN, c=sample(dim=self.dimension), w=sample(dim=self.dimension),
                                d=self.dimension)
        f_5 = get_test_function(FunctionType.CONTINUOUS, c=sample(dim=self.dimension), w=sample(dim=self.dimension),
                                d=self.dimension)
        f_6 = get_test_function(FunctionType.DISCONTINUOUS, c=sample(dim=self.dimension), w=sample(dim=self.dimension),
                                d=self.dimension)

        f_hat_collected = [f_1, f_2, f_3, f_4, f_5, f_6]

        y_hat_individual = list()

        lsq = LeastSquaresInterpolator(include_bias=True, basis_type=BasisType.CHEBYSHEV, grid=self.grid,
                                       method=LeastSquaresMethod.PYTORCH)

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
            self.assertTrue(np.isclose(y_hat_individual[i], y_hat_collected[i]).all(), f"Not close for index {i}")

    def test_singleton_pytorch_neural_net(self):
        f_1 = get_test_function(FunctionType.OSCILLATORY, c=sample(dim=self.dimension), w=sample(dim=self.dimension),
                                d=self.dimension)

        lsq = LeastSquaresInterpolator(include_bias=True, basis_type=BasisType.CHEBYSHEV, grid=self.grid,
                                       method=LeastSquaresMethod.PYTORCH_NEURAL_NET)

        lsq.fit(f_1)

        y_hat_1 = lsq.interpolate(self.test_grid)

        lsq = LeastSquaresInterpolator(include_bias=True, basis_type=BasisType.CHEBYSHEV, grid=self.grid,
                                       method=LeastSquaresMethod.EXACT)

        lsq.fit(f_1)
        y_hat_exact = lsq.interpolate(self.test_grid)

        self.assertTrue(np.isclose(y_hat_1, y_hat_exact, atol=2e0).all())

    def test_parallel_rls(self):
        f_1 = get_test_function(FunctionType.OSCILLATORY, c=sample(dim=self.dimension), w=sample(dim=self.dimension),
                                d=self.dimension)

        lsq = LeastSquaresInterpolator(include_bias=True, basis_type=BasisType.CHEBYSHEV, grid=self.grid,
                                       method=LeastSquaresMethod.RLS)

        lsq.fit(f_1)
        y_hat_1 = lsq.interpolate(self.test_grid)

        lsq = LeastSquaresInterpolator(include_bias=True, basis_type=BasisType.CHEBYSHEV, grid=self.grid,
                                       method=LeastSquaresMethod.EXACT)

        lsq.fit(f_1)
        y_hat_exact = lsq.interpolate(self.test_grid)

        self.assertTrue(np.isclose(y_hat_1, y_hat_exact, atol=1e-1).all())

    def test_singleton_iterative_rls(self):
        f_1 = get_test_function(FunctionType.OSCILLATORY, c=sample(dim=self.dimension), w=sample(dim=self.dimension),
                                d=self.dimension)

        lsq = LeastSquaresInterpolator(include_bias=True, basis_type=BasisType.CHEBYSHEV, grid=self.grid,
                                       method=LeastSquaresMethod.ITERATIVE_RLS)

        lsq.fit(f_1)
        y_hat_1 = lsq.interpolate(self.test_grid)

        lsq = LeastSquaresInterpolator(include_bias=True, basis_type=BasisType.CHEBYSHEV, grid=self.grid,
                                       method=LeastSquaresMethod.EXACT)

        lsq.fit(f_1)
        y_hat_exact = lsq.interpolate(self.test_grid)

        self.assertTrue(np.isclose(y_hat_1, y_hat_exact, atol=1e-1).all())


if __name__ == '__main__':
    unittest.main()
