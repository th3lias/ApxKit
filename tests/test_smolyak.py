import unittest

import numpy as np

from grid.provider.rule_grid_provider import RuleGridProvider
from fit.method.interpolation_method import InterpolationMethod
from interpolate.smolyak import SmolyakInterpolator
from function.provider import ParametrizedFunctionProvider
from function.type import FunctionType

from utils.utils import sample


class SmolyakTests(unittest.TestCase):
    """
    Tests if the self-implemented version of the smolyak algorithm works with multiple functions at once
    """

    def __init__(self, *args, **kwargs):
        super(SmolyakTests, self).__init__(*args, **kwargs)
        self.scale = 3
        self.dimension = 10
        self.n_test_samples = 100
        self.lb = 0.0
        self.ub = 1.0
        self.gp = RuleGridProvider(input_dim=self.dimension)
        self.grid = self.gp.generate(scale=self.scale)
        self.test_grid = np.random.uniform(low=self.lb, high=self.ub, size=(self.n_test_samples, self.dimension))

    def test_parallel_standard(self):
        f_1 = ParametrizedFunctionProvider.get_function(FunctionType.OSCILLATORY, c=sample(dim=self.dimension),
                                                        w=sample(dim=self.dimension), d=self.dimension)
        f_2 = ParametrizedFunctionProvider.get_function(FunctionType.PRODUCT_PEAK, c=sample(dim=self.dimension),
                                                        w=sample(dim=self.dimension), d=self.dimension)
        f_3 = ParametrizedFunctionProvider.get_function(FunctionType.CONTINUOUS, c=sample(dim=self.dimension),
                                                        w=sample(dim=self.dimension), d=self.dimension)
        f_4 = ParametrizedFunctionProvider.get_function(FunctionType.GAUSSIAN, c=sample(dim=self.dimension),
                                                        w=sample(dim=self.dimension), d=self.dimension)
        f_5 = ParametrizedFunctionProvider.get_function(FunctionType.CORNER_PEAK, c=sample(dim=self.dimension),
                                                        w=sample(dim=self.dimension), d=self.dimension)
        f_6 = ParametrizedFunctionProvider.get_function(FunctionType.DISCONTINUOUS, c=sample(dim=self.dimension),
                                                        w=sample(dim=self.dimension), d=self.dimension)
        f_7 = ParametrizedFunctionProvider.get_function(FunctionType.G_FUNCTION, c=sample(dim=self.dimension),
                                                        w=sample(dim=self.dimension), d=self.dimension)
        f_8 = ParametrizedFunctionProvider.get_function(FunctionType.MOROKOFF_CALFISCH_1, c=sample(dim=self.dimension),
                                                        w=sample(dim=self.dimension), d=self.dimension)
        f_9 = ParametrizedFunctionProvider.get_function(FunctionType.MOROKOFF_CALFISCH_2, c=sample(dim=self.dimension),
                                                        w=sample(dim=self.dimension), d=self.dimension)
        f_10 = ParametrizedFunctionProvider.get_function(FunctionType.BRATLEY, c=sample(dim=self.dimension),
                                                         w=sample(dim=self.dimension), d=self.dimension)
        f_11 = ParametrizedFunctionProvider.get_function(FunctionType.ROOS_ARNOLD, c=sample(dim=self.dimension),
                                                         w=sample(dim=self.dimension), d=self.dimension)
        f_12 = ParametrizedFunctionProvider.get_function(FunctionType.ZHOU, c=sample(dim=self.dimension),
                                                         w=sample(dim=self.dimension), d=self.dimension)

        f_hat_collected = [f_1, f_2, f_3, f_4, f_5, f_6, f_7, f_8, f_9, f_10, f_11, f_12]

        y_hat_individual = list()

        smolyak_interpolator = SmolyakInterpolator(grid=self.grid, method=InterpolationMethod.STANDARD)

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

        smolyak_interpolator.fit(f_7)
        y_hat_7 = smolyak_interpolator.interpolate(self.test_grid)
        y_hat_individual.append(y_hat_7)

        smolyak_interpolator.fit(f_8)
        y_hat_8 = smolyak_interpolator.interpolate(self.test_grid)
        y_hat_individual.append(y_hat_8)

        smolyak_interpolator.fit(f_9)
        y_hat_9 = smolyak_interpolator.interpolate(self.test_grid)
        y_hat_individual.append(y_hat_9)

        smolyak_interpolator.fit(f_10)
        y_hat_10 = smolyak_interpolator.interpolate(self.test_grid)
        y_hat_individual.append(y_hat_10)

        smolyak_interpolator.fit(f_11)
        y_hat_11 = smolyak_interpolator.interpolate(self.test_grid)
        y_hat_individual.append(y_hat_11)

        smolyak_interpolator.fit(f_12)
        y_hat_12 = smolyak_interpolator.interpolate(self.test_grid)
        y_hat_individual.append(y_hat_12)

        smolyak_interpolator.fit(f_hat_collected)
        y_hat_collected = smolyak_interpolator.interpolate(self.test_grid)

        for i in range(12):
            self.assertTrue(np.isclose(y_hat_individual[i], y_hat_collected[i]).all(), f"Not close for index {i}")


if __name__ == '__main__':
    unittest.main()
