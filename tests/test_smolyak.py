import unittest

import numpy as np

from genz.genz_functions import get_genz_function, GenzFunctionType
from grid.grid_provider import GridProvider
from grid.grid_type import GridType
from interpolate.smolyak import SmolyakInterpolator


class Smolyak(unittest.TestCase):

    @staticmethod
    def test_smolyak_implementation_oscillatory():
        np.random.seed(42)

        dim = 5
        scale = 3
        n_test_samples = 1000

        c = np.random.uniform(low=0, high=1, size=dim)
        w = np.random.uniform(low=0, high=1, size=dim)
        f = get_genz_function(GenzFunctionType.CONTINUOUS, c=c, w=w, d=dim)

        test_grid = np.random.uniform(low=0, high=1, size=(n_test_samples, dim))

        y_true = f(test_grid)

        gp = GridProvider(dimension=dim, upper_bound=1.0, lower_bound=0.0)
        grid = gp.generate(GridType.CHEBYSHEV, scale=scale)

        sy = SmolyakInterpolator(grid)

        f_hat_smolyak = sy.interpolate(f)

        y_hat_smolyak = f_hat_smolyak(test_grid)

        mean_ad = np.mean(np.abs(y_true - y_hat_smolyak))
        max_ad = np.max(np.abs(y_true - y_hat_smolyak))
        min_ad = np.min(np.abs(y_true - y_hat_smolyak))

        msg = "The mean abs diff is {}\nThe max abs diff is {}\n"
        msg += "The min abs diff is {}"

        print(msg.format(mean_ad, max_ad, min_ad))

        print('\n')

    @staticmethod
    def test_smolyak_implementation_product_peak():
        np.random.seed(42)

        dim = 5
        scale = 3
        n_test_samples = 1000

        c = np.random.uniform(low=0, high=1, size=dim)
        w = np.random.uniform(low=0, high=1, size=dim)
        f = get_genz_function(GenzFunctionType.PRODUCT_PEAK, c=c, w=w, d=dim)

        test_grid = np.random.uniform(low=0, high=1, size=(n_test_samples, dim))

        y_true = f(test_grid)

        gp = GridProvider(dimension=dim, upper_bound=1.0, lower_bound=0.0)
        grid = gp.generate(GridType.CHEBYSHEV, scale=scale)

        sy = SmolyakInterpolator(grid)

        f_hat_smolyak = sy.interpolate(f)

        y_hat_smolyak = f_hat_smolyak(test_grid)

        mean_ad = np.mean(np.abs(y_true - y_hat_smolyak))
        max_ad = np.max(np.abs(y_true - y_hat_smolyak))
        min_ad = np.min(np.abs(y_true - y_hat_smolyak))

        msg = "The mean abs diff is {}\nThe max abs diff is {}\n"
        msg += "The min abs diff is {}"

        print(msg.format(mean_ad, max_ad, min_ad))

        print('\n')

    @staticmethod
    def test_smolyak_implementation_corner_peak():
        np.random.seed(42)

        dim = 5
        scale = 3
        n_test_samples = 1000

        c = np.random.uniform(low=0, high=1, size=dim)
        w = np.random.uniform(low=0, high=1, size=dim)
        f = get_genz_function(GenzFunctionType.CORNER_PEAK, c=c, w=w, d=dim)

        test_grid = np.random.uniform(low=0, high=1, size=(n_test_samples, dim))

        y_true = f(test_grid)

        gp = GridProvider(dimension=dim, upper_bound=1.0, lower_bound=0.0)
        grid = gp.generate(GridType.CHEBYSHEV, scale=scale)

        sy = SmolyakInterpolator(grid)

        f_hat_smolyak = sy.interpolate(f)

        y_hat_smolyak = f_hat_smolyak(test_grid)

        mean_ad = np.mean(np.abs(y_true - y_hat_smolyak))
        max_ad = np.max(np.abs(y_true - y_hat_smolyak))
        min_ad = np.min(np.abs(y_true - y_hat_smolyak))

        msg = "The mean abs diff is {}\nThe max abs diff is {}\n"
        msg += "The min abs diff is {}"

        print(msg.format(mean_ad, max_ad, min_ad))

        print('\n')

    @staticmethod
    def test_smolyak_implementation_gaussian():
        np.random.seed(42)

        dim = int(5)
        scale = int(3)
        n_test_samples = 1000

        c = np.random.uniform(low=0, high=1, size=dim)
        w = np.random.uniform(low=0, high=1, size=dim)
        f = get_genz_function(GenzFunctionType.GAUSSIAN, c=c, w=w, d=dim)

        test_grid = np.random.uniform(low=0, high=1, size=(n_test_samples, dim))

        y_true = f(test_grid)

        gp = GridProvider(dimension=dim, upper_bound=1.0, lower_bound=0.0)
        grid = gp.generate(GridType.CHEBYSHEV, scale=scale)

        sy = SmolyakInterpolator(grid)

        f_hat_smolyak = sy.interpolate(f)

        y_hat_smolyak = f_hat_smolyak(test_grid)

        mean_ad = np.mean(np.abs(y_true - y_hat_smolyak))
        max_ad = np.max(np.abs(y_true - y_hat_smolyak))
        min_ad = np.min(np.abs(y_true - y_hat_smolyak))

        msg = "The mean abs diff is {}\nThe max abs diff is {}\n"
        msg += "The min abs diff is {}"

        print(msg.format(mean_ad, max_ad, min_ad))

        print('\n')

    @staticmethod
    def test_smolyak_implementation_continuous():
        np.random.seed(42)

        dim = int(5)
        scale = int(3)
        n_test_samples = 1000

        c = np.random.uniform(low=0, high=1, size=dim)
        w = np.random.uniform(low=0, high=1, size=dim)
        f = get_genz_function(GenzFunctionType.CONTINUOUS, c=c, w=w, d=dim)

        test_grid = np.random.uniform(low=0, high=1, size=(n_test_samples, dim))

        y_true = f(test_grid)

        gp = GridProvider(dimension=dim, upper_bound=1.0, lower_bound=0.0)
        grid = gp.generate(GridType.CHEBYSHEV, scale=scale)

        sy = SmolyakInterpolator(grid)

        f_hat_smolyak = sy.interpolate(f)

        y_hat_smolyak = f_hat_smolyak(test_grid)

        mean_ad = np.mean(np.abs(y_true - y_hat_smolyak))
        max_ad = np.max(np.abs(y_true - y_hat_smolyak))
        min_ad = np.min(np.abs(y_true - y_hat_smolyak))

        msg = "The mean abs diff is {}\nThe max abs diff is {}\n"
        msg += "The min abs diff is {}"

        print(msg.format(mean_ad, max_ad, min_ad))

        print('\n')

    @staticmethod
    def test_smolyak_implementation_discontinuous():
        np.random.seed(42)

        dim = int(5)
        scale = int(3)
        n_test_samples = 1000

        c = np.random.uniform(low=0, high=1, size=dim)
        w = np.random.uniform(low=0, high=1, size=dim)
        f = get_genz_function(GenzFunctionType.DISCONTINUOUS, c=c, w=w, d=dim)

        test_grid = np.random.uniform(low=0, high=1, size=(n_test_samples, dim))

        y_true = f(test_grid)

        gp = GridProvider(dimension=dim, upper_bound=1.0, lower_bound=0.0)
        grid = gp.generate(GridType.CHEBYSHEV, scale=scale)

        sy = SmolyakInterpolator(grid)

        f_hat_smolyak = sy.interpolate(f)

        y_hat_smolyak = f_hat_smolyak(test_grid)

        mean_ad = np.mean(np.abs(y_true - y_hat_smolyak))
        max_ad = np.max(np.abs(y_true - y_hat_smolyak))
        min_ad = np.min(np.abs(y_true - y_hat_smolyak))

        msg = "The mean abs diff is {}\nThe max abs diff is {}\n"
        msg += "The min abs diff is {}"

        print(msg.format(mean_ad, max_ad, min_ad))

        print('\n')


if __name__ == '__main__':
    unittest.main()
