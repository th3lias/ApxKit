import unittest
import numpy as np


from genz.genz_functions import get_genz_function,GenzFunctionType
from smolyak.smolyak import SmolyakInterpolation

class Smolyak(unittest.TestCase):


    def test_smolyak_implementation_oscillatory(self):

        np.random.seed(42)

        dim = np.int8(5)
        scale = np.int8(3)
        lower_bound = np.float16(-1.0)
        upper_bound = np.float16(1.0)
        n_test_samples = 1000

        c = np.random.uniform(low=0, high=1, size=(dim))
        w = np.random.uniform(low=0, high=1, size=(dim))
        f = get_genz_function(GenzFunctionType.CONTINUOUS, c=c, w=w, d=dim)

        test_grid = np.random.uniform(low=0, high=1, size=(n_test_samples, dim))

        y_true = f(test_grid)

        sy = SmolyakInterpolation(dim, scale, lower_bound, upper_bound)

        f_hat_smolyak = sy.approximate(f)

        y_hat_smolyak = f_hat_smolyak(test_grid)

        mean_ad = np.mean(np.abs(y_true - y_hat_smolyak))
        max_ad = np.max(np.abs(y_true - y_hat_smolyak))
        min_ad = np.min(np.abs(y_true - y_hat_smolyak))

        msg = "The mean abs diff is {}\nThe max abs diff is {}\n"
        msg += "The min abs diff is {}"

        print(msg.format(mean_ad, max_ad, min_ad))

        print('\n')

    def test_smolyak_implementation_product_peak(self):

        np.random.seed(42)

        dim = np.int8(5)
        scale = np.int8(3)
        lower_bound = np.float16(-1.0)
        upper_bound = np.float16(1.0)
        n_test_samples = 1000

        c = np.random.uniform(low=0, high=1, size=(dim))
        w = np.random.uniform(low=0, high=1, size=(dim))
        f = get_genz_function(GenzFunctionType.PRODUCT_PEAK, c=c, w=w, d=dim)

        test_grid = np.random.uniform(low=0, high=1, size=(n_test_samples, dim))

        y_true = f(test_grid)

        sy = SmolyakInterpolation(dim, scale, lower_bound, upper_bound)

        f_hat_smolyak = sy.approximate(f)

        y_hat_smolyak = f_hat_smolyak(test_grid)

        mean_ad = np.mean(np.abs(y_true - y_hat_smolyak))
        max_ad = np.max(np.abs(y_true - y_hat_smolyak))
        min_ad = np.min(np.abs(y_true - y_hat_smolyak))

        msg = "The mean abs diff is {}\nThe max abs diff is {}\n"
        msg += "The min abs diff is {}"

        print(msg.format(mean_ad, max_ad, min_ad))

        print('\n')

    def test_smolyak_implementation_corner_peak(self):

        np.random.seed(42)

        dim = np.int8(5)
        scale = np.int8(3)
        lower_bound = np.float16(-1.0)
        upper_bound = np.float16(1.0)
        n_test_samples = 1000

        c = np.random.uniform(low=0, high=1, size=(dim))
        w = np.random.uniform(low=0, high=1, size=(dim))
        f = get_genz_function(GenzFunctionType.CORNER_PEAK, c=c, w=w, d=dim)

        test_grid = np.random.uniform(low=0, high=1, size=(n_test_samples, dim))

        y_true = f(test_grid)

        sy = SmolyakInterpolation(dim, scale, lower_bound, upper_bound)

        f_hat_smolyak = sy.approximate(f)

        y_hat_smolyak = f_hat_smolyak(test_grid)

        mean_ad = np.mean(np.abs(y_true - y_hat_smolyak))
        max_ad = np.max(np.abs(y_true - y_hat_smolyak))
        min_ad = np.min(np.abs(y_true - y_hat_smolyak))

        msg = "The mean abs diff is {}\nThe max abs diff is {}\n"
        msg += "The min abs diff is {}"

        print(msg.format(mean_ad, max_ad, min_ad))

        print('\n')

    def test_smolyak_implementation_gaussian(self):

        np.random.seed(42)

        dim = np.int8(5)
        scale = np.int8(3)
        lower_bound = np.float16(-1.0)
        upper_bound = np.float16(1.0)
        n_test_samples = 1000

        c = np.random.uniform(low=0, high=1, size=(dim))
        w = np.random.uniform(low=0, high=1, size=(dim))
        f = get_genz_function(GenzFunctionType.GAUSSIAN, c=c, w=w, d=dim)

        test_grid = np.random.uniform(low=0, high=1, size=(n_test_samples, dim))

        y_true = f(test_grid)

        sy = SmolyakInterpolation(dim, scale, lower_bound, upper_bound)

        f_hat_smolyak = sy.approximate(f)

        y_hat_smolyak = f_hat_smolyak(test_grid)

        mean_ad = np.mean(np.abs(y_true - y_hat_smolyak))
        max_ad = np.max(np.abs(y_true - y_hat_smolyak))
        min_ad = np.min(np.abs(y_true - y_hat_smolyak))

        msg = "The mean abs diff is {}\nThe max abs diff is {}\n"
        msg += "The min abs diff is {}"

        print(msg.format(mean_ad, max_ad, min_ad))

        print('\n')

    def test_smolyak_implementation_continuous(self):

        np.random.seed(42)

        dim = np.int8(5)
        scale = np.int8(3)
        lower_bound = np.float16(-1.0)
        upper_bound = np.float16(1.0)
        n_test_samples = 1000

        c = np.random.uniform(low=0, high=1, size=(dim))
        w = np.random.uniform(low=0, high=1, size=(dim))
        f = get_genz_function(GenzFunctionType.CONTINUOUS, c=c, w=w, d=dim)

        test_grid = np.random.uniform(low=0, high=1, size=(n_test_samples, dim))

        y_true = f(test_grid)

        sy = SmolyakInterpolation(dim, scale, lower_bound, upper_bound)

        f_hat_smolyak = sy.approximate(f)

        y_hat_smolyak = f_hat_smolyak(test_grid)

        mean_ad = np.mean(np.abs(y_true - y_hat_smolyak))
        max_ad = np.max(np.abs(y_true - y_hat_smolyak))
        min_ad = np.min(np.abs(y_true - y_hat_smolyak))

        msg = "The mean abs diff is {}\nThe max abs diff is {}\n"
        msg += "The min abs diff is {}"

        print(msg.format(mean_ad, max_ad, min_ad))

        print('\n')

    def test_smolyak_implementation_discontinuous(self):

        np.random.seed(42)

        dim = np.int8(5)
        scale = np.int8(3)
        lower_bound = np.float16(-1.0)
        upper_bound = np.float16(1.0)
        n_test_samples = 1000

        c = np.random.uniform(low=0, high=1, size=(dim))
        w = np.random.uniform(low=0, high=1, size=(dim))
        f = get_genz_function(GenzFunctionType.DISCONTINUOUS, c=c, w=w, d=dim)

        test_grid = np.random.uniform(low=0, high=1, size=(n_test_samples, dim))

        y_true = f(test_grid)

        sy = SmolyakInterpolation(dim, scale, lower_bound, upper_bound)

        f_hat_smolyak = sy.approximate(f)

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
