import unittest
import numpy as np


from genz.genz_functions import get_genz_function,GenzFunctionType
from smolyak.smolyak import SmolyakInterpolation

class Smolyak(unittest.TestCase):


    def test_smolyak_implementation(self):
        degree = np.int8(3)
        dimension = np.int8(5)
        n_test_samples = np.int32(1000)
        scale = 3

        np.random.seed(42)

        c = np.random.uniform(low=0, high=1, size=(dimension))
        w = np.random.uniform(low=0, high=1, size=(dimension))
        f = get_genz_function(GenzFunctionType.CONTINUOUS, c=c, w=w, d=dimension)

        test_grid = np.random.uniform(low=0, high=1, size=(n_test_samples, dimension))

        y_true = f(test_grid)

        ### Smolyak ###

        dim = np.int8(5)
        scale = np.int8(3)
        lower_bound = np.float16(-1.0)
        upper_bound = np.float16(1.0)

        sy = SmolyakInterpolation(dim, scale, lower_bound, upper_bound)

        f_hat_smolyak = sy.approximate(f)

        y_hat_smolyak = f_hat_smolyak(test_grid)


        ### End Smolyak ###

        # print(f'Smolyak grid: number of points {sg.grid.shape[0]}')

        mean_ad = np.mean(np.abs(y_true - y_hat_smolyak))
        max_ad = np.max(np.abs(y_true - y_hat_smolyak))
        min_ad = np.min(np.abs(y_true - y_hat_smolyak))

        msg = "The mean abs diff is {}\nThe max abs diff is {}\n"
        msg += "The min abs diff is {}"

        print(msg.format(mean_ad, max_ad, min_ad))



if __name__ == '__main__':
    unittest.main()
