import unittest
import numpy as np

from genz.genz_functions import get_genz_function, GenzFunctionType

from least_squares.least_squares import approximate_by_polynomial_with_least_squares_iterative
from least_squares.least_squares import approximate_by_polynomial_with_least_squares


class LeastSquaresTests(unittest.TestCase):

    def test_parallel_oscillatory(self):
        degree = np.int8(3)
        dimension = np.int8(20)
        n_samples = 10000
        n_test_samples = np.int32(100)

        grid = np.random.uniform(low=0, high=1, size=(n_samples, dimension))

        c_1 = np.random.uniform(low=0, high=1, size=(dimension))
        w_1 = np.random.uniform(low=0, high=1, size=(dimension))
        c_2 = np.random.uniform(low=0, high=1, size=(dimension))
        w_2 = np.random.uniform(low=0, high=1, size=(dimension))

        f_1 = get_genz_function(GenzFunctionType.OSCILLATORY, c=c_1, w=w_1, d=dimension)
        f_2 = get_genz_function(GenzFunctionType.OSCILLATORY, c=c_2, w=w_2, d=dimension)

        f = [f_1, f_2]

        test_grid = np.random.uniform(low=0, high=1, size=(n_test_samples, dimension))

        f_hat_1 = approximate_by_polynomial_with_least_squares(grid=grid, f=f_1, degree=degree, include_bias=True,
                                                               dim=dimension,
                                                               self_implemented=True)
        f_hat_2 = approximate_by_polynomial_with_least_squares(grid=grid, f=f_2, degree=degree, include_bias=True,
                                                               dim=dimension,
                                                               self_implemented=True)
        f_hat_both = approximate_by_polynomial_with_least_squares(f=f, dim=dimension, degree=degree, grid=grid,
                                                                  include_bias=True, self_implemented=True)

        y_hat = f_hat_both(test_grid)

        y_hat_1_combined = y_hat[:, 0]
        y_hat_2_combined = y_hat[:, 1]

        y_hat_1 = f_hat_1(test_grid)
        y_hat_2 = f_hat_2(test_grid)

        self.assertAlmostEqual(y_hat_1_combined, y_hat_1)
        self.assertAlmostEqual(y_hat_2_combined, y_hat_2)

    def test_parallel_product_peak(self):
        degree = np.int8(3)
        dimension = np.int8(20)
        n_samples = 10000
        n_test_samples = np.int32(100)

        grid = np.random.uniform(low=0, high=1, size=(n_samples, dimension))

        c_1 = np.random.uniform(low=0, high=1, size=(dimension))
        w_1 = np.random.uniform(low=0, high=1, size=(dimension))
        c_2 = np.random.uniform(low=0, high=1, size=(dimension))
        w_2 = np.random.uniform(low=0, high=1, size=(dimension))

        f_1 = get_genz_function(GenzFunctionType.PRODUCT_PEAK, c=c_1, w=w_1, d=dimension)
        f_2 = get_genz_function(GenzFunctionType.PRODUCT_PEAK, c=c_2, w=w_2, d=dimension)

        f = [f_1, f_2]

        test_grid = np.random.uniform(low=0, high=1, size=(n_test_samples, dimension))

        f_hat_1 = approximate_by_polynomial_with_least_squares(grid=grid, f=f_1, degree=degree, include_bias=True,
                                                               dim=dimension,
                                                               self_implemented=True)
        f_hat_2 = approximate_by_polynomial_with_least_squares(grid=grid, f=f_2, degree=degree, include_bias=True,
                                                               dim=dimension,
                                                               self_implemented=True)
        f_hat_both = approximate_by_polynomial_with_least_squares(f=f, dim=dimension, degree=degree, grid=grid,
                                                                  include_bias=True, self_implemented=True)

        y_hat = f_hat_both(test_grid)

        y_hat_1_combined = y_hat[:, 0]
        y_hat_2_combined = y_hat[:, 1]

        y_hat_1 = f_hat_1(test_grid)
        y_hat_2 = f_hat_2(test_grid)

        self.assertAlmostEqual(y_hat_1_combined, y_hat_1)
        self.assertAlmostEqual(y_hat_2_combined, y_hat_2)

    def test_parallel_corner_peak(self):
        degree = np.int8(3)
        dimension = np.int8(20)
        n_samples = 10000
        n_test_samples = np.int32(100)

        grid = np.random.uniform(low=0, high=1, size=(n_samples, dimension))

        c_1 = np.random.uniform(low=0, high=1, size=(dimension))
        w_1 = np.random.uniform(low=0, high=1, size=(dimension))
        c_2 = np.random.uniform(low=0, high=1, size=(dimension))
        w_2 = np.random.uniform(low=0, high=1, size=(dimension))

        f_1 = get_genz_function(GenzFunctionType.CORNER_PEAK, c=c_1, w=w_1, d=dimension)
        f_2 = get_genz_function(GenzFunctionType.CORNER_PEAK, c=c_2, w=w_2, d=dimension)

        f = [f_1, f_2]

        test_grid = np.random.uniform(low=0, high=1, size=(n_test_samples, dimension))

        f_hat_1 = approximate_by_polynomial_with_least_squares(grid=grid, f=f_1, degree=degree, include_bias=True,
                                                               dim=dimension,
                                                               self_implemented=True)
        f_hat_2 = approximate_by_polynomial_with_least_squares(grid=grid, f=f_2, degree=degree, include_bias=True,
                                                               dim=dimension,
                                                               self_implemented=True)
        f_hat_both = approximate_by_polynomial_with_least_squares(f=f, dim=dimension, degree=degree, grid=grid,
                                                                  include_bias=True, self_implemented=True)

        y_hat = f_hat_both(test_grid)

        y_hat_1_combined = y_hat[:, 0]
        y_hat_2_combined = y_hat[:, 1]

        y_hat_1 = f_hat_1(test_grid)
        y_hat_2 = f_hat_2(test_grid)

        self.assertAlmostEqual(y_hat_1_combined, y_hat_1)
        self.assertAlmostEqual(y_hat_2_combined, y_hat_2)

    def test_parallel_gaussian(self):
        degree = np.int8(3)
        dimension = np.int8(20)
        n_samples = 10000
        n_test_samples = np.int32(100)

        grid = np.random.uniform(low=0, high=1, size=(n_samples, dimension))

        c_1 = np.random.uniform(low=0, high=1, size=(dimension))
        w_1 = np.random.uniform(low=0, high=1, size=(dimension))
        c_2 = np.random.uniform(low=0, high=1, size=(dimension))
        w_2 = np.random.uniform(low=0, high=1, size=(dimension))

        f_1 = get_genz_function(GenzFunctionType.GAUSSIAN, c=c_1, w=w_1, d=dimension)
        f_2 = get_genz_function(GenzFunctionType.GAUSSIAN, c=c_2, w=w_2, d=dimension)

        f = [f_1, f_2]

        test_grid = np.random.uniform(low=0, high=1, size=(n_test_samples, dimension))

        f_hat_1 = approximate_by_polynomial_with_least_squares(grid=grid, f=f_1, degree=degree, include_bias=True,
                                                               dim=dimension,
                                                               self_implemented=True)
        f_hat_2 = approximate_by_polynomial_with_least_squares(grid=grid, f=f_2, degree=degree, include_bias=True,
                                                               dim=dimension,
                                                               self_implemented=True)
        f_hat_both = approximate_by_polynomial_with_least_squares(f=f, dim=dimension, degree=degree, grid=grid,
                                                                  include_bias=True, self_implemented=True)

        y_hat = f_hat_both(test_grid)

        y_hat_1_combined = y_hat[:, 0]
        y_hat_2_combined = y_hat[:, 1]

        y_hat_1 = f_hat_1(test_grid)
        y_hat_2 = f_hat_2(test_grid)

        self.assertAlmostEqual(y_hat_1_combined, y_hat_1)
        self.assertAlmostEqual(y_hat_2_combined, y_hat_2)

    def test_parallel_continuous(self):
        degree = np.int8(3)
        dimension = np.int8(20)
        n_samples = 10000
        n_test_samples = np.int32(100)

        grid = np.random.uniform(low=0, high=1, size=(n_samples, dimension))

        c_1 = np.random.uniform(low=0, high=1, size=(dimension))
        w_1 = np.random.uniform(low=0, high=1, size=(dimension))
        c_2 = np.random.uniform(low=0, high=1, size=(dimension))
        w_2 = np.random.uniform(low=0, high=1, size=(dimension))

        f_1 = get_genz_function(GenzFunctionType.CONTINUOUS, c=c_1, w=w_1, d=dimension)
        f_2 = get_genz_function(GenzFunctionType.CONTINUOUS, c=c_2, w=w_2, d=dimension)

        f = [f_1, f_2]

        test_grid = np.random.uniform(low=0, high=1, size=(n_test_samples, dimension))

        f_hat_1 = approximate_by_polynomial_with_least_squares(grid=grid, f=f_1, degree=degree, include_bias=True,
                                                               dim=dimension,
                                                               self_implemented=True)
        f_hat_2 = approximate_by_polynomial_with_least_squares(grid=grid, f=f_2, degree=degree, include_bias=True,
                                                               dim=dimension,
                                                               self_implemented=True)
        f_hat_both = approximate_by_polynomial_with_least_squares(f=f, dim=dimension, degree=degree, grid=grid,
                                                                  include_bias=True, self_implemented=True)

        y_hat = f_hat_both(test_grid)

        y_hat_1_combined = y_hat[:, 0]
        y_hat_2_combined = y_hat[:, 1]

        y_hat_1 = f_hat_1(test_grid)
        y_hat_2 = f_hat_2(test_grid)

        self.assertAlmostEqual(y_hat_1_combined, y_hat_1)
        self.assertAlmostEqual(y_hat_2_combined, y_hat_2)

    def test_parallel_discontinous(self):
        degree = np.int8(3)
        dimension = np.int8(20)
        n_samples = 10000
        n_test_samples = np.int32(100)

        grid = np.random.uniform(low=0, high=1, size=(n_samples, dimension))

        c_1 = np.random.uniform(low=0, high=1, size=(dimension))
        w_1 = np.random.uniform(low=0, high=1, size=(dimension))
        c_2 = np.random.uniform(low=0, high=1, size=(dimension))
        w_2 = np.random.uniform(low=0, high=1, size=(dimension))

        f_1 = get_genz_function(GenzFunctionType.DISCONTINUOUS, c=c_1, w=w_1, d=dimension)
        f_2 = get_genz_function(GenzFunctionType.DISCONTINUOUS, c=c_2, w=w_2, d=dimension)

        f = [f_1, f_2]

        test_grid = np.random.uniform(low=0, high=1, size=(n_test_samples, dimension))

        f_hat_1 = approximate_by_polynomial_with_least_squares(grid=grid, f=f_1, degree=degree, include_bias=True,
                                                               dim=dimension,
                                                               self_implemented=True)
        f_hat_2 = approximate_by_polynomial_with_least_squares(grid=grid, f=f_2, degree=degree, include_bias=True,
                                                               dim=dimension,
                                                               self_implemented=True)
        f_hat_both = approximate_by_polynomial_with_least_squares(f=f, dim=dimension, degree=degree, grid=grid,
                                                                  include_bias=True, self_implemented=True)

        y_hat = f_hat_both(test_grid)

        y_hat_1_combined = y_hat[:, 0]
        y_hat_2_combined = y_hat[:, 1]

        y_hat_1 = f_hat_1(test_grid)
        y_hat_2 = f_hat_2(test_grid)

        self.assertAlmostEqual(y_hat_1_combined, y_hat_1)
        self.assertAlmostEqual(y_hat_2_combined, y_hat_2)

    def test_self_implemented_oscillatory(self):
        degree = np.int8(3)
        dimension = np.int8(20)
        n_samples = 10000
        n_test_samples = np.int32(100)

        grid = np.random.uniform(low=0, high=1, size=(n_samples, dimension))

        c = np.random.uniform(low=0, high=1, size=(dimension))
        w = np.random.uniform(low=0, high=1, size=(dimension))

        f = get_genz_function(GenzFunctionType.OSCILLATORY, c=c, w=w, d=dimension)

        test_grid = np.random.uniform(low=0, high=1, size=(n_test_samples, dimension))

        f_hat_self = approximate_by_polynomial_with_least_squares(grid=grid, f=f, degree=degree, include_bias=True,
                                                                  dim=dimension, self_implemented=True)

        f_hat_sklearn = approximate_by_polynomial_with_least_squares(grid=grid, f=f, degree=degree, include_bias=True,
                                                                     dim=dimension, self_implemented=False)

        f_hat_iterative = approximate_by_polynomial_with_least_squares_iterative(grid=grid, f=f, degree=degree,
                                                                                 include_bias=True,
                                                                                 dim=dimension, self_implemented=False)

        y_hat_self = f_hat_self(test_grid)
        y_hat_sklearn = f_hat_sklearn(test_grid)
        y_hat_iterative = f_hat_iterative(test_grid)

        self.assertAlmostEqual(y_hat_self, y_hat_sklearn)
        self.assertAlmostEqual(y_hat_iterative, y_hat_sklearn)

    def test_self_implemented_product_peak(self):
        degree = np.int8(3)
        dimension = np.int8(20)
        n_samples = 10000
        n_test_samples = np.int32(100)

        grid = np.random.uniform(low=0, high=1, size=(n_samples, dimension))

        c = np.random.uniform(low=0, high=1, size=(dimension))
        w = np.random.uniform(low=0, high=1, size=(dimension))

        f = get_genz_function(GenzFunctionType.PRODUCT_PEAK, c=c, w=w, d=dimension)

        test_grid = np.random.uniform(low=0, high=1, size=(n_test_samples, dimension))

        f_hat_self = approximate_by_polynomial_with_least_squares(grid=grid, f=f, degree=degree, include_bias=True,
                                                                  dim=dimension, self_implemented=True)

        f_hat_sklearn = approximate_by_polynomial_with_least_squares(grid=grid, f=f, degree=degree, include_bias=True,
                                                                     dim=dimension, self_implemented=False)

        f_hat_iterative = approximate_by_polynomial_with_least_squares_iterative(grid=grid, f=f, degree=degree,
                                                                                 include_bias=True,
                                                                                 dim=dimension, self_implemented=False)

        y_hat_self = f_hat_self(test_grid)
        y_hat_sklearn = f_hat_sklearn(test_grid)
        y_hat_iterative = f_hat_iterative(test_grid)

        self.assertAlmostEqual(y_hat_self, y_hat_sklearn)
        self.assertAlmostEqual(y_hat_iterative, y_hat_sklearn)

    def test_self_implemented_corner_peak(self):
        degree = np.int8(3)
        dimension = np.int8(20)
        n_samples = 10000
        n_test_samples = np.int32(100)

        grid = np.random.uniform(low=0, high=1, size=(n_samples, dimension))

        c = np.random.uniform(low=0, high=1, size=(dimension))
        w = np.random.uniform(low=0, high=1, size=(dimension))

        f = get_genz_function(GenzFunctionType.CORNER_PEAK, c=c, w=w, d=dimension)

        test_grid = np.random.uniform(low=0, high=1, size=(n_test_samples, dimension))

        f_hat_self = approximate_by_polynomial_with_least_squares(grid=grid, f=f, degree=degree, include_bias=True,
                                                                  dim=dimension, self_implemented=True)

        f_hat_sklearn = approximate_by_polynomial_with_least_squares(grid=grid, f=f, degree=degree, include_bias=True,
                                                                     dim=dimension, self_implemented=False)

        f_hat_iterative = approximate_by_polynomial_with_least_squares_iterative(grid=grid, f=f, degree=degree,
                                                                                 include_bias=True,
                                                                                 dim=dimension, self_implemented=False)

        y_hat_self = f_hat_self(test_grid)
        y_hat_sklearn = f_hat_sklearn(test_grid)
        y_hat_iterative = f_hat_iterative(test_grid)

        self.assertAlmostEqual(y_hat_self, y_hat_sklearn)
        self.assertAlmostEqual(y_hat_iterative, y_hat_sklearn)

    def test_self_implemented_gaussian(self):
        degree = np.int8(3)
        dimension = np.int8(20)
        n_samples = 10000
        n_test_samples = np.int32(100)

        grid = np.random.uniform(low=0, high=1, size=(n_samples, dimension))

        c = np.random.uniform(low=0, high=1, size=(dimension))
        w = np.random.uniform(low=0, high=1, size=(dimension))

        f = get_genz_function(GenzFunctionType.GAUSSIAN, c=c, w=w, d=dimension)

        test_grid = np.random.uniform(low=0, high=1, size=(n_test_samples, dimension))

        f_hat_self = approximate_by_polynomial_with_least_squares(grid=grid, f=f, degree=degree, include_bias=True,
                                                                  dim=dimension, self_implemented=True)

        f_hat_sklearn = approximate_by_polynomial_with_least_squares(grid=grid, f=f, degree=degree, include_bias=True,
                                                                     dim=dimension, self_implemented=False)

        f_hat_iterative = approximate_by_polynomial_with_least_squares_iterative(grid=grid, f=f, degree=degree,
                                                                                 include_bias=True,
                                                                                 dim=dimension, self_implemented=False)

        y_hat_self = f_hat_self(test_grid)
        y_hat_sklearn = f_hat_sklearn(test_grid)
        y_hat_iterative = f_hat_iterative(test_grid)

        self.assertAlmostEqual(y_hat_self, y_hat_sklearn)
        self.assertAlmostEqual(y_hat_iterative, y_hat_sklearn)

    def test_self_implemented_continuous(self):
        degree = np.int8(3)
        dimension = np.int8(20)
        n_samples = 10000
        n_test_samples = np.int32(100)

        grid = np.random.uniform(low=0, high=1, size=(n_samples, dimension))

        c = np.random.uniform(low=0, high=1, size=(dimension))
        w = np.random.uniform(low=0, high=1, size=(dimension))

        f = get_genz_function(GenzFunctionType.CONTINUOUS, c=c, w=w, d=dimension)

        test_grid = np.random.uniform(low=0, high=1, size=(n_test_samples, dimension))

        f_hat_self = approximate_by_polynomial_with_least_squares(grid=grid, f=f, degree=degree, include_bias=True,
                                                                  dim=dimension, self_implemented=True)

        f_hat_sklearn = approximate_by_polynomial_with_least_squares(grid=grid, f=f, degree=degree, include_bias=True,
                                                                     dim=dimension, self_implemented=False)

        f_hat_iterative = approximate_by_polynomial_with_least_squares_iterative(grid=grid, f=f, degree=degree,
                                                                                 include_bias=True,
                                                                                 dim=dimension, self_implemented=False)

        y_hat_self = f_hat_self(test_grid)
        y_hat_sklearn = f_hat_sklearn(test_grid)
        y_hat_iterative = f_hat_iterative(test_grid)

        self.assertAlmostEqual(y_hat_self, y_hat_sklearn)
        self.assertAlmostEqual(y_hat_iterative, y_hat_sklearn)

    def test_self_implemented_discontinuous(self):
        degree = np.int8(3)
        dimension = np.int8(20)
        n_samples = 10000
        n_test_samples = np.int32(100)

        grid = np.random.uniform(low=0, high=1, size=(n_samples, dimension))

        c = np.random.uniform(low=0, high=1, size=(dimension))
        w = np.random.uniform(low=0, high=1, size=(dimension))

        f = get_genz_function(GenzFunctionType.DISCONTINUOUS, c=c, w=w, d=dimension)

        test_grid = np.random.uniform(low=0, high=1, size=(n_test_samples, dimension))

        f_hat_self = approximate_by_polynomial_with_least_squares(grid=grid, f=f, degree=degree, include_bias=True,
                                                                  dim=dimension, self_implemented=True)

        f_hat_sklearn = approximate_by_polynomial_with_least_squares(grid=grid, f=f, degree=degree, include_bias=True,
                                                                     dim=dimension, self_implemented=False)

        f_hat_iterative = approximate_by_polynomial_with_least_squares_iterative(grid=grid, f=f, degree=degree,
                                                                                 include_bias=True,
                                                                                 dim=dimension, self_implemented=False)

        y_hat_self = f_hat_self(test_grid)
        y_hat_sklearn = f_hat_sklearn(test_grid)
        y_hat_iterative = f_hat_iterative(test_grid)

        self.assertAlmostEqual(y_hat_self, y_hat_sklearn)
        self.assertAlmostEqual(y_hat_iterative, y_hat_sklearn)


if __name__ == '__main__':
    unittest.main()
