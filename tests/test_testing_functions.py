import unittest

import numpy as np

from function.type import FunctionType
from function.provider import ParametrizedFunctionProvider


class FunctionTests(unittest.TestCase):

    @staticmethod
    def get_1d_test_points():
        test_points = np.array([0, 1, 0.35, -1 / 6, -0.25, 1])

        return test_points

    @staticmethod
    def get_1d_test_points_2():
        return np.array([1, 0, -2, 1.5, np.sqrt(2)])

    @staticmethod
    def get_2d_test_points():
        test_points = np.array([[0, 0], [4, 10], [1, 1], [-1, 1], [2, 0]])

        return test_points

    @staticmethod
    def get_4d_test_points():
        sample_1 = [1.5, 2.3, -1.4, 0]
        sample_2 = [-1.5, 1, -0.11, 2.123]
        sample_3 = [np.sqrt(2), np.exp(1), -np.sin(2), np.sqrt(3)]

        return np.array([sample_1, sample_2, sample_3])

    def test_oscillatory_1d_1(self):
        d = 1

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.OSCILLATORY, d, np.array([1]), np.array([0]))
        f = lambda t: np.cos(t).squeeze()

        data_1 = self.get_1d_test_points()
        data_2 = self.get_1d_test_points_2()
        y_hat_1 = f_hat(data_1)
        y_1 = f(data_1)
        y_hat_2 = f_hat(data_2)
        y_2 = f(data_2)

        self.assertTrue(np.isclose(y_hat_1, y_1).all())
        self.assertTrue(np.isclose(y_hat_2, y_2).all())

    def test_oscillatory_1d_2(self):
        d = 1

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.OSCILLATORY, d, np.array([0.5]), np.array([-0.25]))
        f = lambda t: np.sin(0.5 * t).squeeze()

        data_1 = self.get_1d_test_points()
        data_2 = self.get_1d_test_points_2()
        y_hat_1 = f_hat(data_1)
        y_1 = f(data_1)
        y_hat_2 = f_hat(data_2)
        y_2 = f(data_2)

        self.assertTrue(np.isclose(y_hat_1, y_1).all())
        self.assertTrue(np.isclose(y_hat_2, y_2).all())

    def test_oscillatory_2d(self):
        d = 2

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.OSCILLATORY, d, np.array([-1, 2]), np.array([-2.1, 0]))
        f = lambda t: np.cos(-2 * np.pi * 2.1 - t[:, 0] + 2 * t[:, 1])

        data = self.get_2d_test_points()
        y_hat = f_hat(data)
        y = f(data)

        self.assertTrue(np.isclose(y_hat, y).all())

    def test_oscillatory_4d(self):
        d = 4

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.OSCILLATORY, d, np.array([-12, -12, 1, 2.3]), np.array([1.2, 0, 1, 4]))
        f = lambda t: np.cos(- 12 * t[:, 0] - 12 * t[:, 1] + t[:, 2] + 2.3 * t[:, 3] + 2 * np.pi * 1.2)

        data = self.get_4d_test_points()
        y_hat = f_hat(data)
        y = f(data)

        self.assertTrue(np.isclose(y_hat, y).all())

    def test_product_peak_1d_1(self):
        d = 1

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.PRODUCT_PEAK, c=np.array([np.sqrt(2)]), w=np.array([3 / 8]), d=d)
        f = lambda t: 1 / (np.power(np.sqrt(2), -2) + np.square(t - 3 / 8)).squeeze()

        data_1 = self.get_1d_test_points()
        data_2 = self.get_1d_test_points_2()
        y_hat_1 = f_hat(data_1)
        y_1 = f(data_1)
        y_hat_2 = f_hat(data_2)
        y_2 = f(data_2)

        self.assertTrue(np.isclose(y_hat_1, y_1).all())
        self.assertTrue(np.isclose(y_hat_2, y_2).all())

    def test_product_peak_1d_2(self):
        d = 1

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.PRODUCT_PEAK, c=np.array([3]), w=np.array([1]), d=d)
        f = lambda t: 1 / (1 / (np.square(3)) + np.square((t - 1))).squeeze()

        data_1 = self.get_1d_test_points()
        data_2 = self.get_1d_test_points_2()
        y_hat_1 = f_hat(data_1)
        y_1 = f(data_1)
        y_hat_2 = f_hat(data_2)
        y_2 = f(data_2)

        self.assertTrue(np.isclose(y_hat_1, y_1).all())
        self.assertTrue(np.isclose(y_hat_2, y_2).all())

    def test_product_peak_2d(self):
        d = 2

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.PRODUCT_PEAK, c=np.array([0.2, 1.5]), w=np.array([0.5, 0.25]), d=d)
        f = lambda t: 1 / (
                (np.power(0.2, -2) + np.square(t[:, 0] - 0.5)) * (np.power(1.5, -2) + np.square(t[:, 1] - 0.25)))

        data = self.get_2d_test_points()
        y_hat = f_hat(data)
        y = f(data)

        self.assertTrue(np.isclose(y_hat, y).all())

    def test_product_peak_4d(self):
        d = 4

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.PRODUCT_PEAK, c=np.array([-0.7, 0.2, 0.1, 1 / 3]),
                                  w=np.array([2.1, -1.0, 0, 0.5]), d=d)
        f = lambda t: 1 / ((100 / 49 + np.square(t[:, 0] - 2.1)) * (25 + np.square(t[:, 1] + 1)) * (
                100 + np.square(t[:, 2])) * (9 + np.square(t[:, 3] - 0.5)))

        data = self.get_4d_test_points()
        y_hat = f_hat(data)
        y = f(data)

        self.assertTrue(np.isclose(y_hat, y).all())

    def test_corner_peak_1d_1(self):
        d = 1

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.CORNER_PEAK, c=np.array([np.exp(1)]), w=None, d=d)
        f = lambda t: 1 / (np.power((1 + np.exp(1) * t), d + 1)).squeeze()

        data_1 = self.get_1d_test_points()
        data_2 = self.get_1d_test_points_2()
        y_hat_1 = f_hat(data_1)
        y_1 = f(data_1)
        y_hat_2 = f_hat(data_2)
        y_2 = f(data_2)

        self.assertTrue(np.isclose(y_hat_1, y_1).all())
        self.assertTrue(np.isclose(y_hat_2, y_2).all())

    def test_corner_peak_1d_2(self):
        d = 1

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.CORNER_PEAK, c=np.array([1]), w=None, d=d)
        f = lambda t: 1 / (np.power((1 + t), d + 1)).squeeze()

        data_1 = self.get_1d_test_points()
        data_2 = self.get_1d_test_points_2()
        y_hat_1 = f_hat(data_1)
        y_1 = f(data_1)
        y_hat_2 = f_hat(data_2)
        y_2 = f(data_2)

        self.assertTrue(np.isclose(y_hat_1, y_1).all())
        self.assertTrue(np.isclose(y_hat_2, y_2).all())

    def test_corner_peak_2d(self):
        d = 2

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.CORNER_PEAK, c=np.array([0.2, 1.5]), w=None, d=d)
        f = lambda t: np.power((1 + 0.2 * t[:, 0] + 1.5 * t[:, 1]), -d - 1)

        data = self.get_2d_test_points()
        y_hat = f_hat(data)
        y = f(data)

        self.assertTrue(np.isclose(y_hat, y).all())

    def test_corner_peak_4d(self):
        d = 4

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.CORNER_PEAK, c=np.array([-0.7, 0.2, 0.1, 1.2]), w=None, d=d)
        f = lambda t: np.power((1 + -0.7 * t[:, 0] + 0.2 * t[:, 1]) + 0.1 * t[:, 2] + 1.2 * t[:, 3], -d - 1)

        data = self.get_4d_test_points()
        y_hat = f_hat(data)
        y = f(data)

        self.assertTrue(np.isclose(y_hat, y).all())

    def test_gaussian_1d_1(self):
        d = 1

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.GAUSSIAN, c=np.array([0.6]), w=np.array([0.35]), d=d)
        f = lambda t: np.exp(-0.36 * np.square(t - 0.35)).squeeze()

        data_1 = self.get_1d_test_points()
        data_2 = self.get_1d_test_points_2()
        y_hat_1 = f_hat(data_1)
        y_1 = f(data_1)
        y_hat_2 = f_hat(data_2)
        y_2 = f(data_2)

        self.assertTrue(np.isclose(y_hat_1, y_1).all())
        self.assertTrue(np.isclose(y_hat_2, y_2).all())

    def test_gaussian_1d_2(self):
        d = 1

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.GAUSSIAN, c=np.array([-0.1]), w=np.array([-0.1]), d=d)
        f = lambda t: np.exp(-0.01 * (t + 0.1) ** 2).squeeze()

        data_1 = self.get_1d_test_points()
        data_2 = self.get_1d_test_points_2()
        y_hat_1 = f_hat(data_1)
        y_1 = f(data_1)
        y_hat_2 = f_hat(data_2)
        y_2 = f(data_2)

        self.assertTrue(np.isclose(y_hat_1, y_1).all())
        self.assertTrue(np.isclose(y_hat_2, y_2).all())

    def test_gaussian_2d(self):
        d = 2

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.GAUSSIAN, c=np.array([0.2, 1.5]), w=np.array([1, 2]), d=d)
        f = lambda t: np.exp(-0.04 * (t[:, 0] - 1) ** 2 - 2.25 * (t[:, 1] - 2) ** 2)

        data = self.get_2d_test_points()
        y_hat = f_hat(data)
        y = f(data)

        self.assertTrue(np.isclose(y_hat, y).all())

    def test_gaussian_4d(self):
        d = 4

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.GAUSSIAN, c=np.array([np.sqrt(2), 1 / 6, 0, 1]),
                                  w=np.array([0, -2, 0.17, -1.42]), d=d)
        f = lambda t: np.exp(-2 * (t[:, 0]) ** 2 - 1 / 36 * (t[:, 1] + 2) ** 2 - (t[:, 3] + 1.42) ** 2)

        data = self.get_4d_test_points()
        y_hat = f_hat(data)
        y = f(data)

        self.assertTrue(np.isclose(y_hat, y).all())

    def test_continuous_1d_1(self):
        d = 1

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.CONTINUOUS, c=np.array([0.6]), w=np.array([0.35]), d=d)
        f = lambda t: np.exp(-0.6 * np.abs(t - 0.35)).squeeze()

        data_1 = self.get_1d_test_points()
        data_2 = self.get_1d_test_points_2()
        y_hat_1 = f_hat(data_1)
        y_1 = f(data_1)
        y_hat_2 = f_hat(data_2)
        y_2 = f(data_2)

        self.assertTrue(np.isclose(y_hat_1, y_1).all())
        self.assertTrue(np.isclose(y_hat_2, y_2).all())

    def test_continuous_1d_2(self):
        d = 1

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.CONTINUOUS, c=np.array([-0.1]), w=np.array([-0.1]), d=d)
        f = lambda t: np.exp(0.1 * np.abs(t + 0.1)).squeeze()

        data_1 = self.get_1d_test_points()
        data_2 = self.get_1d_test_points_2()
        y_hat_1 = f_hat(data_1)
        y_1 = f(data_1)
        y_hat_2 = f_hat(data_2)
        y_2 = f(data_2)

        self.assertTrue(np.isclose(y_hat_1, y_1).all())
        self.assertTrue(np.isclose(y_hat_2, y_2).all())

    def test_continuous_2d(self):
        d = 2

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.CONTINUOUS, c=np.array([0.2, 1.5]), w=np.array([1, 2]), d=d)
        f = lambda t: np.exp(-0.2 * np.abs(t[:, 0] - 1) - 1.5 * np.abs(t[:, 1] - 2))

        data = self.get_2d_test_points()
        y_hat = f_hat(data)
        y = f(data)

        self.assertTrue(np.isclose(y_hat, y).all())

    def test_continuous_4d(self):
        d = 4

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.CONTINUOUS, c=np.array([np.sqrt(2), 1 / 6, 0.237, -3]),
                                  w=np.array([0, -2, 1, -1 / 2]), d=d)
        f = lambda t: np.exp(
            -np.sqrt(2) * np.abs(t[:, 0]) - 1 / 6 * np.abs(t[:, 1] + 2) - 0.237 * np.abs(t[:, 2] - 1) + 3 * np.abs(
                t[:, 3] + 1 / 2))

        data = self.get_4d_test_points()
        y_hat = f_hat(data)
        y = f(data)

        self.assertTrue(np.isclose(y_hat, y).all())

    def test_discontinuous_1d_1(self):
        d = 1

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.DISCONTINUOUS, c=np.array([0.6]), w=np.array([0.35]), d=d)

        def f(x):
            return np.array([0 if i > 0.35 else np.exp(0.6 * i) for i in x])

        data_1 = self.get_1d_test_points()
        data_2 = self.get_1d_test_points_2()
        y_hat_1 = f_hat(data_1)
        y_1 = f(data_1)
        y_hat_2 = f_hat(data_2)
        y_2 = f(data_2)

        self.assertTrue(np.isclose(y_hat_1, y_1).all())
        self.assertTrue(np.isclose(y_hat_2, y_2).all())

    def test_discontinuous_1d_2(self):
        d = 1

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.DISCONTINUOUS, c=np.array([-0.1]), w=np.array([-0.1]), d=d)

        def f(x):
            return np.array([0 if i > -0.1 else np.exp(-0.1 * i) for i in x])

        data_1 = self.get_1d_test_points()
        data_2 = self.get_1d_test_points_2()
        y_hat_1 = f_hat(data_1)
        y_1 = f(data_1)
        y_hat_2 = f_hat(data_2)
        y_2 = f(data_2)

        self.assertTrue(np.isclose(y_hat_1, y_1).all())
        self.assertTrue(np.isclose(y_hat_2, y_2).all())

    def test_discontinuous_2d(self):
        d = 2

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.DISCONTINUOUS, c=np.array([0.2, 1.5]), w=np.array([1, 2]), d=d)

        def f(x):
            return np.array([0 if i[0] > 1 or i[1] > 2 else np.exp(0.2 * i[0] + 1.5 * i[1]) for i in x])

        data = self.get_2d_test_points()
        y_hat = f_hat(data)
        y = f(data)

        self.assertTrue(np.isclose(y_hat, y).all())

    def test_discontinuous_4d(self):
        d = 4

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.DISCONTINUOUS, c=np.array([np.sqrt(2), 1 / 6, 1.1, -0.7]),
                                  w=np.array([0, -2, 0, 0]), d=d)

        def f(x):
            return np.array(
                [0 if i[0] > 0 or i[1] > -2 else np.exp(np.sqrt(2) * i[0] + 1 / 6 * i[1] + 1.1 * i[2] - 0.7 * i[3]) for
                 i in x])

        data = self.get_4d_test_points()
        y_hat = f_hat(data)
        y = f(data)

        self.assertTrue(np.isclose(y_hat, y).all())

    def test_g_function_1d(self):
        d = 1

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.G_FUNCTION, d)
        f = lambda t: (2 * np.abs(4 * t - 2) - 1).squeeze()

        data_1 = self.get_1d_test_points()
        data_2 = self.get_1d_test_points_2()
        y_hat_1 = f_hat(data_1)
        y_1 = f(data_1)
        y_hat_2 = f_hat(data_2)
        y_2 = f(data_2)

        self.assertTrue(np.isclose(y_hat_1, y_1).all())
        self.assertTrue(np.isclose(y_hat_2, y_2).all())

    def test_g_function_2d(self):
        d = 2

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.G_FUNCTION, d)

        def f(x):
            a = (2 * np.abs(4 * x[:, 0] - 2) - 1)
            b = (np.abs(4 * x[:, 1] - 2))
            return np.multiply(a, b)

        data = self.get_2d_test_points()
        y_hat = f_hat(data)
        y = f(data)

        self.assertTrue(np.isclose(y_hat, y).all())

    def test_morokoff_calfisch_1_function_1d(self):
        d = 1

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.MOROKOFF_CALFISCH_1, d)
        f = lambda t: (2 * t).squeeze()

        data_1 = self.get_1d_test_points()
        data_2 = self.get_1d_test_points_2()
        y_hat_1 = f_hat(data_1)
        y_1 = f(data_1)
        y_hat_2 = f_hat(data_2)
        y_2 = f(data_2)

        self.assertTrue(np.isclose(y_hat_1, y_1).all())
        self.assertTrue(np.isclose(y_hat_2, y_2).all())

    def test_morokoff_calfisch_1_function_2d(self):
        d = 2

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.MOROKOFF_CALFISCH_1, d)

        def f(x):
            val = 2.25 * np.sqrt(x[:, 0]) * np.sqrt(x[:, 1])
            return val

        data = self.get_2d_test_points()
        y_hat = f_hat(data)
        y = f(data)

        self.assertTrue(np.isclose(y_hat, y, equal_nan=True).all())

    def test_morokoff_calfisch_2_function_1d(self):
        d = 1

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.MOROKOFF_CALFISCH_2, d)
        f = lambda t: (2 * (1 - t)).squeeze()

        data_1 = self.get_1d_test_points()
        data_2 = self.get_1d_test_points_2()
        y_hat_1 = f_hat(data_1)
        y_1 = f(data_1)
        y_hat_2 = f_hat(data_2)
        y_2 = f(data_2)

        self.assertTrue(np.isclose(y_hat_1, y_1).all())
        self.assertTrue(np.isclose(y_hat_2, y_2).all())

    def test_morokoff_calfisch_2_function_2d(self):
        d = 2

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.MOROKOFF_CALFISCH_2, d)

        def f(x):
            val = 1.5 ** (-2) * (2 - x[:, 0]) * (2 - x[:, 1])
            return val

        data = self.get_2d_test_points()
        y_hat = f_hat(data)
        y = f(data)

        self.assertTrue(np.isclose(y_hat, y).all())

    def test_roos_arnold_function_1d(self):
        d = 1

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.ROOS_ARNOLD, d)
        f = lambda t: np.abs(4 * t - 2).squeeze()

        data_1 = self.get_1d_test_points()
        data_2 = self.get_1d_test_points_2()
        y_hat_1 = f_hat(data_1)
        y_1 = f(data_1)
        y_hat_2 = f_hat(data_2)
        y_2 = f(data_2)

        self.assertTrue(np.isclose(y_hat_1, y_1).all())
        self.assertTrue(np.isclose(y_hat_2, y_2).all())

    def test_roos_arnold_function_2d(self):
        d = 2

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.ROOS_ARNOLD, d)

        def f(x):
            val = np.abs(4 * x[:, 0] - 2) * np.abs(4 * x[:, 1] - 2)
            return val

        data = self.get_2d_test_points()
        y_hat = f_hat(data)
        y = f(data)

        self.assertTrue(np.isclose(y_hat, y).all())

    def test_bratley_function_1d(self):
        d = 1

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.BRATLEY, d)
        f = lambda t: (-t).squeeze()

        data_1 = self.get_1d_test_points()
        data_2 = self.get_1d_test_points_2()
        y_hat_1 = f_hat(data_1)
        y_1 = f(data_1)
        y_hat_2 = f_hat(data_2)
        y_2 = f(data_2)

        self.assertTrue(np.isclose(y_hat_1, y_1).all())
        self.assertTrue(np.isclose(y_hat_2, y_2).all())

    def test_bratley_function_2d(self):
        d = 2

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.BRATLEY, d)

        def f(x):
            val = -x[:, 0] + x[:, 0] * x[:, 1]
            return val

        data = self.get_2d_test_points()
        y_hat = f_hat(data)
        y = f(data)

        self.assertTrue(np.isclose(y_hat, y).all())

    def test_zhou_function_1d(self):
        d = 1

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.ZHOU, d)

        def f(x):
            phi_1 = 1 / ((2 * np.pi) ** (1 / 2)) * np.exp(-0.5 * np.square(10 * x - 10 / 3))
            phi_2 = 1 / ((2 * np.pi) ** (1 / 2)) * np.exp(-0.5 * np.square(10 * x - 20 / 3))

            val = (5 * (phi_1 + phi_2)).squeeze()
            return val

        data_1 = self.get_1d_test_points()
        data_2 = self.get_1d_test_points_2()
        y_hat_1 = f_hat(data_1)
        y_1 = f(data_1)
        y_hat_2 = f_hat(data_2)
        y_2 = f(data_2)

        self.assertTrue(np.isclose(y_hat_1, y_1).all())
        self.assertTrue(np.isclose(y_hat_2, y_2).all())

    def test_zhou_function_2d(self):
        d = 2

        f_hat = ParametrizedFunctionProvider.get_function(FunctionType.ZHOU, d)

        def f(x):
            phi_1 = 1 / (2 * np.pi) * np.exp(-0.5 * np.sum(np.square(10 * x - 10 / 3), axis=1))
            phi_2 = 1 / (2 * np.pi) * np.exp(-0.5 * np.sum(np.square(10 * x - 20 / 3), axis=1))

            val = (50 * (phi_1 + phi_2)).squeeze()
            return val

        data = self.get_2d_test_points()
        y_hat = f_hat(data)
        y = f(data)

        self.assertTrue(np.isclose(y_hat, y).all())


if __name__ == '__main__':
    unittest.main()
