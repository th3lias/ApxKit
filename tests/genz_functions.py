import unittest
from typing import Callable
import numpy as np
from enum import Enum


class GenzFunctionType(Enum):
    OSCILLATORY = 1,
    PRODUCT_PEAK = 2,
    CORNER_PEAK = 3,
    GAUSSIAN = 4,
    CONTINUOUS = 5,
    DISCONTINUOUS = 6


def get_genz_function(function_type: GenzFunctionType, c: np.array, w: np.array, d: int) -> Callable:
    """
    Creates a callable function from the Genz family and given hyperparameters c and w.
    Note that in the original definition, the functions are only defined for [0,1]^d
    :param function_type: Specifies which function we want to create.
    :param c: The higher this parameter, the more difficult the function gets
    :param w: Operates as a shift parameter
    :param d: dimension of the function
    :return: A callable function
    """

    if type(function_type) != GenzFunctionType:
        raise ValueError("Wrong input type for function_type. Use a GenzFunctionType")

    if function_type == GenzFunctionType.OSCILLATORY:
        return lambda x: np.cos(np.inner(c, x) + 2 * np.pi * w[0]).squeeze()

    if function_type == GenzFunctionType.PRODUCT_PEAK:
        return lambda x: np.prod(1 / (1 / (np.square(c)) + np.square(x - w))).squeeze()

    if function_type == GenzFunctionType.CORNER_PEAK:
        return lambda x: 1 / (np.power((1 + np.inner(c, x)), d + 1)).squeeze()

    if function_type == GenzFunctionType.GAUSSIAN:
        return lambda x: np.exp(-np.sum(np.square(np.multiply(c, x - w)))).squeeze()

    if function_type == GenzFunctionType.CONTINUOUS:
        return lambda x: np.exp(-np.sum(np.multiply(c, np.abs(x - w)))).squeeze()

    if function_type == GenzFunctionType.DISCONTINUOUS:
        def f(x):
            if d == 1:
                if x[0] > w[0]:
                    return 0
                else:
                    return np.exp(np.inner(c, x)).squeeze()
            if d > 1:
                if x[0] > w[0] or x[1] > w[1]:
                    return 0
                else:
                    return np.exp(np.inner(c, x)).squeeze()

        return f


class MyTestCase(unittest.TestCase):

    @staticmethod
    def get_1d_testpoints():

        test_points = [0, 1, 1 / 3, -1, -0.25, 1]

        return test_points

    @staticmethod
    def get_2d_testpoints():
        test_points = [np.array([0, 0]), np.array([4, 10]), np.array([1, 1]), np.array([-1, 1]), np.array([2, 0])]

        return test_points

    def test_osciallatory_1d_1(self):
        d = 1

        f_hat = get_genz_function(GenzFunctionType.OSCILLATORY, np.array([1]), np.array([0]), d)
        f = lambda t: np.cos(t)

        for x in self.get_1d_testpoints():
            self.assertAlmostEqual(f_hat(x), f(x), msg=f"x={x}")

    def test_osciallatory_1d_2(self):
        d = 1

        f_hat = get_genz_function(GenzFunctionType.OSCILLATORY, np.array([0.5]), np.array([-0.25]), d)
        f = lambda t: np.sin(0.5 * t)

        for x in self.get_1d_testpoints():
            self.assertAlmostEqual(f_hat(x), f(x), msg=f"x={x}")

    def test_osciallatory_2d_1(self):
        d = 2

        f_hat = get_genz_function(GenzFunctionType.OSCILLATORY, np.array([-1, 2]), np.array([-2.1, 0]), d)
        f = lambda t: np.cos(2 * np.pi * -2.1 - t[0] + 2 * t[1])

        for x in self.get_2d_testpoints():
            self.assertAlmostEqual(f_hat(x), f(x), msg=f"x={x}")

    def test_osciallatory_2d_2(self):
        d = 2

        f_hat = get_genz_function(GenzFunctionType.OSCILLATORY, np.array([-12, -12]), np.array([0, 0]), d)
        f = lambda t: np.cos(- 12 * t[0] - 12 * t[1])

        for x in self.get_2d_testpoints():
            self.assertAlmostEqual(f_hat(x), f(x), msg=f"x={x}")

    def test_productpeak_1d_1(self):
        d = 1

        f_hat = get_genz_function(GenzFunctionType.PRODUCT_PEAK, c=np.array([np.sqrt(2)]), w=np.array([3 / 8]), d=d)
        f = lambda t: 1 / (np.power(np.sqrt(2), -2) + np.square(t - 3 / 8))

        for x in self.get_1d_testpoints():
            self.assertAlmostEqual(f_hat(x), f(x), msg=f"x={x}")

    def test_productpeak_1d_2(self):
        d = 1

        f_hat = get_genz_function(GenzFunctionType.PRODUCT_PEAK, c=np.array([3]), w=np.array([1]), d=d)
        f = lambda t: 1 / (1 / (np.square(3)) + np.square((t - 1)))

        for x in self.get_1d_testpoints():
            self.assertAlmostEqual(f_hat(x), f(x), msg=f"x={x}")

    def test_productpeak_2d_1(self):
        d = 2

        f_hat = get_genz_function(GenzFunctionType.PRODUCT_PEAK, c=np.array([0.2, 1.5]), w=np.array([0.5, 0.25]), d=d)
        f = lambda t: 1 / ((np.power(0.2, -2) + np.square(t[0] - 0.5)) * (np.power(1.5, -2) + np.square(t[1] - 0.25)))

        for x in self.get_2d_testpoints():
            self.assertAlmostEqual(f_hat(x), f(x), msg=f"x={x}")

    def test_productpeak_2d_2(self):
        d = 2

        f_hat = get_genz_function(GenzFunctionType.PRODUCT_PEAK, c=np.array([-0.7, 0.2]), w=np.array([2.1, -1.0]), d=d)
        f = lambda t: 1 / ((np.power(-0.7, -2) + np.square(t[0] - 2.1)) * (np.power(0.2, -2) + np.square(t[1] + 1)))

        for x in self.get_2d_testpoints():
            self.assertAlmostEqual(f_hat(x), f(x), msg=f"x={x}")

    def test_cornerpeak_1d_1(self):
        d = 1

        f_hat = get_genz_function(GenzFunctionType.CORNER_PEAK, c=np.array([np.exp(1)]), w=None, d=d)
        f = lambda t: 1 / (np.power((1 + np.exp(1) * t), d + 1))

        for x in self.get_1d_testpoints():
            self.assertAlmostEqual(f_hat(x), f(x), msg=f"x={x}")

    def test_cornerpeak_1d_2(self):
        d = 1

        f_hat = get_genz_function(GenzFunctionType.CORNER_PEAK, c=np.array([1]), w=None, d=d)
        f = lambda t: 1 / (np.power((1 + t), d + 1))

        for x in self.get_1d_testpoints():
            self.assertAlmostEqual(f_hat(x), f(x), msg=f"x={x}")

    def test_cornerpeak_2d_1(self):
        d = 2

        f_hat = get_genz_function(GenzFunctionType.CORNER_PEAK, c=np.array([0.2, 1.5]), w=None, d=d)
        f = lambda t: np.power((1 + 0.2 * t[0] + 1.5 * t[1]), -d - 1)

        for x in self.get_2d_testpoints():
            self.assertAlmostEqual(f_hat(x), f(x), msg=f"x={x}")

    def test_cornerpeak_2d_2(self):
        d = 2

        f_hat = get_genz_function(GenzFunctionType.CORNER_PEAK, c=np.array([-0.7, 0.2]), w=None, d=d)
        f = lambda t: np.power((1 + -0.7 * t[0] + 0.2 * t[1]), -d - 1)

        for x in self.get_2d_testpoints():
            self.assertAlmostEqual(f_hat(x), f(x), msg=f"x={x}")

    def test_gaussian_1d_1(self):
        d = 1

        f_hat = get_genz_function(GenzFunctionType.GAUSSIAN, c=np.array([0.6]), w=np.array([0.35]), d=d)
        f = lambda t: np.exp(-0.36 * (t - 0.35) ** 2)

        for x in self.get_1d_testpoints():
            self.assertAlmostEqual(f_hat(x), f(x), msg=f"x={x}")

    def test_gaussian_1d_2(self):
        d = 1

        f_hat = get_genz_function(GenzFunctionType.GAUSSIAN, c=np.array([-0.1]), w=np.array([-0.1]), d=d)
        f = lambda t: np.exp(-0.01 * (t + 0.1) ** 2)

        for x in self.get_1d_testpoints():
            self.assertAlmostEqual(f_hat(x), f(x), msg=f"x={x}")

    def test_gaussian_2d_1(self):
        d = 2

        f_hat = get_genz_function(GenzFunctionType.GAUSSIAN, c=np.array([0.2, 1.5]), w=np.array([1, 2]), d=d)
        f = lambda t: np.exp(-0.04 * (t[0] - 1) ** 2 - 2.25 * (t[1] - 2) ** 2)

        for x in self.get_2d_testpoints():
            self.assertAlmostEqual(f_hat(x), f(x), msg=f"x={x}")

    def test_gaussian_2d_2(self):
        d = 2

        f_hat = get_genz_function(GenzFunctionType.GAUSSIAN, c=np.array([np.sqrt(2), 1 / 6]), w=np.array([0, -2]), d=d)
        f = lambda t: np.exp(-2 * (t[0]) ** 2 - 1 / 36 * (t[1] + 2) ** 2)

        for x in self.get_2d_testpoints():
            self.assertAlmostEqual(f_hat(x), f(x), msg=f"x={x}")

    def test_continuous_1d_1(self):
        d = 1

        f_hat = get_genz_function(GenzFunctionType.CONTINUOUS, c=np.array([0.6]), w=np.array([0.35]), d=d)
        f = lambda t: np.exp(-0.6 * np.abs(t - 0.35))

        for x in self.get_1d_testpoints():
            self.assertAlmostEqual(f_hat(x), f(x), msg=f"x={x}")

    def test_continuous_1d_2(self):
        d = 1

        f_hat = get_genz_function(GenzFunctionType.CONTINUOUS, c=np.array([-0.1]), w=np.array([-0.1]), d=d)
        f = lambda t: np.exp(0.1 * np.abs(t + 0.1))

        for x in self.get_1d_testpoints():
            self.assertAlmostEqual(f_hat(x), f(x), msg=f"x={x}")

    def test_continuous_2d_1(self):
        d = 2

        f_hat = get_genz_function(GenzFunctionType.CONTINUOUS, c=np.array([0.2, 1.5]), w=np.array([1, 2]), d=d)
        f = lambda t: np.exp(-0.2 * np.abs(t[0] - 1) - 1.5 * np.abs(t[1] - 2))

        for x in self.get_2d_testpoints():
            self.assertAlmostEqual(f_hat(x), f(x), msg=f"x={x}")

    def test_continuous_2d_2(self):
        d = 2

        f_hat = get_genz_function(GenzFunctionType.CONTINUOUS, c=np.array([np.sqrt(2), 1 / 6]), w=np.array([0, -2]),
                                  d=d)
        f = lambda t: np.exp(-np.sqrt(2) * np.abs(t[0]) - 1 / 6 * np.abs(t[1] + 2))

        for x in self.get_2d_testpoints():
            self.assertAlmostEqual(f_hat(x), f(x), msg=f"x={x}")

    def test_discontinuous_1d_1(self):
        d = 1

        f_hat = get_genz_function(GenzFunctionType.DISCONTINUOUS, c=np.array([0.6]), w=np.array([0.35]), d=d)

        def f(t):
            if t > 0.35:
                return 0
            else:
                return np.exp(0.6 * t)

        for x in self.get_1d_testpoints():
            self.assertAlmostEqual(f_hat(np.array([x])), f(x), msg=f"x={x}")

    def test_discontinuous_1d_2(self):
        d = 1

        f_hat = get_genz_function(GenzFunctionType.DISCONTINUOUS, c=np.array([-0.1]), w=np.array([-0.1]), d=d)

        def f(t):
            if t > -0.1:
                return 0
            else:
                return np.exp(-0.1 * t)

        for x in self.get_1d_testpoints():
            self.assertAlmostEqual(f_hat(np.array([x])), f(x), msg=f"x={x}")

    def test_discontinuous_2d_1(self):
        d = 2

        f_hat = get_genz_function(GenzFunctionType.DISCONTINUOUS, c=np.array([0.2, 1.5]), w=np.array([1, 2]), d=d)

        def f(t):
            if t[0] > 1 or t[1] > 2:
                return 0
            else:
                return np.exp(0.2 * t[0] + 1.5 * t[1])

        for x in self.get_2d_testpoints():
            self.assertAlmostEqual(f_hat(x), f(x), msg=f"x={x}")

    def test_discontinuous_2d_2(self):
        d = 2

        f_hat = get_genz_function(GenzFunctionType.DISCONTINUOUS, c=np.array([np.sqrt(2), 1 / 6]), w=np.array([0, -2]),
                                  d=d)

        def f(t):
            if t[0] > 0 or t[1] > -2:
                return 0
            else:
                return np.exp(np.sqrt(2) * t[0] + 1 / 6 * t[1])

        for x in self.get_2d_testpoints():
            self.assertAlmostEqual(f_hat(x), f(x), msg=f"x={x}")


if __name__ == '__main__':
    unittest.main()
