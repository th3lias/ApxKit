from typing import Callable
import numpy as np
from genz.genz_function_types import GenzFunctionType


def get_genz_function(function_type: GenzFunctionType, c: np.array, w: np.array, d: int) -> Callable:
    """
    Creates a callable function from the Genz family and given hyperparameters c and w.
    Note that in the original definition, the functions are only defined for [0,1]^d
    For the Genz functions, see https://link.springer.com/chapter/10.1007/978-94-009-3889-2_33
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
        if d == 1:
            def f(x):
                if x[0] > w[0]:
                    return 0
                else:
                    return np.exp(np.inner(c, x)).squeeze()
        if d > 1:
            def f(x):
                if x[0] > w[0] or x[1] > w[1]:
                    return 0
                else:
                    return np.exp(np.inner(c, x)).squeeze()
        return f