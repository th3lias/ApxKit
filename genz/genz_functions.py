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

    if not isinstance(function_type, GenzFunctionType):
        raise ValueError("Wrong input type for function_type. Use a GenzFunctionType")

    if function_type == GenzFunctionType.OSCILLATORY:
        def f(x):

            if not isinstance(x, np.ndarray):
                raise ValueError("Cannot work with non-numpy arrays")

            x = x.squeeze()
            if x.ndim == 1:
                if d != 1:
                    return np.cos(np.inner(c, x) + 2 * np.pi * w[0]).squeeze()
                else:
                    return np.array([np.cos(i * c[0] + 2 * np.pi * w[0]) for i in x])
            elif x.ndim == 2:
                return np.cos(np.inner(c, x) + 2 * np.pi * w[0]).squeeze()
            else:
                raise ValueError(f"Cannot handle an array with number of dimension ={x.ndim}")

        return f

    if function_type == GenzFunctionType.PRODUCT_PEAK:

        def f(x):
            if not isinstance(x, np.ndarray):
                raise ValueError("Cannot work with non-numpy arrays")
            if x.ndim == 1:
                return 1 / (1 / (np.square(c)) + np.square(x - w)).squeeze()
            elif x.ndim == 2:
                return np.prod(1 / (1 / (np.square(c)) + np.square(x - w)), axis=1).squeeze()
            else:
                raise ValueError(f"Cannot handle an array with number of dimension ={x.ndim}")

        return f

    if function_type == GenzFunctionType.CORNER_PEAK:

        def f(x):
            if not isinstance(x, np.ndarray):
                raise ValueError("Cannot work with non-numpy arrays")
            if x.ndim == 1:
                if d != 1:
                    return 1 / (np.power((1 + np.inner(c, x)), d + 1)).squeeze()
                else:
                    return 1 / np.array([np.power(1 + i * c[0], d + 1) for i in x])
            elif x.ndim == 2:
                return 1 / (np.power((1 + np.inner(c, x)), d + 1)).squeeze()
            else:
                raise ValueError(f"Cannot handle an array with number of dimension ={x.ndim}")

        return f

    if function_type == GenzFunctionType.GAUSSIAN:
        def f(x):
            if not isinstance(x, np.ndarray):
                raise ValueError("Cannot work with non-numpy arrays")
            if x.ndim == 1:
                return np.exp(-np.square(np.multiply(c, x - w))).squeeze()
            elif x.ndim == 2:
                return np.exp(-np.sum(np.square(np.multiply(c, x - w)), axis=1)).squeeze()
            else:
                raise ValueError(f"Cannot handle an array with number of dimension ={x.ndim}")

        return f

    if function_type == GenzFunctionType.CONTINUOUS:
        def f(x):
            if not isinstance(x, np.ndarray):
                raise ValueError("Cannot work with non-numpy arrays")
            if x.ndim == 1:
                return np.exp(-np.multiply(c, np.abs(x - w))).squeeze()
            elif x.ndim == 2:
                return np.exp(-np.sum(np.multiply(c, np.abs(x - w)), axis=1)).squeeze()
            else:
                raise ValueError(f"Cannot handle an array with number of dimension ={x.ndim}")

        return f

    if function_type == GenzFunctionType.DISCONTINUOUS:
        if d == 1:
            def f(x):
                if not isinstance(x, np.ndarray):
                    raise ValueError("Cannot work with non-numpy arrays")
                x = x.squeeze()
                return np.array([0 if i > w[0] else np.exp(c[0] * i) for i in x])

        elif d > 1:
            def f(x):
                if not isinstance(x, np.ndarray):
                    raise ValueError("Cannot work with non-numpy arrays")
                x = x.squeeze()
                return np.array([0 if i[0] > w[0] or i[1] > w[1] else np.exp(np.inner(i, c)) for i in x])
        else:
            raise ValueError("Wrong dimension!")
        return f

    # if function_type == GenzFunctionType.T_ULLRICH_1:
    #
    #     def f(x):
    #         if not isinstance(x, np.ndarray):
    #             raise ValueError("Cannot work with non-numpy arrays")
    #         if x.ndim == 1:
    #             return np.sqrt(1.5) * np.sqrt(np.sqrt(1-np.abs(2*(np.remainder(x, 1))-1)))
    #         elif x.ndim == 2:
    #             return (np.power(1.5, d/2.0)* np.prod(np.sqrt(np.sqrt(1-np.abs(2*(np.remainder(x, 1))-1))))).squeeze()
    #         else:
    #             raise ValueError(f"Cannot handle an array with number of dimension ={x.ndim}")
    #
    #     return f

