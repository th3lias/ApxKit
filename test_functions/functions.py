from typing import Callable, Union

import numpy as np

from test_functions.function_types import FunctionType


def get_test_function(function_type: FunctionType, d: int, c: Union[np.array, None] = None,
                      w: Union[np.array, None] = None) -> tuple[Callable, np.ndarray, np.ndarray]:
    """
    Creates a callable function from the various kind of function (families).
    They're defined in the interval [0,1]^d.
    The first 6 functions are so-called Genz functions.
    The other 6 are other common functions to benchmark integration problems. For a reference, see
    https://www.sfu.ca/~ssurjano/integration.html
    Note, that only the Genz-Functions depend on the parameter c and w.
    :param function_type: Specifies which function we want to create.
    :param d: dimension of the function
    :param c: The higher this parameter, the more difficult the function gets, only relevant for Genz functions
    :param w: Operates as a shift parameter, only relevant for Genz functions
    :return: A callable function
    """

    if not isinstance(function_type, FunctionType):
        raise ValueError("Wrong input type for function_type. Use a FunctionType")

    if not isinstance(d, int):
        raise ValueError("Dimension must be an integer")

    if function_type == FunctionType.OSCILLATORY:

        if c is None or w is None:
            raise ValueError("c and w must be specified for the Oscillatory function")

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

        return f, c, w

    if function_type == FunctionType.PRODUCT_PEAK:

        if c is None or w is None:
            raise ValueError("c and w must be specified for the Product Peak function")

        def f(x):
            if not isinstance(x, np.ndarray):
                raise ValueError("Cannot work with non-numpy arrays")
            if x.ndim == 1:
                return 1 / (1 / (np.square(c)) + np.square(x - w)).squeeze()
            elif x.ndim == 2:
                return np.prod(1 / (1 / (np.square(c)) + np.square(x - w)), axis=1).squeeze()
            else:
                raise ValueError(f"Cannot handle an array with number ofa dimension ={x.ndim}")

        return f, c, w

    if function_type == FunctionType.CORNER_PEAK:

        if c is None:
            raise ValueError("c must be specified for the Corner Peak function")

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

        return f, c, w

    if function_type == FunctionType.GAUSSIAN:

        if c is None or w is None:
            raise ValueError("c and w must be specified for the Gaussian function")

        def f(x):
            if not isinstance(x, np.ndarray):
                raise ValueError("Cannot work with non-numpy arrays")
            if x.ndim == 1:
                return np.exp(-np.square(np.multiply(c, x - w))).squeeze()
            elif x.ndim == 2:
                return np.exp(-np.sum(np.square(np.multiply(c, x - w)), axis=1)).squeeze()
            else:
                raise ValueError(f"Cannot handle an array with number of dimension ={x.ndim}")

        return f, c, w

    if function_type == FunctionType.CONTINUOUS:

        if c is None or w is None:
            raise ValueError("c and w must be specified for the Continuous function")

        def f(x):
            if not isinstance(x, np.ndarray):
                raise ValueError("Cannot work with non-numpy arrays")
            if x.ndim == 1:
                return np.exp(-np.multiply(c, np.abs(x - w))).squeeze()
            elif x.ndim == 2:
                return np.exp(-np.sum(np.multiply(c, np.abs(x - w)), axis=1)).squeeze()
            else:
                raise ValueError(f"Cannot handle an array with number of dimension ={x.ndim}")

        return f, c, w

    if function_type == FunctionType.DISCONTINUOUS:

        if c is None or w is None:
            raise ValueError("c and w must be specified for the Discontinuous function")

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
        return f, c, w

    if function_type == FunctionType.G_FUNCTION:

        if c is None:
            c = (np.arange(1, d + 1, dtype=np.float64) - 2) / 2

        if w is None:
            w = np.zeros(d)

        def f(x):

            if not isinstance(x, np.ndarray):
                raise ValueError("Cannot work with non-numpy arrays")

            x = x.squeeze()
            if x.ndim == 1:
                if d != 1:
                    return np.prod(np.divide((np.abs(4 * x - 2 - w) + c), 1 + c), axis=1).squeeze()
                else:
                    return np.array([np.divide(np.abs(4 * i - 2 - w[0]) + c[0], (1 + c[0])) for i in x])
            elif x.ndim == 2:
                return np.prod(np.divide((np.abs(4 * x - 2 - w) + c), 1 + c), axis=1).squeeze()
            else:
                raise ValueError(f"Cannot handle an array with number of dimension ={x.ndim}")

        return f, c, w

    if function_type == FunctionType.MOROKOFF_CALFISCH_1:

        if c is None:
            c = np.ones(d)

        if w is None:
            w = np.zeros(d)

        def f(x):

            if not isinstance(x, np.ndarray):
                raise ValueError("Cannot work with non-numpy arrays")

            x = x.squeeze()
            if x.ndim == 1:
                if d != 1:
                    return (1 + 1 / d) ** d * (np.prod(np.multiply(x, c) + w, axis=1) ** (1 / d)).squeeze()
                else:
                    return np.array([2 * (i * c[0] + w[0]) for i in x])
            elif x.ndim == 2:
                return (1 + 1 / d) ** d * (np.prod(np.multiply(x, c) + w, axis=1) ** (1 / d)).squeeze()
            else:
                raise ValueError(f"Cannot handle an array with number of dimension ={x.ndim}")

        return f, c, w

    if function_type == FunctionType.MOROKOFF_CALFISCH_2:
        if c is None:
            c = np.ones(d)

        if w is None:
            w = np.zeros(d)

        def f(x):

            if not isinstance(x, np.ndarray):
                raise ValueError("Cannot work with non-numpy arrays")

            x = x.squeeze()
            if x.ndim == 1:
                if d != 1:
                    return ((d - 1 / 2) ** (-d) * np.prod(d - np.multiply(c, x) - w, axis=1)).squeeze()
                else:
                    return np.array([2 * (1 - c[0] * i - w[0]) for i in x])
            elif x.ndim == 2:
                return ((d - 1 / 2) ** (-d) * np.prod(d - np.multiply(c, x) - w, axis=1)).squeeze()
            else:
                raise ValueError(f"Cannot handle an array with number of dimension ={x.ndim}")

        return f, c, w

    if function_type == FunctionType.ROOS_ARNOLD:
        if c is None:
            c = np.ones(d)

        if w is None:
            w = np.zeros(d)

        def f(x):

            if not isinstance(x, np.ndarray):
                raise ValueError("Cannot work with non-numpy arrays")

            x = x.squeeze()
            if x.ndim == 1:
                if d != 1:
                    return np.prod(np.abs(4 * np.multiply(c, x) - 2 - w), axis=1).squeeze()
                else:
                    return np.array([np.abs(4 * (i * c[0]) - 2 - w[0]) for i in x])
            elif x.ndim == 2:
                return np.prod(np.abs(4 * np.multiply(c, x) - 2 - w), axis=1).squeeze()
            else:
                raise ValueError(f"Cannot handle an array with number of dimension ={x.ndim}")

        return f, c, w

    if function_type == FunctionType.BRATLEY:
        if c is None:
            c = np.ones(d)

        if w is None:
            w = np.zeros(d)

        def f(x):

            if not isinstance(x, np.ndarray):
                raise ValueError("Cannot work with non-numpy arrays")

            x = x.squeeze()
            if x.ndim == 1:
                if d != 1:
                    val = 0
                    for i in range(1, d + 1):
                        prod = 1
                        for j in range(1, i + 1):
                            prod *= (c[j] * x[j] - w)

                        val += (-1) ** i * prod
                    return np.array(val).squeeze()
                else:
                    return np.array([-i * c[0] + w[0] for i in x])
            elif x.ndim == 2:
                if d != 1:
                    val = np.zeros(x.shape[0])
                    for i in range(1, d + 1):
                        prod = np.ones_like(val)
                        for j in range(1, i + 1):
                            prod *= (c[j - 1] * x[:, j - 1] - w[j - 1])

                        val += (-1) ** i * prod
                    return np.array(val).squeeze()
            else:
                raise ValueError(f"Cannot handle an array with number of dimension ={x.ndim}")

        return f, c, w

    if function_type == FunctionType.ZHOU:

        if c is None:
            c = 10 * np.ones(d)

        if w is None:
            w = 1 / 3 * np.ones(d)

        def f(x):

            if not isinstance(x, np.ndarray):
                raise ValueError("Cannot work with non-numpy arrays")

            x = x.squeeze()
            if x.ndim == 1:
                if d != 1:
                    phi_1 = (2 * np.pi) ** (-d / 2) * np.exp(-np.sum(np.square(np.multiply(c, (x - w))), axis=1) / 2)
                    phi_2 = (2 * np.pi) ** (-d / 2) * np.exp(
                        -np.sum(np.square(np.multiply(c, (x - 1 + w))), axis=1) / 2)

                    return (10 ** d * 0.5 * (phi_1 + phi_2)).squeeze()
                else:
                    phi_1 = [(2 * np.pi) ** (-1 / 2) * np.exp(-np.sum(np.square(c[0] * (i - w[0]))) / 2) for i in x]
                    phi_2 = [(2 * np.pi) ** (-1 / 2) * np.exp(-np.sum(np.square(c[0] * (i - 1 + w[0]))) / 2) for i in x]
                    return np.array([5 * (phi_1[i] + phi_2[i]) for i in range(len(x))])
            elif x.ndim == 2:
                phi_1 = (2 * np.pi) ** (-d / 2) * np.exp(-np.sum(np.square(np.multiply(c, (x - w))), axis=1) / 2)
                phi_2 = (2 * np.pi) ** (-d / 2) * np.exp(-np.sum(np.square(np.multiply(c, (x - 1 + w))), axis=1) / 2)

                return (10 ** d * 0.5 * (phi_1 + phi_2)).squeeze()
            else:
                raise ValueError(f"Cannot handle an array with number of dimension ={x.ndim}")

        return f, c, w
