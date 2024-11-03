import numpy as np


def oscillatory(x, d, c, w):
    """
        Oscillatory function.
    """
    if x.ndim == 1:
        if d != 1:
            return np.cos(np.inner(c, x) + 2 * np.pi * w[0]).squeeze()
        else:
            return np.array([np.cos(i * c[0] + 2 * np.pi * w[0]) for i in x])
    elif x.ndim == 2:
        return np.cos(np.inner(c, x) + 2 * np.pi * w[0]).squeeze()
    else:
        raise ValueError(f"Cannot handle an array with number of dimension ={x.ndim}")


def product_peak(x, d, c, w):
    """
        Product peak function.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError("Cannot work with non-numpy arrays")
    if x.ndim == 1:
        return 1 / (1 / (np.square(c)) + np.square(x - w)).squeeze()
    elif x.ndim == 2:
        return np.prod(1 / (1 / (np.square(c)) + np.square(x - w)), axis=1).squeeze()
    else:
        raise ValueError(f"Cannot handle an array with number of dimension ={x.ndim}")


def corner_peak(x, d, c, w):
    """
        Corner peak function.
    """
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


def gaussian(x, d, c, w):
    """
        Gaussian function.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError("Cannot work with non-numpy arrays")
    if x.ndim == 1:
        return np.exp(-np.square(np.multiply(c, x - w))).squeeze()
    elif x.ndim == 2:
        return np.exp(-np.sum(np.square(np.multiply(c, x - w)), axis=1)).squeeze()
    else:
        raise ValueError(f"Cannot handle an array with number of dimension ={x.ndim}")


def continuous(x, d, c, w):
    """
        Continuous function.
    """
    if not isinstance(x, np.ndarray):
        raise ValueError("Cannot work with non-numpy arrays")
    if x.ndim == 1:
        return np.exp(-np.multiply(c, np.abs(x - w))).squeeze()
    elif x.ndim == 2:
        return np.exp(-np.sum(np.multiply(c, np.abs(x - w)), axis=1)).squeeze()
    else:
        raise ValueError(f"Cannot handle an array with number of dimension ={x.ndim}")


def discountinuous_1d(x, d, c, w):
    x = x.squeeze()
    return np.array([0 if i > w[0] else np.exp(c[0] * i) for i in x])


def discountinuous_nd(x, d, c, w):
    x = x.squeeze()
    return np.array([0 if i[0] > w[0] or i[1] > w[1] else np.exp(np.inner(i, c)) for i in x])


def g_function(x, d, c, w):
    """
        G-function.
    """
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


def morokoff_calfisch_1(x, d, c, w):
    """
        Morokoff Calfisch function.
    """
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


def morokoff_calfisch_2(x, d, c, w):
    """
        Morokoff Calfisch function.
    """
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


def roos_arnold(x, d, c, w):
    """
        Roos Arnold Function.
    """
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


def bratley(x, d, c, w):
    """
        Bratley function.
    """
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


def zhou(x, d, c, w):
    """
        Zhou function.
    """
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
