import numpy as np


def oscillatory(x, d, c, w):
    """
        Oscillatory function.
    """
    return np.cos(np.dot(x, c) + 2 * np.pi * w[0])


def product_peak(x, d, c, w):
    """
        Product peak function.
    """
    return np.prod(1 / (1/np.square(c)) + np.square(x - w), axis=1)


def corner_peak(x, d, c, w):
    """
        Corner peak function.
    """
    return np.power(1+np.dot(x, c), -(d+1))


def gaussian(x, d, c, w):
    """
        Gaussian function.
    """
    return np.exp(-np.sum(np.square(np.multiply(c, x - w)), axis=1))


def continuous(x, d, c, w):
    """
        Continuous function.
    """
    return np.exp(-np.sum(np.multiply(c, np.abs(x - w)), axis=1))


def discountinuous_1d(x, d, c, w):
    return np.array([0 if i > w[0] else np.exp(c[0] * i) for i in x])


def discountinuous_nd(x, d, c, w):
    return np.array([0 if i[0] > w[0] or i[1] > w[1] else np.exp(np.inner(i, c)) for i in x])


def g_function(x, d, c, w):
    """
        G-function.
    """
    return np.prod(
        np.divide(
            np.abs(4 * x - 2 - w) + c,
            1 + c
        ), axis=1
    )


def morokoff_calfisch_1(x, d, c, w):
    """
        Morokoff Calfisch function.
    """
    return (1+1/d)**d * np.prod(np.multiply(x, c) + w, axis=1)**(1/d)


def morokoff_calfisch_2(x, d, c, w):
    """
        Morokoff Calfisch function.
    """
    return 1/(d-1/2)**d * np.prod(d - np.multiply(c, x) - w, axis=1)


def roos_arnold(x, d, c, w):
    """
        Roos Arnold Function.
    """
    return np.prod(np.abs(4 * np.multiply(c, x) - 2 - w), axis=1)


def bratley(x, d, c, w):
    """
        Bratley function.
    """
    return np.sum(np.power(-1, np.arange(1, d+1))) * np.prod(np.multiply(c, x) - w, axis=1)


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
