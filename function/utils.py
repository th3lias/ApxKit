"""
Standard test function implementations from the Genz family and others.

All functions have the signature ``f(x, d, c, w) → np.ndarray`` where
*x* is an (n, d) array of evaluation points and *c*, *w* are parameter
vectors of length d.

References: https://www.sfu.ca/~ssurjano/integration.html
"""

import numpy as np


def oscillatory(x, d, c, w):
    return np.cos(3 * np.dot(x, c) + 2 * np.pi * w[0])


def product_peak(x, d, c, w):
    return np.prod(1 / ((1 / np.square(c)) + np.square(x - w)), axis=1)


def corner_peak(x, d, c, w):
    return np.power(1 + np.dot(x, c), -(d + 1))


def gaussian(x, d, c, w):
    return np.exp(-(50 / d) * np.sum(np.square(np.multiply(c, x - w)), axis=1))


def continuous(x, d, c, w):
    return np.exp(-np.sum(np.multiply(c, np.abs(x - w)) / d, axis=1))


def discontinuous(x, d, c, w):
    if d == 1:
        x_1d = np.asarray(x).ravel()
        return np.where(x_1d > w[0], 0.0, np.exp(c[0] * x_1d))
    else:
        return np.array([0 if i[0] > w[0] or i[1] > w[1] else np.exp(np.inner(i, c)) for i in x])


def g_function(x, d, c, w):
    return np.prod(np.divide(np.abs(4 * x - 2 - w) + c, 1 + c), axis=1)


def morokoff_calfisch_1(x, d, c, w):
    return (1 + 1 / d) ** d * np.prod(np.multiply(x, c) + w, axis=1) ** (1 / d)


def morokoff_calfisch_2(x, d, c, w):
    return np.multiply((1 / (d - 1 / 2) ** d), np.prod(d - np.multiply(c, x) + w, axis=1))


def roos_arnold(x, d, c, w):
    return np.prod(np.abs(4 * np.multiply(c, x) - 2 - w), axis=1)


def bratley(x, d, c, w):
    return np.sum(np.multiply(np.power(-1, np.arange(1, d + 1)), np.cumprod(np.multiply(c, x) - w, axis=1)), axis=1)


def zhou(x, d, c, w):
    """Bimodal Gaussian (Zhou function)."""
    x = np.squeeze(x)
    if x.ndim not in (1, 2):
        raise ValueError(f"Cannot handle an array with number of dimensions = {x.ndim}")
    if d == 1 and x.ndim == 1:
        x = x[:, np.newaxis]

    phi_1 = np.exp(-(50 / d) * np.sum((c * (x - w)) ** 2, axis=-1))
    phi_2 = np.exp(-(50 / d) * np.sum((c * (x + w - 1)) ** 2, axis=-1))
    return (phi_1 + phi_2).squeeze()


def noise(x, d, c, w):
    """Gaussian noise (used as training-only surrogate for the zero function)."""
    return np.random.normal(0, 1e-7, x.shape[0])


def zero(x, d, c, w):
    """Exact zero function (used as test counterpart of ``noise``)."""
    return np.zeros(x.shape[0], dtype=np.float64)
