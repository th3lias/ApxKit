import numpy as np
import sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from typing import Callable
from scipy.sparse.linalg import lsmr


def approximate_by_polynomial_with_least_squares_iterative(f: Callable, dim: np.int8, degree: np.int8,
                                                           grid: np.ndarray) -> np.ndarray:
    """
    Approximation of a function with a polynomial with least squares iterative approach (using the lsmr algorithm).
    :param f: function that needs to be approximated
    :param dim: dimension of the data
    :param degree: maximum allowed degree of the polynomials
    :param grid: data array containing the points where the function should be approximated
    :return: coefficients
    """
    if np.shape(grid)[1] != dim:
        raise ValueError("Grid dimension must be equal to input dimension of f")

    y = f(grid)

    poly = PolynomialFeatures(degree=degree, include_bias=False)

    X_poly = poly.fit_transform(grid)

    res = lsmr(X_poly, y)

    coef = res[0]

    return coef


def approximate_by_polynomial_with_least_squares(f: Callable, dim: np.int8, degree: np.int8,
                                                 grid: np.ndarray) -> sklearn.linear_model:
    """
    Approximates a function with a polynomial with least squares approach.
    :param f: function that needs to be approximated
    :param dim: dimension of the data
    :param degree: degree of the polynomials
    :param grid: data array containing the points where the function should be approximated
    :return: sklearn linear model
    """
    if np.shape(grid)[1] != dim:
        raise ValueError("Grid dimension must be equal to input dimension of f")

    y = f(grid)

    poly = PolynomialFeatures(degree=degree, include_bias=False)

    X_poly = poly.fit_transform(grid)

    model = LinearRegression()
    model.fit(X_poly, y)

    return model
