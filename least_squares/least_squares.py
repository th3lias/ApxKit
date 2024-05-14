import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from typing import Callable, Union, List
from scipy.sparse.linalg import lsmr


def approximate_by_polynomial_with_least_squares_iterative(f: Callable, dim: np.int8, degree: np.int8,
                                                           grid: np.ndarray, include_bias: bool) -> Callable:
    """
    Approximation of a function with a polynomial by least squares iterative approach (using the lsmr algorithm).
    :param f: function that needs to be approximated
    :param dim: dimension of the data
    :param degree: maximum allowed degree of the polynomials
    :param grid: data array containing the points where the function should be approximated
    :param include_bias: whether to include bias (equivalent to intercept) in the polynomial
    :return: fitted function
    """
    if np.shape(grid)[1] != dim:
        raise ValueError("Grid dimension must be equal to input dimension of f")

    y = f(grid)

    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)

    x_poly = poly.fit_transform(grid)

    res = lsmr(x_poly, y)

    coeff = res[0]

    def f_hat(x):
        pol = PolynomialFeatures(degree=degree, include_bias=include_bias)
        x_pol = pol.fit_transform(x)
        return x_pol @ coeff

    return f_hat


def approximate_by_polynomial_with_least_squares(f: Union[Callable, List[Callable]], dim: np.int8, degree: np.int8,
                                                 grid: np.ndarray, include_bias: bool,
                                                 self_implemented: bool = True) -> Callable:
    """
    Approximates a (or multiple) function(s) with polynomials by least squares.
    :param f: function or list of functions that need to be approximated on the same points
    :param dim: dimension of the data
    :param degree: degree of the polynomials
    :param grid: data array containing the points where the function should be approximated
    :param include_bias: whether to include bias (equivalent to intercept) in the polynomial
    :param self_implemented: whether to use the self-implemented version as it is faster and only relies on numpy
    :return: fitted function(s)
    """
    if np.shape(grid)[1] != dim:
        raise ValueError("Grid dimension must be equal to input dimension of f")

    if self_implemented and not include_bias:
        print("Please be aware that the result may become significantly worse when using no intercepts (bias)")

    if not isinstance(f, list) and not isinstance(f, Callable):
        raise ValueError(f"f needs to be a function or a list of functions but is {type(list)}")

    n_samples = grid.shape[0]

    if isinstance(f, list):
        y = np.empty(shape=(n_samples, len(f)), dtype=np.float64)
        for i, func in enumerate(f):
            if not isinstance(func, Callable):
                raise ValueError(f"One element of the list is not a function but from the type {type(func)}")
            y[:, i] = func(grid)
    else:
        y = f(grid)

    poly = PolynomialFeatures(degree=degree, include_bias=include_bias)

    x_poly = poly.fit_transform(grid)

    if self_implemented:
        y_prime = x_poly.T @ y
        del y
        coeff = np.linalg.solve(x_poly.T@x_poly, y_prime)

        def f_hat(x):
            pol = PolynomialFeatures(degree=degree, include_bias=include_bias)
            x_pol = pol.fit_transform(x)
            pred = x_pol @ coeff
            if pred.ndim == 2:
                return pred.T
            return pred

    else:
        model = LinearRegression()
        model.fit(x_poly, y)

        def f_hat(x):
            pol = PolynomialFeatures(degree=degree, include_bias=include_bias)
            x_pol = pol.fit_transform(x)
            return model.predict(x_pol)

    return f_hat
