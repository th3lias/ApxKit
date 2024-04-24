import numpy as np
import sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from typing import Callable, Union
from genz.genz_functions import get_genz_function
from genz.genz_functions import GenzFunctionType
from utils.utils import ell_2_error_estimate
from scipy.sparse.linalg import lsmr


def approximate_by_polynomial_with_least_squares(f: Callable, dim: np.int8, degree: Union[np.int8, None],
                                                 grid: np.ndarray) -> sklearn.linear_model:
    n_samples = grid.shape[0]

    if degree is None:
        degree = n_samples - 1

    if np.shape(grid)[1] != dim:
        raise ValueError("Grid dimension must be equal to input dimension of f")

    y = f(grid)

    poly = PolynomialFeatures(degree=degree, include_bias=False)

    X_poly = poly.fit_transform(grid)

    res = lsmr(X_poly, y)

    coef = res[0]
    n_iter = res[2]

    print(f'Number of iterations: {n_iter}')
    print(coef) # returns combin(d + n, n) coefficients (without intercept)

    # model = LinearRegression()
    # model.fit(X_poly, y)

    # return model

    return coef




if __name__ == '__main__':
    # TODO: Does not work well (but maybe functional -> not sure for now) TODO: Basis of multivariate polynomial
    #  function grows with combin(d+n, d) where d is the dimension and n the number of variables
    np.random.seed(23)
    n_samples = 5
    n_test_samples = 5

    dimension = np.int8(5)

    c = np.random.uniform(size=dimension)
    w = np.random.uniform(size=dimension)

    # works for OSCILLATORY, CORNER_PEAK, CONTINUOUS, PRODUCT_PEAK, DISCONTINUOUS, CORNER_PEAK
    # does not work for CONTINUOUS (dimension mismatch)

    f = get_genz_function(GenzFunctionType.DISCONTINUOUS, d=dimension, c=c, w=w)

    data = np.random.uniform(0, 1, (n_samples, dimension))

    # model = approximate_by_polynomial_with_least_squares(f, dimension, None, data)
    # print(model.coef_)

    coef = approximate_by_polynomial_with_least_squares(f, dimension, None, data)

    test_data = np.random.uniform(low=0, high=1, size=(n_test_samples, dimension))

    poly = PolynomialFeatures(degree=n_samples - 1, include_bias=False)

    test_data_poly = poly.fit_transform(test_data)

    # y_hat = model.predict(test_data_poly)
    y = f(test_data)
    y_hat = test_data_poly@coef
    print(y_hat)
    print(y)

    err = np.mean(np.square(y-y_hat))
    print(f'Error: {err}')
    # print(y_hat)
    # print(y)
