import numpy as np
import sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from typing import Callable, Union
from genz.genz_functions import get_genz_function
from genz.genz_functions import GenzFunctionType


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

    model = LinearRegression()
    model.fit(X_poly, y)

    return model


if __name__ == '__main__':
    # TODO: Does not work well (but maybe functional -> not sure for now) TODO: Basis of multivariate polynomial
    #  function grows with combin(d+n, d) where d is the dimension and n the number of variables
    np.random.seed(23)
    n_samples = 25
    n_test_samples = 10

    d = np.int8(10)

    c = np.random.uniform(size=d)
    w = np.random.uniform(size=d)

    f = get_genz_function(GenzFunctionType.OSCILLATORY, d=d, c=c, w=w)

    data = np.random.uniform(0, 1, (n_samples, d))

    model = approximate_by_polynomial_with_least_squares(f, d, None, data)

    test_data = np.random.uniform(low=0, high=1, size=(n_test_samples, d))

    poly = PolynomialFeatures(degree=n_samples - 1, include_bias=False)

    test_data_poly = poly.fit_transform(test_data)

    y_hat = model.predict(test_data_poly)
    y = f(test_data)
    print(y_hat)
    print(y)
