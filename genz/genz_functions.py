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
        return lambda x: np.prod(1 / (1 / (np.square(c)) + np.square(x - w)), axis=1).squeeze()

    if function_type == GenzFunctionType.CORNER_PEAK:
        return lambda x: 1 / (np.power((1 + np.inner(c, x)), d + 1)).squeeze()

    if function_type == GenzFunctionType.GAUSSIAN:
        return lambda x: np.exp(-np.sum(np.square(np.multiply(c, x - w)), axis=1)).squeeze()

    if function_type == GenzFunctionType.CONTINUOUS:
        return lambda x: np.exp(-np.sum(np.multiply(c, np.abs(x - w)), axis=1)).squeeze()

    if function_type == GenzFunctionType.DISCONTINUOUS:
        """
        # TODO: Vectorize this

        if d == 1:
            def f(x):
                for row in x:
                    if row[0] > w[0]:
                        return 0
                    else:
                        return np.exp(np.inner(c, x)).squeeze()

                return np.where(x[:, 0] > w[0], 0, np.exp(np.inner(c, x)).squeeze())
        elif d > 1:
            def f(x):
                for row in x:
                    if row[0] > w[0] or row[1] > w[1]:
                        return 0
                    else:
                        return np.exp(np.inner(c, x)).squeeze()
        else:
            raise ValueError("Wrong dimension!")
        return f

"""

        if d == 1:
            def f(x):
                x = x.squeeze()
                mask = x > w[0]
                res = np.exp(c[0]*x)
                res[mask] = 0
                return res
        elif d > 1:
            def f(x):
                x = x.squeeze()
                mask1 = x[:,0] > w[0]
                mask2 = x[:,1] > w[1]
                mask = np.empty_like(mask1)
                np.bitwise_or(mask1, mask2, out=mask)
                res = np.exp(np.inner(x, c)).squeeze()
                res[mask] = 0
                return res
        else:
            raise ValueError("Wrong dimension!")
        return f


if __name__ == '__main__':
    import numpy as np
    import sklearn
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from typing import Callable, Union
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
        print(coef)  # returns combin(d + n, n) coefficients (without intercept)

        # model = LinearRegression()
        # model.fit(X_poly, y)

        # return model

        return coef


    np.random.seed(2302)
    n_samples = 10
    n_test_samples = 5

    dimension = np.int8(2)

    c = np.random.uniform(size=dimension)
    w = np.random.uniform(size=dimension)

    f = get_genz_function(GenzFunctionType.DISCONTINUOUS, d=dimension, c=c, w=w)

    data = np.random.uniform(0, 1, (n_samples, dimension))

    # model = approximate_by_polynomial_with_least_squares(f, dimension, None, data)
    # print(model.coef_)

    coef = approximate_by_polynomial_with_least_squares(f, dimension, np.int8(2), data)

    test_data = np.random.uniform(low=0, high=1, size=(n_test_samples, dimension))

    poly = PolynomialFeatures(degree=n_samples - 1, include_bias=False)

    test_data_poly = poly.fit_transform(test_data)

    # y_hat = model.predict(test_data_poly)
    y = f(test_data)
    y_hat = test_data_poly @ coef
    print(y_hat)
    print(y)

    err = np.mean(np.square(y - y_hat))
    print(f'Error: {err}')
    # print(y_hat)
    # print(y)
