import numpy as np

from utils import utils
from utils.utils import l2_error, max_abs_error, min_abs_error

from typing import Callable

from grid.grid_provider import GridProvider, GridType
from genz.genz_functions import GenzFunctionType, get_genz_function
from smolyak.smolyak import SmolyakInterpolation
from least_squares.least_squares import approximate_by_polynomial_with_least_squares as least_squares


def test_params_novak(fun_type: GenzFunctionType, scale: np.int8, sum_c: np.float16,
                      grid: np.ndarray, dimension: np.int8 = 10,
                      lower: np.float16 = np.float16(0.0),
                      upper: np.float16 = np.float16(1.0)):
    """
    tests the parameters such that we can figure out the corresponding hyperparameter c and w in order to reproduce
    results from the paper from 2000
    :param fun_type: type of function that should be approximated
    :param scale: defines the fineness of the sparse grid
    :param sum_c: special sum of the coefficients; needed for reproducible results of the paper from 2000
    :param dimension: dimension of the grid and of the function
    :param grid: grid where the approximation should be assessed
    :param lower: lower bound where the function should be approximated
    :param upper: upper bound where the function should be approximated
    """

    w = np.random.uniform(low=lower, high=upper, size=dimension)
    c = np.random.uniform(low=lower, high=upper, size=dimension)
    c = c / np.sum(c)

    c = sum_c * c

    f = get_genz_function(fun_type, c, w, dimension)

    sy = SmolyakInterpolation(dimension, scale)

    f_hat_smolyak = sy.approximate(f)

    print(f'{fun_type.name}: dimension:{dimension}, scale: {scale}')
    print(f'c={c}')
    print('______')
    print('Smolyak:')
    print(f'max_abs_error = {max_abs_error(f, f_hat_smolyak, grid=grid)}')
    print(f'L2_error = {l2_error(f, f_hat_smolyak, grid=grid)}')
    print('_' * 100)


def plots_novak(f: Callable, name: str, grid: np.ndarray, degree_ls: np.int8, scales: range):
    results = dict()
    results['smolyak'] = dict()
    results['least_squares'] = dict()

    for scale in scales:
        sy = SmolyakInterpolation(dim, np.int8(scale))

        print("Done with Smolyak Interpolation")

        n_samples = sy.grid.shape[0]

        f_hat_smolyak = sy.approximate(f)

        print("Done with Smolyak Approximation")

        train_grid = GridProvider(dim).generate(GridType.RANDOM, scale=n_samples)

        f_hat_ls = least_squares(f, dim, degree_ls, train_grid, include_bias=True, self_implemented=True)

        print("Done with Least Squares Approximation")

        max_abs_ls = max_abs_error(f, f_hat_ls, grid=grid)
        max_abs_smolyak = max_abs_error(f, f_hat_smolyak, grid=grid)
        min_abs_ls = min_abs_error(f, f_hat_ls, grid=grid)
        min_abs_smolyak = min_abs_error(f, f_hat_smolyak, grid=grid)
        ell_2_ls = l2_error(f, f_hat_ls, grid=grid)
        ell_2_smolyak = l2_error(f, f_hat_smolyak, grid=grid)

        results['smolyak'][scale] = {'max_diff': max_abs_smolyak, 'ell_2': ell_2_smolyak, 'min_diff': min_abs_smolyak}
        results['least_squares'][scale] = {'max_diff': max_abs_ls, 'ell_2': ell_2_ls, 'min_diff': min_abs_ls}

        print(f'Done with scale {scale} for {name}')

    utils.plot_error_vs_scale(results, scales, name)


if __name__ == '__main__':
    dim = np.int8(10)
    lower_bound = np.float16(0.0)
    upper_bound = np.float16(1.0)
    n_test_samples = np.int8(50)

    test_grid = GridProvider(dim, lower_bound=lower_bound, upper_bound=upper_bound).generate(GridType.RANDOM, scale=n_test_samples)

    # Tests for finding parameter c

    print('_' * 100)

    # test_params_novak(GenzFunctionType.OSCILLATORY, np.int8(1), np.float16(9.0), test_grid)
    # test_params_novak(GenzFunctionType.PRODUCT_PEAK, np.int8(1), np.float16(7.25), test_grid)
    # test_params_novak(GenzFunctionType.CORNER_PEAK, np.int8(1), np.float16(1.85), test_grid)
    # test_params_novak(GenzFunctionType.GAUSSIAN, np.int8(1), np.float16(7.03), test_grid)
    # test_params_novak(GenzFunctionType.CONTINUOUS, np.int8(1), np.float16(20.4), test_grid)
    # test_params_novak(GenzFunctionType.DISCONTINUOUS, np.int8(1), np.float16(4.3), test_grid)
    # Tests as soon as parameter c is found

    w = np.random.uniform(low=lower_bound, high=upper_bound, size=dim)
    c = np.random.uniform(low=lower_bound, high=upper_bound, size=dim)
    c = c / np.sum(c)

    c = 9.0 * c
    f = get_genz_function(GenzFunctionType.OSCILLATORY, c=c, w=w, d=dim)

    scale_range = range(1, 7)
    plots_novak(f, "Oscillatory", test_grid, np.int8(3), scale_range)
