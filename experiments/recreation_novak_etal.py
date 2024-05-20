from typing import Callable

import numpy as np

from genz.genz_functions import GenzFunctionType, get_genz_function
from grid.grid import Grid
from grid.grid_provider import GridProvider, GridType
from interpolate.least_squares import LeastSquaresInterpolator
from interpolate.smolyak import SmolyakInterpolator
from utils import utils
from utils.utils import l2_error, max_abs_error, sample


def test_params_novak(fun_type: GenzFunctionType, scale: int, sum_c: float, grid: np.ndarray, dimension: int = 10,
                      lower: float = 0.0, upper: float = 1.0):
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

    w = sample(dimension, lower, upper)
    c = sample(dimension, lower, upper)
    c = c / np.sum(c)

    c = sum_c * c
    f = get_genz_function(fun_type, c, w, dimension)
    si = SmolyakInterpolator(dimension, scale)
    f_hat_smolyak = si.interpolate(f)

    print(f'{fun_type.name}: dimension:{dimension}, scale: {scale}')
    print(f'c={c}')
    print('______')
    print('Smolyak:')
    print(f'max_abs_error = {max_abs_error(f, f_hat_smolyak, grid=grid)}')
    print(f'L2_error = {l2_error(f, f_hat_smolyak, grid=grid)}')
    print('_' * 100)


def plots_novak(f: Callable, name: str, grid: Grid, degree_ls: int, scales: range):
    results = dict()
    results['smolyak'] = dict()
    results['least_squares'] = dict()

    lsq = LeastSquaresInterpolator(degree_ls, True, None, True, False)
    for scale in scales:
        print(f'Starting with scale {scale} for {name}')
        print(f'Starting Smolyak Approximation')
        si = SmolyakInterpolator(dim, scale)
        n_samples = si.grid.grid.shape[0]
        f_hat_smolyak = si.interpolate(f)
        print("Done with Smolyak Approximation")

        print(f'Starting Least Squares Approximation')
        train_grid = GridProvider(dim).generate(GridType.RANDOM, scale=n_samples)
        lsq.set_grid(train_grid)
        f_hat_ls = lsq.interpolate(f)
        print("Done with Least Squares Approximation")

        print(f'Calculating errors')
        max_abs_ls = max_abs_error(f, f_hat_ls, grid=grid.grid)
        max_abs_smolyak = max_abs_error(f, f_hat_smolyak, grid=grid.grid)
        ell_2_ls = l2_error(f, f_hat_ls, grid=grid.grid)
        ell_2_smolyak = l2_error(f, f_hat_smolyak, grid=grid.grid)

        results['smolyak'][scale] = {'max_diff': max_abs_smolyak, 'ell_2': ell_2_smolyak}
        results['least_squares'][scale] = {'max_diff': max_abs_ls, 'ell_2': ell_2_ls}

        print(f'Done with scale {scale} for {name}')

    utils.plot_error_vs_scale(results, scales, name)


if __name__ == '__main__':
    dim = 10
    lower_bound = float(0.0)
    upper_bound = float(1.0)
    n_test_samples = 50

    test_grid = GridProvider(dim, lower_bound=lower_bound, upper_bound=upper_bound).generate(GridType.RANDOM,
                                                                                             scale=n_test_samples)

    # Tests for finding parameter c

    print('_' * 100)

    # test_params_novak(GenzFunctionType.OSCILLATORY, 1, 9.0, test_grid)
    # test_params_novak(GenzFunctionType.PRODUCT_PEAK, 1, 7.25, test_grid)
    # test_params_novak(GenzFunctionType.CORNER_PEAK, 1, 1.85, test_grid)
    # test_params_novak(GenzFunctionType.GAUSSIAN, 1, 7.03, test_grid)
    # test_params_novak(GenzFunctionType.CONTINUOUS, 1, 20.4, test_grid)
    # test_params_novak(GenzFunctionType.DISCONTINUOUS, 1, 4.3, test_grid)
    # Tests as soon as parameter c is found

    _w = np.random.uniform(low=lower_bound, high=upper_bound, size=dim)
    _c = np.random.uniform(low=lower_bound, high=upper_bound, size=dim)
    _c = _c / np.sum(_c)

    _c = 9.0 * _c
    fun = get_genz_function(GenzFunctionType.OSCILLATORY, c=_w, w=_w, d=dim)

    scale_range = range(1, 4)
    plots_novak(fun, "Oscillatory", test_grid, int(3), scale_range)
