import numpy as np
from utils.utils import ell_2_error_estimate, max_abs_error

from typing import Callable
import matplotlib.pyplot as plt

from grid.grid_provider import GridProvider, GridType
from genz.genz_functions import GenzFunctionType, get_genz_function
from smolyak.smolyak import SmolyakInterpolation
from least_squares.least_squares import approximate_by_polynomial_with_least_squares as least_squares


def test_parameters_like_in_2000_paper(fun_type: GenzFunctionType, scale: np.int8, sum_c: np.float16,
                                       test_grid: np.ndarray, dim: np.int8 = 10,
                                       lower_bound: np.float16 = np.float16(0.0),
                                       upper_bound: np.float16 = np.float16(1.0)):
    """
    tests the parameters such that we can figure out the corresponding hyperparameter c and w in order to reproduce results from the paper from 2000
    :param fun_type: type of function that should be approximated
    :param scale: defines the fineness of the sparse grid
    :param sum_c: special sum of the coefficients; needed for reproducible results of the paper from 2000
    :param dim: dimension of the grid and of the function
    :param test_grid: grid where the approximation should be assessed
    :param lower_bound: lower bound where the function should be approximated
    :param upper_bound upper bound where the function should be approximated
    """

    w = np.random.uniform(low=lower_bound, high=upper_bound, size=dim)
    c = np.random.uniform(low=lower_bound, high=upper_bound, size=dim)
    c = c / np.sum(c)

    c = sum_c * c

    f = get_genz_function(fun_type, c, w, dim)

    sy = SmolyakInterpolation(dim, scale)
    n_samples = sy.grid.shape[0]

    f_hat_smolyak = sy.approximate(f)

    train_grid = GridProvider(dim).generate(GridType.RANDOM, scale=n_samples)

    f_hat_ls = least_squares(f, dim, np.int8(3), train_grid, include_bias=True, self_implemented=True)

    print(f'{fun_type.name}: dimension:{dim}, scale: {scale}')

    print(f'c={c}')
    print('______')
    print('Smolyak:')
    print(f'max_abs_error = {max_abs_error(f, f_hat_smolyak, grid=test_grid)}')
    print(f'L2_error = {ell_2_error_estimate(f, f_hat_smolyak, grid=test_grid)}')
    # print('______')
    # print('Least-Squares:')
    # print(f'max_abs_error = {max_abs_error(f, f_hat_ls, grid=test_grid)}')
    # print(f'L2_error = {ell_2_error_estimate(f, f_hat_ls, grid=test_grid)}')
    print('___________________________________')


def reproduce_graphics_from_paper_2000(f: Callable, name: str, test_grid: np.ndarray, degree_ls: np.int8, scale_range:range):
    results = dict()
    results['smolyak'] = dict()
    results['least_squares'] = dict()

    for scale in scale_range:
        sy = SmolyakInterpolation(dim, np.int8(scale))

        print("Done with Smolyak Interpolation")

        n_samples = sy.grid.shape[0]

        f_hat_smolyak = sy.approximate(f)

        print("Done with Smolyak Approximation")

        train_grid = GridProvider(dim).generate(GridType.RANDOM, scale=n_samples)

        f_hat_ls = least_squares(f, dim, degree_ls, train_grid, include_bias=True, self_implemented=True)

        print("Done with Least Squares Approximation")

        max_abs_ls = max_abs_error(f, f_hat_ls, grid=test_grid)
        max_abs_smolyak = max_abs_error(f, f_hat_smolyak, grid=test_grid)
        ell_2_ls = ell_2_error_estimate(f, f_hat_ls, grid=test_grid)
        ell_2_smolyak = ell_2_error_estimate(f, f_hat_smolyak, grid=test_grid)

        results['smolyak'][scale] = {'max_diff': max_abs_smolyak, 'ell_2': ell_2_smolyak}
        results['least_squares'][scale] = {'max_diff': max_abs_ls, 'ell_2': ell_2_ls}

        print(f'Done with scale {scale} for {name}')

    # Plotting
    # TODO: Maybe make own method out of this plotting.
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    scales = scale_range
    smolyak_max_diff = [results['smolyak'][scale]['max_diff'] for scale in scales]
    least_squares_max_diff = [results['least_squares'][scale]['max_diff'] for scale in scales]

    smolyak_ell_2 = [results['smolyak'][scale]['ell_2'] for scale in scales]
    least_squares_ell_2 = [results['least_squares'][scale]['ell_2'] for scale in scales]

    axs[0].plot(scales, smolyak_max_diff, label='Smolyak')
    axs[0].plot(scales, least_squares_max_diff, label='Least Squares')
    axs[0].set_xticks(scale_range)
    axs[0].set_title('Max (Abs) Error')
    axs[0].set_xlabel('Scale')
    axs[0].set_ylabel('Max Error')
    axs[0].set_yscale('log')
    axs[0].legend()

    axs[1].plot(scales, smolyak_ell_2, label='Smolyak')
    axs[1].plot(scales, least_squares_ell_2, label='Least Squares')
    axs[1].set_xticks(scale_range)
    axs[1].set_title('L2 Error')
    axs[1].set_xlabel('Scale')
    axs[1].set_ylabel('L2 Error')
    axs[1].set_yscale('log')
    axs[1].legend()

    fig.suptitle(name)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    dim = np.int8(10)
    lower_bound = np.float16(0.0)
    upper_bound = np.float16(1.0)
    n_test_samples = np.int8(50)

    test_grid = GridProvider(dim).generate(GridType.RANDOM, scale=n_test_samples)

    # Tests for finding parameter c

    print('___________________________________')
    # test_parameters_like_in_2000_paper(GenzFunctionType.OSCILLATORY, np.int8(1), sum_c=np.float16(9.0), test_grid)
    # test_parameters_like_in_2000_paper(GenzFunctionType.PRODUCT_PEAK, np.int8(1), np.float16(7.25), test_grid)
    # test_parameters_like_in_2000_paper(GenzFunctionType.CORNER_PEAK, np.int8(1), np.float16(1.85))
    # test_parameters_like_in_2000_paper(GenzFunctionType.GAUSSIAN, np.int8(1), np.float16(7.03))
    # test_parameters_like_in_2000_paper(GenzFunctionType.CONTINUOUS, np.int8(1), np.float16(20.4))
    # test_parameters_like_in_2000_paper(GenzFunctionType.DISCONTINUOUS, np.int8(1), np.float16(4.3))

    # Tests as soon as parameter c is found
    w = np.random.uniform(low=lower_bound, high=upper_bound, size=dim)
    c = np.random.uniform(low=lower_bound, high=upper_bound, size=dim)
    c = c / np.sum(c)

    c = 9.0 * c
    f = get_genz_function(GenzFunctionType.OSCILLATORY, c=c, w=w, d=dim)

    scale_range = range(1,8)

    reproduce_graphics_from_paper_2000(f, "Osciallatory", test_grid, np.int8(3), scale_range)
