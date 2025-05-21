import argparse
import os
from typing import Union

from experiments.experiment_executor import ExperimentExecutor
from fit import BasisType
from fit.method.interpolation_method import InterpolationMethod
from fit.method.least_squares_method import LeastSquaresMethod
from function.type import FunctionType
from grid import TasmanianGridType
from plot.plot_distribution import plot_all_errors_fixed_dim, plot_all_errors_fixed_scale


def main_method(folder_name: Union[str, None] = None):
    dim_scale_dict = {
        2: [1, 2, 3, 4, 5, 6, 7, 8, 9],
        3: [1, 2, 3, 4, 5, 6, 7, 8, 9],
        4: [1, 2, 3, 4, 5, 6, 7, 8, 9],
        5: [1, 2, 3, 4, 5, 6, 7, 8],
        6: [1, 2, 3, 4, 5, 6, 7],
        7: [1, 2, 3, 4, 5, 6, 7],
        8: [1, 2, 3, 4, 5, 6],
        9: [1, 2, 3, 4, 5, 6],
        10: [1, 2, 3, 4, 5, 6],
    }

    function_types = [FunctionType.ZHOU, FunctionType.CONTINUOUS, FunctionType.CORNER_PEAK,
                      FunctionType.DISCONTINUOUS, FunctionType.GAUSSIAN, FunctionType.MOROKOFF_CALFISCH_1,
                      FunctionType.G_FUNCTION, FunctionType.OSCILLATORY, FunctionType.PRODUCT_PEAK, FunctionType.NOISE]

    seed = 42

    average_c = {
        FunctionType.CONTINUOUS: 1.0,
        FunctionType.CORNER_PEAK: 1.0,
        FunctionType.DISCONTINUOUS: 1.0,
        FunctionType.GAUSSIAN: 1.0,
        FunctionType.G_FUNCTION: 1.0,
        FunctionType.OSCILLATORY: 1.0,
        FunctionType.MOROKOFF_CALFISCH_1: 1.0,
        FunctionType.PRODUCT_PEAK: 1.0,
        FunctionType.ZHOU: 1.0,
        FunctionType.NOISE: 1.0
    }

    multiplier_fun = lambda x: 2 * x
    n_fun_parallel = 10

    store_indices = True

    smolyak_method_type = InterpolationMethod.TASMANIAN
    ls_method_type = LeastSquaresMethod.SCIPY_LSTSQ_GELSY
    least_squares_basis_type = BasisType.CHEBYSHEV
    tasmanian_grid_type = TasmanianGridType.STANDARD_GLOBAL

    if folder_name is not None:
        path = os.path.join("results", folder_name, "results_numerical_experiments.csv")
    else:
        path = None

    ex = ExperimentExecutor(dim_scale_dict, smolyak_method_type, least_squares_method=ls_method_type,
                            seed=seed, ls_basis_type=least_squares_basis_type, tasmanian_grid_type=tasmanian_grid_type,
                            path=path, store_indices=store_indices)
    ex.execute_experiments(function_types, n_fun_parallel, avg_c=average_c, ls_multiplier_fun=multiplier_fun)

    # Plot error distribution
    plot_all_errors_fixed_dim(file_name=ex.results_path, save=True, latex=True, only_maximum=False)
    plot_all_errors_fixed_dim(file_name=ex.results_path, save=True, latex=True, only_maximum=True)
    # plot_all_errors_fixed_scale(file_name=ex.results_path, save=True, latex=True, only_maximum=False)
    # plot_all_errors_fixed_scale(file_name=ex.results_path, save=True, latex=True, only_maximum=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the main method and store the results in the given folder')
    parser.add_argument('-f', '--folder_name', default=None, type=str, required=False,
                        help='The name of the folder where the results will be stored')
    args = parser.parse_args()
    main_method(folder_name=args.folder_name)
