import argparse
import os
from typing import Union

from tqdm import tqdm

from experiments.experiment_executor import ExperimentExecutor
from fit import BasisType
from fit.method.interpolation_method import InterpolationMethod
from fit.method.least_squares_method import LeastSquaresMethod
from function.type import FunctionType
from grid import TasmanianGridType
from plot.plot_distribution import plot_all_errors_fixed_dim, plot_all_errors_fixed_scale
from plot.plot_function import plot_errors


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
                      FunctionType.G_FUNCTION, FunctionType.OSCILLATORY, FunctionType.PRODUCT_PEAK]

    function_types = [FunctionType.GAUSSIAN, FunctionType.ZHOU, FunctionType.OSCILLATORY, FunctionType.PRODUCT_PEAK]

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
        FunctionType.ZHOU: 1.0
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

    folder_name = os.path.dirname(ex.results_path)

    # Plot error distribution
    plot_all_errors_fixed_dim(file_name=ex.results_path, save=True, latex=True, only_maximum=False)
    plot_all_errors_fixed_dim(file_name=ex.results_path, save=True, latex=True, only_maximum=True)
    # plot_all_errors_fixed_scale(file_name=ex.results_path, save=True, latex=True, only_maximum=False)
    # plot_all_errors_fixed_scale(file_name=ex.results_path, save=True, latex=True, only_maximum=True)

    # Plot errors for each function
    # total_iterations = 0
    # for _ in dim_scale_dict.values():
    #     total_iterations += 1
    # total_iterations *= len(function_types)
    # with tqdm(total=total_iterations, desc="Plotting the results") as pbar:
    #     for dim in dim_scale_dict.keys():
    #         for fun_type in function_types:
    #             plot_errors(dim, seed, fun_type, dim_scale_dict.get(dim), multiplier_fun, save=True,
    #                         folder_name=folder_name,
    #                         same_axis_both_plots=True, latex=True)
    #             pbar.update(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the main method and store the results in the given folder')
    parser.add_argument('-f', '--folder_name', default=None, type=str, required=False,
                        help='The name of the folder where the results will be stored')
    args = parser.parse_args()
    main_method(folder_name=args.folder_name)
