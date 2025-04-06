import argparse
import datetime
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
    dim_list = [2, 3, 4, 5, 6, 7, 8]
    scale_list = [1, 2, 3, 4, 5]

    function_types = [FunctionType.OSCILLATORY, FunctionType.PRODUCT_PEAK, FunctionType.CORNER_PEAK,
                      FunctionType.GAUSSIAN, FunctionType.CONTINUOUS, FunctionType.DISCONTINUOUS,
                      FunctionType.G_FUNCTION, FunctionType.MOROKOFF_CALFISCH_1, FunctionType.MOROKOFF_CALFISCH_2,
                      FunctionType.ROOS_ARNOLD, FunctionType.BRATLEY, FunctionType.ZHOU]

    seed = 42

    average_c = 1.0
    multiplier_fun = lambda x: 2 * x
    n_fun_parallel = 50

    store_indices = True

    smolyak_method_type = InterpolationMethod.STANDARD
    ls_method_type = LeastSquaresMethod.SCIPY_LSTSQ_GELSY
    least_squares_basis_type = BasisType.CHEBYSHEV
    tasmanian_grid_type = TasmanianGridType.STANDARD_GLOBAL

    if folder_name is not None:
        path = os.path.join("results", folder_name, "results_numerical_experiments.csv")
    else:
        path = None

    ex = ExperimentExecutor(dim_list, scale_list, smolyak_method_type, least_squares_method=ls_method_type,
                            seed=seed, ls_basis_type=least_squares_basis_type, tasmanian_grid_type=tasmanian_grid_type,
                            path=path, store_indices=store_indices)
    ex.execute_experiments(function_types, n_fun_parallel, avg_c=average_c, ls_multiplier_fun=multiplier_fun)

    folder_name = os.path.dirname(ex.results_path)

    # Plot distribution
    plot_all_errors_fixed_dim(file_name=ex.results_path, save=True, latex=True)
    plot_all_errors_fixed_scale(file_name=ex.results_path, save=True, latex=True)

    # save all images in results folder
    total_iterations = len(dim_list) * len(function_types)
    with tqdm(total=total_iterations, desc="Plotting the results") as pbar:
        for dim in dim_list:
            for fun_type in function_types:
                plot_errors(dim, seed, fun_type, scale_list, multiplier_fun, save=True, folder_name=folder_name,
                            same_axis_both_plots=True, latex=True)
                pbar.update(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the main method and store the results in the given folder')
    parser.add_argument('-f', '--folder_name', default=None, type=str, required=False,
                        help='The name of the folder where the results will be stored')
    args = parser.parse_args()
    main_method(folder_name=args.folder_name)
