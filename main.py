import argparse
import os
from typing import Union

from experiments.experiment_executor import ExperimentExecutor
from fit import BasisType
from fit.method.interpolation_method import InterpolationMethod
from fit.method.least_squares_method import LeastSquaresMethod
from function.type import FunctionType
from grid import TasmanianGridType
from grid.rule.random_grid_rule import RandomGridRule
from plot.plot_distribution import plot_all_errors_fixed_dim, plot_all_errors_fixed_scale


def main_method(folder_name: Union[str, None] = None):
    dim_scale_dict = {
        # 2: [1, 2, 3, 4, 5, 6, 7, 8, 9],
        # 3: [1, 2, 3, 4, 5, 6, 7, 8, 9],
        # 4: [1, 2, 3, 4, 5, 6, 7, 8, 9],
        # 5: [1, 2, 3, 4, 5, 6, 7, 8],
        # 6: [1, 2, 3, 4, 5, 6, 7],
        # 7: [1, 2, 3, 4, 5, 6, 7],
        # 8: [1, 2, 3, 4, 5, 6],
        # 9: [1, 2, 3, 4, 5, 6],
        10: [1, 2, 3, 4, 5, 6],
    }

    function_types = [FunctionType.BNR_OSCILLATORY, FunctionType.BNR_PRODUCT_PEAK, FunctionType.BNR_CORNER_PEAK,
                      FunctionType.BNR_GAUSSIAN, FunctionType.BNR_CONTINUOUS, FunctionType.BNR_DISCONTINUOUS]

    seed = 42

    average_c = {
        FunctionType.BNR_CONTINUOUS: 2.04,
        FunctionType.BNR_CORNER_PEAK: 0.185,
        FunctionType.BNR_DISCONTINUOUS: 0.43,
        FunctionType.BNR_GAUSSIAN: 0.703,
        FunctionType.BNR_OSCILLATORY: 0.9,
        FunctionType.BNR_PRODUCT_PEAK: 0.725,
    }

    multiplier_fun_ls_train = lambda x: 2 * x
    multiplier_fun_test = lambda x: x
    n_fun_parallel = 50

    store_indices = True

    smolyak_method_type = InterpolationMethod.TASMANIAN
    ls_method_type = LeastSquaresMethod.SCIPY_LSTSQ_GELSY
    least_squares_basis_type = BasisType.CHEBYSHEV
    tasmanian_grid_type = TasmanianGridType.STANDARD_GLOBAL
    test_rule = RandomGridRule.UNIFORM
    use_max_scale = False  # Whether to use the maximum scale for the test grid

    if folder_name is not None:
        path = os.path.join("results", folder_name, "results_numerical_experiments.csv")
    else:
        path = None

    ex = ExperimentExecutor(dim_scale_dict, smolyak_method_type, least_squares_method=ls_method_type,
                            seed=seed, ls_basis_type=least_squares_basis_type, tasmanian_grid_type=tasmanian_grid_type,
                            test_rule=test_rule, use_max_scale=use_max_scale, path=path,
                            store_indices=store_indices)
    ex.execute_experiments(function_types, n_fun_parallel, avg_c=average_c, ls_multiplier_fun=multiplier_fun_ls_train,
                           test_multiplier_fun=multiplier_fun_test)

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
