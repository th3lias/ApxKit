import datetime
import os
import platform
import time
from typing import Union, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from test_functions.functions import FunctionType, get_test_function
from grid.grid import Grid
from grid.grid_provider import GridType, GridProvider
from interpolate.basis_types import BasisType
from interpolate.least_squares import LeastSquaresInterpolator
from interpolate.interpolation_methods import LeastSquaresMethod, SmolyakMethod
from interpolate.smolyak import SmolyakInterpolator
from utils.utils import max_error_function_values, l2_error_function_values
from utils.utils import calculate_num_points

from typing import Callable

import psutil


def run_experiments_smolyak(dim: int, w: np.ndarray, c: np.ndarray, f_types: list[FunctionType], seed_list: list[int],
                            n_parallel: int, n_avg_c: int, scale: int, grid: Union[Grid, None], test_grid_seed: int,
                            test_grid: Union[np.ndarray, Grid], lb: float, ub: float, method_type: SmolyakMethod,
                            folder_name: str, path: Union[str, None] = None) -> Grid:
    """
    Runs an experiment (or multiple depending on passed parameters) and appends the results to a results file
    :param dim: dimension of the grid/function
    :param w: shift-parameter for the test_functions-functions, can be multidimensional if multiple functions are used
    :param c: parameter for the test_functions-functions, can be multidimensional if multiple functions are used
    :param f_types: List of function types that should be tested
    :param seed_list: list of seeds that should be used for the experiments
    :param n_parallel: number of parallel functions per type
    :param n_avg_c: number of average c values that are used
    :param scale: related to the number of samples used to fit the smolyak model
    :param grid: grid on which the Smolyak Algorithm should operate. If None, a new grid will be created
    :param test_grid_seed: seed used to generate test grid
    :param test_grid: grid which is used to test the quality of the fit
    :param lb: lower bound of the interval
    :param ub: upper bound of the interval
    :param method_type: Specifies which type of solving algorithm should be used
    :param folder_name: Specifies the folder name where the results should be stored
    :param path: path of the results file. If None, the default path is used

    :return: The created grid, such that it can be used again for an increased scale
    """

    # TODO: Delete seed list. It is currently only used to ensure that we have a right plotting

    start_time = time.time()

    if isinstance(test_grid, Grid):
        test_grid = test_grid.grid

    n_function_types = int(len(f_types))

    n_samples = calculate_num_points(scale, dim)

    gp = GridProvider(dimension=dim, multiplier_fun=lambda x: x, lower_bound=lb, upper_bound=ub)

    if grid is None or not grid.dim == dim:
        grid = gp.generate(GridType.CHEBYSHEV, scale=scale)
    else:
        grid = gp.increase_scale(grid)

    functions = list()
    function_names = list()
    y = np.empty(shape=(n_parallel * n_function_types * n_avg_c, test_grid.shape[0]), dtype=np.float64)

    for i, func_type in enumerate(f_types):
        for j in range(n_parallel):
            for k in range(n_avg_c):
                index = i * n_parallel + j * n_avg_c + k
                f = get_test_function(function_type=func_type, d=dim, c=c[index, :], w=w[index, :])
                functions.append(f)
                y[index, :] = f(test_grid)
                function_names.append(func_type.name)

    si = SmolyakInterpolator(grid, method=method_type)
    si.fit(functions)

    y_hat = si.interpolate(test_grid)

    l_2_error = l2_error_function_values(y, y_hat).reshape(n_parallel * n_function_types * n_avg_c)
    max_error = max_error_function_values(y, y_hat).reshape(n_parallel * n_function_types * n_avg_c)

    end_time = time.time()
    needed_time = end_time - start_time

    cpu = platform.processor()
    cur_datetime = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    results = list()

    for i in range(n_parallel * n_function_types * n_avg_c):
        for seed in seed_list:
            row_entry = dict()
            row_entry['dim'] = dim
            row_entry['method'] = 'Smolyak'
            row_entry['w'] = w[i, :]
            row_entry['c'] = c[i, :]
            row_entry['sum_c'] = row_entry['c'].sum()
            row_entry['grid_type'] = si.grid.grid_type.name
            row_entry['basis_type'] = si.basis_type.name
            row_entry['method_type'] = method_type.name
            row_entry['n_samples'] = n_samples
            row_entry['scale'] = scale
            row_entry['seed'] = seed
            row_entry['test_grid_seed'] = test_grid_seed
            row_entry['f_name'] = function_names[i]
            row_entry['l_2_error'] = l_2_error[i]
            row_entry['max_error'] = max_error[i]
            row_entry['cpu'] = cpu
            row_entry['datetime'] = cur_datetime
            row_entry['needed_time'] = needed_time
            results.append(row_entry)

    if path is None:
        path = os.path.join("results", folder_name, "results_numerical_experiments.csv")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.isfile(path):
        data = pd.read_csv(path, sep=',', header=0)
    else:
        data = pd.DataFrame()

    new_data = pd.DataFrame(results)
    data = pd.concat([data, new_data], ignore_index=True)

    data['sum_c'] = data['sum_c'].apply(lambda x: round(x, 3))

    data.to_csv(path, sep=',', index=False)

    return grid


def run_experiments_least_squares(dim: int, w: np.ndarray, c: np.ndarray, f_types: list[FunctionType],
                                  n_parallel: int, n_avg_c: int, scale: int, seed: int, multiplier_fun: Callable,
                                  grid: Union[Grid, None],
                                  test_grid_seed: int, test_grid: Union[np.ndarray, Grid], lb: float, ub: float,
                                  grid_type: GridType, basis_type: BasisType, method_type: LeastSquaresMethod,
                                  folder_name: str, sample_new: bool = True, path: Union[str, None] = None) -> Grid:
    """
    Runs an experiment (or multiple depending on passed parameters) and appends the results to a results file
    :param dim: dimension of the grid/function
    :param w: shift-parameter for the test_functions-functions, can be multidimensional if multiple functions are used
    :param c: parameter for the test_functions-functions, can be multidimensional if multiple functions are used
    :param f_types: List of function types that should be tested
    :param n_parallel: number of parallel functions per type
    :param n_avg_c: number of average c values that are used
    :param scale: related to the number of samples used to fit the least-squares model
    :param seed: seed used to generate the training data
    :param multiplier_fun: Applies this function to the number of samples to obtain a new number of samples
    :param grid: Grid on which least-squares should be fitted. If None, a new grid is created
    :param test_grid_seed: seed used to generate test grid
    :param test_grid: grid which is used to test the quality of the fit
    :param lb: lower bound of the interval
    :param ub: upper bound of the interval
    :param grid_type: Specifies which grid should be used. Usually sampled from uniform or chebyshev weight
    :param basis_type: Specifies the type of the polynomial basis. If Chebyshev, then the exact same basis is used like
    in the Smolyak algorithm, otherwise a comparable standard basis
    :param method_type: Specifies which type of solving algorithm should be used
    :param folder_name: Specifies the folder name where the results should be stored
    :param sample_new: Specifies, whether the current points in the grid should be kept or newly sampled
    :param path: path of the results file. If None, the default path is used

    :return: The created grid, such that it can be used again for an increased scale
    """

    start_time = time.time()

    if isinstance(test_grid, Grid):
        test_grid = test_grid.grid

    n_samples = calculate_num_points(scale, dim)

    n_function_types = int(len(f_types))

    gp = GridProvider(dimension=dim, multiplier_fun=multiplier_fun, lower_bound=lb, upper_bound=ub, seed=seed)

    if grid is None or not grid.dim == dim:
        grid = gp.generate(grid_type, scale=scale)
    else:
        grid = gp.increase_scale(grid, sample_new)

    functions = list()
    function_names = list()
    y = np.empty(shape=(n_parallel * n_function_types * n_avg_c, test_grid.shape[0]), dtype=np.float64)

    for i, func_type in enumerate(f_types):
        for j in range(n_parallel):
            for k in range(n_avg_c):
                index = i * n_parallel + j * n_avg_c + k
                f = get_test_function(function_type=func_type, d=dim, c=c[index, :], w=w[index, :])
                functions.append(f)
                y[index, :] = f(test_grid)
                function_names.append(func_type.name)

    ls = LeastSquaresInterpolator(include_bias=True, basis_type=basis_type, grid=grid, method=method_type)
    ls.fit(functions)

    y_hat = ls.interpolate(test_grid)

    l_2_error = l2_error_function_values(y, y_hat).reshape(n_parallel * n_function_types * n_avg_c)
    max_error = max_error_function_values(y, y_hat).reshape(n_parallel * n_function_types * n_avg_c)

    end_time = time.time()
    needed_time = end_time - start_time

    cpu = platform.processor()
    cur_datetime = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    results = list()

    for i in range(n_parallel * n_function_types * n_avg_c):
        row_entry = dict()
        row_entry['dim'] = dim
        row_entry['method'] = 'Least_Squares'
        row_entry['w'] = w[i, :]
        row_entry['c'] = c[i, :]
        row_entry['sum_c'] = row_entry['c'].sum()
        row_entry['grid_type'] = grid.grid_type.name
        row_entry['basis_type'] = ls.basis_type.name
        row_entry['method_type'] = method_type.name
        row_entry['n_samples'] = int(multiplier_fun(n_samples))
        row_entry['scale'] = scale
        row_entry['seed'] = seed
        row_entry['test_grid_seed'] = test_grid_seed
        row_entry['f_name'] = function_names[i]
        row_entry['l_2_error'] = l_2_error[i]
        row_entry['max_error'] = max_error[i]
        row_entry['cpu'] = cpu
        row_entry['datetime'] = cur_datetime
        row_entry['needed_time'] = needed_time
        results.append(row_entry)

    if path is None:
        path = os.path.join("results", folder_name, "results_numerical_experiments.csv")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.isfile(path):
        data = pd.read_csv(path, sep=',', header=0)
    else:
        data = pd.DataFrame()

    new_data = pd.DataFrame(results)
    data = pd.concat([data, new_data], ignore_index=True)

    data['sum_c'] = data['sum_c'].apply(lambda x: round(x, 3))

    data.to_csv(path, sep=',', index=False)

    return grid


def run_experiments(function_types: list[FunctionType], n_functions_parallel: int, seed_realizations: list[int],
                    scales: range, dims: range,
                    methods: list, multiplier_fun: Callable, average_c: List[float],
                    ls_method: LeastSquaresMethod, smolyak_method: SmolyakMethod, folder_name: str):
    """
    Runs multiple experiments for least-squares with various parameter combinations
    :param function_types: Specifies the functions that should be tested
    :param n_functions_parallel: number of parallel functions per type that should be tested
    :param seed_realizations: Specifies the seeds that should be used for the experiments for sampling the training data
    :param scales: Specifies which scale range should be used for the experiments
    :param dims: Specifies which dimension range should be used for the experiments
    :param methods: Specifies which methods should be used for the experiments
    :param average_c: Specifies the average c value for the test functions
    :param multiplier_fun: Multiplies the number of samples for the least squares experiments
    :param ls_method: Specifies which method should be used to solve the Least Squares Problem
    :param smolyak_method: Specifies which method should be used to solve the Smolyak Problem
    :param folder_name: Specifies the folder name where the results should be stored
    """

    print(
        f"Starting experiments with cpu {platform.processor()} and "
        f"{psutil.virtual_memory().total / 1024 / 1024 / 1024} GB RAM")

    n_function_types = len(function_types)
    n_avg_c = len(average_c)

    lb = float(0.0)
    ub = float(1.0)

    n_iterations = len(scales) * len(dims) * len(methods) * len(seed_realizations)

    # sum_c = [float(9.0), float(7.25), float(1.85), float(7.03), float(20.4), float(4.3)]

    pbar = tqdm(total=n_iterations, desc="Running experiments")

    for dim in dims:
        w = np.random.uniform(low=0.0, high=1.0, size=(n_function_types * n_functions_parallel * n_avg_c, dim))
        c = np.random.uniform(low=0.0, high=1.0, size=(n_function_types * n_functions_parallel * n_avg_c, dim))

        c_row_sum = np.sum(c, axis=1)
        c = c / c_row_sum[:, np.newaxis] * dim

        avg_c_repeated = np.array(
            [average_c[i % len(average_c)] for i in range(n_function_types * n_functions_parallel * n_avg_c)])

        avg_c_repeated = avg_c_repeated[:, np.newaxis]

        c *= avg_c_repeated

        for seed_id, seed in enumerate(seed_realizations):

            smolyak_grid = None
            ls_chebyshev_grid = None
            ls_uniform_grid = None

            for scale in scales:

                n_samples = calculate_num_points(scale, dim)

                test_grid_seed = 42
                np.random.seed(test_grid_seed)
                gp = GridProvider(dimension=dim, multiplier_fun=multiplier_fun, lower_bound=lb, upper_bound=ub)
                test_grid = gp.generate(GridType.RANDOM_UNIFORM, scale)

                for method in methods:
                    cur_datetime = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
                    pbar.set_postfix(
                        {"Dim": dim, "Meth.": method, "Scale": scale, "n_samples": n_samples, "seed_id": seed_id,
                         "datetime": cur_datetime})

                    if method == 'Smolyak':
                        if seed_id == 0:
                            smolyak_grid = run_experiments_smolyak(dim=dim, w=w, c=c, f_types=function_types,
                                                                   seed_list=seed_realizations,
                                                                   n_parallel=n_functions_parallel, n_avg_c=n_avg_c,
                                                                   scale=scale, grid=smolyak_grid,
                                                                   test_grid_seed=test_grid_seed,
                                                                   test_grid=test_grid, lb=lb, ub=ub,
                                                                   method_type=smolyak_method, folder_name=folder_name,
                                                                   path=None)

                        else:
                            # reuse the results from the previous seed
                            pass
                    elif method == 'Least_Squares_Uniform':

                        ls_uniform_grid = run_experiments_least_squares(dim=dim, w=w, c=c,
                                                                        f_types=function_types,
                                                                        n_parallel=n_functions_parallel,
                                                                        n_avg_c=n_avg_c,
                                                                        scale=scale, seed=seed,
                                                                        multiplier_fun=multiplier_fun,
                                                                        grid=ls_uniform_grid,
                                                                        test_grid_seed=test_grid_seed,
                                                                        test_grid=test_grid, lb=lb,
                                                                        ub=ub,
                                                                        grid_type=GridType.RANDOM_UNIFORM,
                                                                        basis_type=BasisType.CHEBYSHEV,
                                                                        method_type=ls_method,
                                                                        folder_name=folder_name,
                                                                        sample_new=False, path=None)

                    elif method == 'Least_Squares_Chebyshev_Weight':

                        ls_chebyshev_grid = run_experiments_least_squares(dim=dim, w=w, c=c,
                                                                          f_types=function_types,
                                                                          n_parallel=n_functions_parallel,
                                                                          n_avg_c=n_avg_c,
                                                                          scale=scale, seed=seed,
                                                                          multiplier_fun=multiplier_fun,
                                                                          grid=ls_chebyshev_grid,
                                                                          test_grid_seed=test_grid_seed,
                                                                          test_grid=test_grid, lb=lb,
                                                                          ub=ub,
                                                                          grid_type=GridType.RANDOM_CHEBYSHEV,
                                                                          basis_type=BasisType.CHEBYSHEV,
                                                                          method_type=ls_method,
                                                                          folder_name=folder_name,
                                                                          sample_new=False, path=None)

                    else:
                        raise ValueError(
                            f"The method {method} is not supported. "
                            f"Please use 'Smolyak', 'Least_Squares_Chebyshev_Weight' "
                            f"or 'Least_Squares_Uniform'!")
                    pbar.update(1)

    pbar.close()
