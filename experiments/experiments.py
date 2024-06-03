import datetime
import os
import platform
import time
from typing import Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from genz.genz_functions import GenzFunctionType, get_genz_function
from grid.grid import Grid
from grid.grid_provider import GridType, GridProvider
from interpolate.basis_types import BasisType
from interpolate.least_squares import LeastSquaresInterpolator
from interpolate.least_squares_method import LeastSquaresMethod
from interpolate.smolyak import SmolyakInterpolator
from utils.utils import max_error_function_values, l2_error_function_values
from utils.utils import calculate_num_points

import psutil


def run_experiments_smolyak(dim: int, w: np.ndarray, c: np.ndarray,
                            n_parallel: int, scale: int, grid: Union[Grid, None], test_grid_seed: int,
                            n_test_samples: int, lb: float, ub: float, path: Union[str, None] = None) -> Grid:
    """
    Runs an experiment (or multiple depending on passed parameters) and appends the results to a results file
    :param dim: dimension of the grid/function
    :param w: shift-parameter for the genz-functions, can be multidimensional if multiple functions are used
    :param c: parameter for the genz-functions, can be multidimensional if multiple functions are used
    :param n_parallel: number of parallel functions per type
    :param scale: related to the number of samples used to fit the smolyak model
    :param grid: grid on which the Smolyak Algorithm should operate. If None, a new grid will be created
    :param test_grid_seed: seed used to generate test grid
    :param n_test_samples: number of samples used to assess the quality of the fit
    :param lb: lower bound of the interval
    :param ub: upper bound of the interval
    :param path: path of the results file. If None, the default path is used

    :return: The created grid, such that it can be used again for an increased scale
    """

    start_time = time.time()

    np.random.seed(test_grid_seed)
    test_grid = np.random.uniform(low=lb, high=ub, size=(n_test_samples, dim))

    n_function_types = int(len(GenzFunctionType))

    n_samples = calculate_num_points(scale, dim)

    gp = GridProvider(dimension=dim, lower_bound=lb, upper_bound=ub)

    if grid is None or not grid.dim == dim:
        grid = gp.generate(GridType.CHEBYSHEV, scale=scale)
    else:
        grid = gp.increase_scale(grid)

    functions = list()
    function_names = list()
    y = np.empty(shape=(n_parallel * n_function_types, n_test_samples), dtype=np.float64)

    for i, func_type in enumerate(GenzFunctionType):
        for j in range(n_parallel):
            index = i * n_parallel + j
            f = get_genz_function(function_type=func_type, d=dim, c=c[index, :], w=w[index, :])
            functions.append(f)
            y[index, :] = f(test_grid)
            function_names.append(func_type.name)

    si = SmolyakInterpolator(grid)
    f_hat = si.interpolate(functions)

    y_hat = f_hat(test_grid)

    l_2_error = l2_error_function_values(y, y_hat)
    max_error = max_error_function_values(y, y_hat)

    end_time = time.time()
    needed_time = end_time - start_time

    cpu = platform.processor()
    cur_datetime = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    results = list()

    for i in range(n_parallel * n_function_types):
        row_entry = dict()
        row_entry['dim'] = dim
        row_entry['method'] = 'Smolyak'
        row_entry['w'] = w[i, :]
        row_entry['c'] = c[i, :]
        row_entry['sum_c'] = row_entry['c'].sum()
        row_entry['grid_type'] = si.grid.grid_type.name
        row_entry['basis_type'] = si.basis_type.name
        row_entry['method'] = 'STANDARD'  # TODO: Later change to Lagrange interpolation if implemented
        row_entry['n_samples'] = n_samples
        row_entry['scale'] = scale
        row_entry['test_grid_seed'] = test_grid_seed
        row_entry['n_test_samples'] = n_test_samples
        row_entry['f_name'] = function_names[i]
        row_entry['l_2_error'] = l_2_error[i]
        row_entry['max_error'] = max_error[i]
        row_entry['cpu'] = cpu
        row_entry['datetime'] = cur_datetime
        row_entry['needed_time'] = needed_time
        results.append(row_entry)

    if path is None:
        path = os.path.join("results", "results_numerical_experiments.csv")

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


def run_experiments_least_squares(dim: int, w: np.ndarray, c: np.ndarray,
                                  n_parallel: int, scale: int, additional_multiplier: float, grid: Union[Grid, None],
                                  test_grid_seed: int, n_test_samples: int, lb: float, ub: float,
                                  grid_type: GridType, basis_type: BasisType, method_type: LeastSquaresMethod,
                                  sample_new: bool = True, path: Union[str, None] = None) -> Grid:
    """
    Runs an experiment (or multiple depending on passed parameters) and appends the results to a results file
    :param dim: dimension of the grid/function
    :param w: shift-parameter for the genz-functions, can be multidimensional if multiple functions are used
    :param c: parameter for the genz-functions, can be multidimensional if multiple functions are used
    :param n_parallel: number of parallel functions per type
    :param scale: related to the number of samples used to fit the least-squares model
    :param additional_multiplier: Multiplies the number of samples of least squares by this factor
    :param grid: Grid on which least-squares should be fitted. If None, a new grid is created
    :param test_grid_seed: seed used to generate test grid
    :param n_test_samples: number of samples used to assess the quality of the fit
    :param lb: lower bound of the interval
    :param ub: upper bound of the interval
    :param grid_type: Specifies which grid should be used. Usually sampled from uniform or chebyshev weight
    :param basis_type: Specifies the type of the polynomial basis. If Chebyshev, then the exact same basis is used like
    in the Smolyak algorithm, otherwise a comparable standard basis
    :param method_type: Specifies which type of solving algorithm should be used
    :param sample_new: Specifies, whether the current points in the grid should be kept or newly sampled
    :param path: path of the results file. If None, the default path is used

    :return: The created grid, such that it can be used again for an increased scale
    """

    start_time = time.time()

    np.random.seed(test_grid_seed)

    n_samples = calculate_num_points(scale, dim)

    multiplier = np.log(n_samples) * additional_multiplier

    test_grid = np.random.uniform(low=lb, high=ub, size=(n_test_samples, dim))

    n_function_types = int(len(GenzFunctionType))

    gp = GridProvider(dimension=dim, lower_bound=lb, upper_bound=ub)

    if grid is None or not grid.dim == dim:
        grid = gp.generate(grid_type, scale=scale, multiplier=multiplier)
    else:
        grid = gp.increase_scale(grid, sample_new)

    functions = list()
    function_names = list()
    y = np.empty(shape=(n_parallel * n_function_types, n_test_samples), dtype=np.float64)

    for i, func_type in enumerate(GenzFunctionType):
        for j in range(n_parallel):
            index = i * n_parallel + j
            f = get_genz_function(function_type=func_type, d=dim, c=c[index, :], w=w[index, :])
            functions.append(f)
            y[index, :] = f(test_grid)
            function_names.append(func_type.name)

    ls = LeastSquaresInterpolator(include_bias=True, basis_type=basis_type, grid=grid, method=method_type)
    f_hat = ls.interpolate(functions)

    y_hat = f_hat(test_grid)

    l_2_error = l2_error_function_values(y, y_hat)
    max_error = max_error_function_values(y, y_hat)

    end_time = time.time()
    needed_time = end_time - start_time

    cpu = platform.processor()
    cur_datetime = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    results = list()

    for i in range(n_parallel * n_function_types):
        row_entry = dict()
        row_entry['dim'] = dim
        row_entry['method'] = 'Least_Squares'
        row_entry['w'] = w[i, :]
        row_entry['c'] = c[i, :]
        row_entry['sum_c'] = row_entry['c'].sum()
        row_entry['grid_type'] = grid.grid_type.name
        row_entry['basis_type'] = ls.basis_type.name
        row_entry['method'] = method_type.name
        row_entry['n_samples'] = int(n_samples * multiplier)
        row_entry['scale'] = scale
        row_entry['test_grid_seed'] = test_grid_seed
        row_entry['n_test_samples'] = n_test_samples
        row_entry['f_name'] = function_names[i]
        row_entry['l_2_error'] = l_2_error[i]
        row_entry['max_error'] = max_error[i]
        row_entry['cpu'] = cpu
        row_entry['datetime'] = cur_datetime
        row_entry['needed_time'] = needed_time
        results.append(row_entry)

    if path is None:
        path = os.path.join("results", "results_numerical_experiments.csv")

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


def run_experiments(n_functions_parallel: int, scales: range, dims: range, methods: list, add_mul: float, ls_method:LeastSquaresMethod):
    """
    Runs multiple experiments for least-squares with various parameter combinations
    :param n_functions_parallel: number of parallel functions per type that should be tested
    :param scales: Specifies which scale range should be used for the experiments
    :param dims: Specifies which dimension range should be used for the experiments
    :param methods: Specifies which methods should be used for the experiments
    :param add_mul: Multiplies the number of samples for the least squares experiments
    :param ls_method: Specifies which method should be used to solve Least Squares Problem
    """

    print(
        f"Starting experiments with cpu {platform.processor()} and "
        f"{psutil.virtual_memory().total / 1024 / 1024 / 1024} GB RAM")

    n_function_types = len(GenzFunctionType)

    lb = float(0.0)
    ub = float(1.0)
    test_grid_seed = 42
    n_test_samples = 50

    n_iterations = len(scales) * len(dims) * len(methods)

    # sum_c = [float(9.0), float(7.25), float(1.85), float(7.03), float(20.4), float(4.3)]

    pbar = tqdm(total=n_iterations, desc="Running experiments")

    smolyak_grid = None
    least_squares_chebyshev_grid = None
    least_squares_uniform_grid = None

    for dim in dims:
        w = np.random.uniform(low=0.0, high=1.0, size=(n_function_types * n_functions_parallel, dim))
        c = np.random.uniform(low=0.0, high=1.0, size=(n_function_types * n_functions_parallel, dim))

        for scale in scales:

            n_samples = calculate_num_points(scale, dim)

            for method in methods:

                pbar.set_postfix({"Dimension": dim, "Method": method, "Scale": scale, "n_samples": n_samples})

                for i in range(n_function_types):
                    cur_slice = c[n_functions_parallel * i:n_functions_parallel * (i + 1), :]
                    cur_sum = cur_slice.sum(axis=1, keepdims=True)
                    # factor = sum_c[i] / cur_sum
                    factor = dim / cur_sum
                    c[n_functions_parallel * i:n_functions_parallel * (i + 1), :] *= factor

                if method == 'Smolyak':

                    smolyak_grid = run_experiments_smolyak(dim=dim, w=w, c=c, n_parallel=n_functions_parallel,
                                                           scale=scale, grid=smolyak_grid,
                                                           test_grid_seed=test_grid_seed,
                                                           n_test_samples=n_test_samples, lb=lb, ub=ub, path=None)
                elif method == 'Least_Squares_Uniform':

                    least_squares_uniform_grid = run_experiments_least_squares(dim=dim, w=w, c=c,
                                                                               n_parallel=n_functions_parallel,
                                                                               scale=scale,
                                                                               additional_multiplier=add_mul,
                                                                               grid=least_squares_uniform_grid,
                                                                               test_grid_seed=test_grid_seed,
                                                                               n_test_samples=n_test_samples, lb=lb,
                                                                               ub=ub, grid_type=GridType.RANDOM_UNIFORM,
                                                                               basis_type=BasisType.CHEBYSHEV,
                                                                               method_type=ls_method,
                                                                               sample_new=False, path=None)

                elif method == 'Least_Squares_Chebyshev_Weight':

                    least_squares_chebyshev_grid = run_experiments_least_squares(dim=dim, w=w, c=c,
                                                                                 n_parallel=n_functions_parallel,
                                                                                 scale=scale,
                                                                                 additional_multiplier=add_mul,
                                                                                 grid=least_squares_chebyshev_grid,
                                                                                 test_grid_seed=test_grid_seed,
                                                                                 n_test_samples=n_test_samples, lb=lb,
                                                                                 ub=ub,
                                                                                 grid_type=GridType.RANDOM_CHEBYSHEV,
                                                                                 basis_type=BasisType.CHEBYSHEV,
                                                                                 method_type=ls_method,
                                                                                 sample_new=False, path=None)

                else:
                    raise ValueError(
                        f"The method {method} is not supported. Please use 'Smolyak', 'Least_Squares_Chebyshev_Weight' "
                        f"or 'Least_Squares_Uniform'!")
                pbar.update(1)

    pbar.close()
