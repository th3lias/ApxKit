import datetime
import os
import platform
import time
from typing import Union

import numpy as np
import pandas as pd
from tqdm import tqdm

from genz.genz_functions import GenzFunctionType, get_genz_function
from grid.grid_provider import GridType, GridProvider
from interpolate.basis_types import BasisType
from interpolate.least_squares import LeastSquaresInterpolator
from interpolate.smolyak import SmolyakInterpolator
from utils.utils import max_error_function_values, l2_error_function_values
from utils.utils import calculate_num_points, plot_errors


def run_experiments_smolyak(dim: int, w: np.ndarray, c: np.ndarray,
                            n_parallel: int, scale: int, test_grid_seed: int,
                            n_test_samples: int, lb: float, ub: float, path: Union[str, None] = None):
    """
    Runs an experiment (or multiple depending on passed parameters) and appends the results to a results file
    :param dim: dimension of the grid/function
    :param w: shift-parameter for the genz-functions, can be multidimensional if multiple functions are used
    :param c: parameter for the genz-functions, can be multidimensional if multiple functions are used
    :param n_parallel: number of parallel functions per type
    :param scale: related to the number of samples used to fit the smolyak model
    :param test_grid_seed: seed used to generate test grid
    :param n_test_samples: number of samples used to assess the quality of the fit
    :param lb: lower bound of the interval
    :param ub: upper bound of the interval
    :param path: path of the results file. If None, the default path is used
    """

    start_time = time.time()

    np.random.seed(test_grid_seed)

    test_grid = np.random.uniform(low=lb, high=ub, size=(n_test_samples, dim))

    n_function_types = int(len(GenzFunctionType))

    n_samples = calculate_num_points(scale, dim)

    functions = list()
    function_names = list()
    y = np.empty(shape=(n_parallel * n_function_types, n_test_samples), dtype=np.float64)

    for i, func_type in enumerate(GenzFunctionType):
        # TODO: [Jakob] Maybe possible to vectorize
        for j in range(n_parallel):
            index = i * n_function_types + j
            f = get_genz_function(function_type=func_type, d=dim, c=c[index, :], w=w[i, :])
            functions.append(f)
            y[index, :] = f(test_grid)
            function_names.append(func_type.name)

    si = SmolyakInterpolator(dimension=dim, scale=scale, lb=lb, ub=ub)
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
        row_entry['w'] = w[int(i // n_parallel), :]
        row_entry['c'] = c[i, :]
        row_entry['sum_c'] = row_entry['c'].sum()
        row_entry['grid_type'] = si.grid.grid_type.name
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
        path = os.path.join("..", "results", "results_numerical_experiments.csv")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.isfile(path):
        data = pd.read_csv(path, sep=',', header=0)
    else:
        data = pd.DataFrame()

    new_data = pd.DataFrame(results)
    data = pd.concat([data, new_data], ignore_index=True)

    data['sum_c'] = data['sum_c'].apply(lambda x: round(x, 3))

    data.to_csv(path, sep=',', index=False)


def run_experiments_least_squares(dim: int, w: np.ndarray, c: np.ndarray,
                                  n_parallel: int, scale: int,
                                  test_grid_seed: int, n_test_samples: int, lb: float, ub: float,
                                  path: Union[str, None] = None):
    """
    Runs an experiment (or multiple depending on passed parameters) and appends the results to a results file
    :param dim: dimension of the grid/function
    :param w: shift-parameter for the genz-functions, can be multidimensional if multiple functions are used
    :param c: parameter for the genz-functions, can be multidimensional if multiple functions are used
    :param n_parallel: number of parallel functions per type
    :param scale: related to the number of samples used to fit the least-squares model
    :param test_grid_seed: seed used to generate test grid
    :param n_test_samples: number of samples used to assess the quality of the fit
    :param lb: lower bound of the interval
    :param ub: upper bound of the interval
    :param path: path of the results file. If None, the default path is used
    """

    start_time = time.time()

    np.random.seed(test_grid_seed)

    n_samples = calculate_num_points(scale, dim)

    multiplier = np.log10(n_samples)

    test_grid = np.random.uniform(low=lb, high=ub, size=(n_test_samples, dim))

    n_function_types = int(6)

    gp = GridProvider(dimension=dim, lower_bound=lb, upper_bound=ub)

    grid = gp.generate(GridType.RANDOM_CHEBYSHEV, scale=scale, multiplier=multiplier)

    functions = list()
    function_names = list()
    y = np.empty(shape=(n_parallel * n_function_types, n_test_samples), dtype=np.float64)

    for i, func_type in enumerate(GenzFunctionType):
        # TODO: [Jakob] Maybe possible to vectorize
        for j in range(n_parallel):
            index = i * n_function_types + j
            f = get_genz_function(function_type=func_type, d=dim, c=c[index, :], w=w[i, :])
            functions.append(f)
            y[index, :] = f(test_grid)
            function_names.append(func_type.name)

    ls = LeastSquaresInterpolator(include_bias=True, basis_type=BasisType.CHEBYSHEV, grid=grid)
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
        row_entry['w'] = w[int(i // n_parallel), :]
        row_entry['c'] = c[i, :]
        row_entry['sum_c'] = row_entry['c'].sum()
        row_entry['grid_type'] = grid.grid_type.name
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
        path = os.path.join("..", "results", "results_numerical_experiments.csv")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.isfile(path):
        data = pd.read_csv(path, sep=',', header=0)
    else:
        data = pd.DataFrame()

    new_data = pd.DataFrame(results)
    data = pd.concat([data, new_data], ignore_index=True)

    data['sum_c'] = data['sum_c'].apply(lambda x: round(x, 3))

    data.to_csv(path, sep=',', index=False)


def run_experiments():
    """
    Runs multiple experiments for least-squares with various parameter combinations
    """
    n_functions_per_type_parallel = int(50)
    n_function_types = len(GenzFunctionType)

    lb = float(0.0)
    ub = float(1.0)
    test_grid_seed = 42
    n_test_samples = 50

    scale_range = range(1, 5)
    dim_range = range(1, 2)
    methods = ['Smolyak', 'Least_Squares']

    n_iterations = len(scale_range) * len(dim_range) * len(methods)

    sum_c = [float(9.0), float(7.25), float(1.85), float(7.03), float(20.4), float(4.3)]

    pbar = tqdm(total=n_iterations, desc="Running experiments")

    for dim in dim_range:
        w = np.random.uniform(low=lb, high=ub, size=(n_function_types, dim))
        c = np.random.uniform(low=lb, high=ub, size=(n_function_types * n_functions_per_type_parallel, dim))

        for scale in scale_range:

            n_samples = calculate_num_points(scale, dim)

            for method in methods:

                pbar.set_postfix({"Dimension": dim, "Method": method, "Scale": scale, "n_samples": n_samples})

                for i in range(n_function_types):
                    cur_slice = c[n_functions_per_type_parallel * i:n_functions_per_type_parallel * (i + 1), :]
                    cur_sum = cur_slice.sum(axis=1, keepdims=True)
                    factor = sum_c[i] / cur_sum
                    c[n_functions_per_type_parallel * i:n_functions_per_type_parallel * (i + 1), :] *= factor

                if method == 'Smolyak':

                    # w_smol = w[0::n_functions_per_type_parallel, :]
                    # c_smol = c[0::n_functions_per_type_parallel, :]

                    run_experiments_smolyak(dim=dim, w=w, c=c, n_parallel=n_functions_per_type_parallel,
                                            scale=scale, test_grid_seed=test_grid_seed,
                                            n_test_samples=n_test_samples, lb=lb, ub=ub, path=None)
                elif method == 'Least_Squares':

                    run_experiments_least_squares(dim=dim, w=w, c=c,
                                                  n_parallel=n_functions_per_type_parallel, scale=scale,
                                                  test_grid_seed=test_grid_seed, n_test_samples=n_test_samples, lb=lb,
                                                  ub=ub, path=None)

                else:
                    raise ValueError(f"The method {method} is not supported. Please use 'Smolyak' or 'Least_Squares!")
                pbar.update(1)

    pbar.close()


if __name__ == '__main__':
    run_experiments()

    # visualize one specific instance
    # plot_errors(10, GenzFunctionType.OSCILLATORY, range(1, 5), save=True)

    # save all images in results folder
    # for dim in range(10, 11):
    #     for fun_type in GenzFunctionType:
    #         plot_errors(dim, fun_type, range(1, 5), save=True)
