import os
import time

import numpy as np
import pandas as pd

from typing import Union

from tqdm import tqdm

from grid.grid_provider import GridProvider
from grid.grid_type import GridType
from interpolate.least_squares import approximate_by_polynomial_with_least_squares
from utils.utils import max_error_function_values, min_error_function_values, l2_error_function_values

from genz.genz_functions import GenzFunctionType, get_genz_function

import platform

import datetime


def run_experiments_least_squares(dim: int, degree: int, w: np.ndarray, c: np.ndarray, n_parallel: int,
                                  n_samples: int, test_grid_seed: int, n_test_samples: int,
                                  lb: float, up: float,
                                  path: Union[str, None] = None):
    """
    Runs an experiment (or multiple depending on passed parameters) and appends the results to a results file
    :param dim: dimension of the grid/function
    :param degree: maximum degree that least-squares approximation has (sum of all exponents)
    :param w: shift-parameter for the genz-functions, can be multidimensional if multiple functions are used
    :param c: parameter for the genz-functions, can be multidimensional if multiple functions are used
    :param n_parallel: number of parallel functions per type
    :param n_samples: number of samples used to fit the least-squares model
    :param test_grid_seed: seed used to generate test grid
    :param n_test_samples: number of samples used to assess the quality of the fit
    :param lb: lower bound of the interval
    :param up: upper bound of the interval
    :param path: path of the results file. If None, the default path is used
    """
    start_time = time.time()

    gp = GridProvider(dimension=dim, lower_bound=lb, upper_bound=up)
    grid = gp.generate(GridType.RANDOM, scale=n_samples)
    gp.set_seed(test_grid_seed)
    test_grid = gp.generate(GridType.RANDOM, scale=n_test_samples)

    functions = list()
    function_names = list()
    y = np.empty(shape=(n_parallel * 6, n_test_samples), dtype=np.float64)

    for i, fun_type in enumerate(GenzFunctionType):
        for j in range(n_parallel):
            index = i * 6 + j
            f = get_genz_function(function_type=fun_type, d=dim, c=c[index, :], w=w[index, :])
            functions.append(f)
            y[index, :] = f(test_grid)
            function_names.append(fun_type.name)

    f_hat = approximate_by_polynomial_with_least_squares(functions, degree=degree, include_bias=True,
                                                         self_implemented=True,
                                                         dim=dim, points=grid)

    y_hat = f_hat(test_grid)

    l_2_error = l2_error_function_values(y, y_hat)
    min_error = min_error_function_values(y, y_hat)
    max_error = max_error_function_values(y, y_hat)

    end_time = time.time()
    needed_time = end_time - start_time

    username = os.getlogin()
    cpu = platform.processor()
    cur_datetime = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    results = list()

    for i in range(n_parallel * 6):
        row_entry = dict()
        row_entry['dim'] = dim
        row_entry['degree'] = degree
        row_entry['w'] = w[i, :]
        row_entry['c'] = c[i, :]
        row_entry['sum_c'] = row_entry['c'].sum()
        row_entry['n_samples'] = n_samples
        row_entry['test_grid_seed'] = test_grid_seed
        row_entry['n_test_samples'] = n_test_samples
        row_entry['f_name'] = function_names[i]
        row_entry['l_2_error'] = l_2_error[i]
        row_entry['min_error'] = min_error[i]
        row_entry['max_error'] = max_error[i]
        row_entry['user'] = username
        row_entry['cpu'] = cpu
        row_entry['datetime'] = cur_datetime
        row_entry['needed_time'] = needed_time
        results.append(row_entry)

    if path is None:
        path = os.path.join("..", "results", "results_least_squares.csv")

    os.makedirs(os.path.dirname(path), exist_ok=True)

    if os.path.isfile(path):
        data = pd.read_csv(path, sep=',', header=0)
    else:
        data = pd.DataFrame()

    new_data = pd.DataFrame(results)
    data = pd.concat([data, new_data], ignore_index=True)
    data.to_csv(path, sep=',', index=False)


def run_experiments():
    """
    Runs multiple experiments for least-squares with various parameter combinations
    """
    n_functions_per_type_parallel = 10
    lb = 0.0
    up = 1.0
    test_grid_seed = 42
    n_test_samples = 50

    n_samples_list = [21, 221, 1581, 8801, 41265, 171425, 652065]

    sum_c = [9.0, 7.25, 1.85, 7.03, 20.4, 4.3]

    for dim in tqdm(range(10, 31), desc="Dimension"):
        for degree in tqdm(range(1, 4), desc="Degree", leave=False):
            for n_samples in tqdm(n_samples_list, desc="No_Samples", leave=False, dynamic_ncols=False):

                w = np.random.uniform(low=lb, high=up, size=(6 * n_functions_per_type_parallel, dim))
                c = np.random.uniform(low=lb, high=up, size=(6 * n_functions_per_type_parallel, dim))

                for i in range(6):
                    cur_slice = c[n_functions_per_type_parallel * i:n_functions_per_type_parallel * (i + 1), :]
                    cur_sum = cur_slice.sum(axis=1, keepdims=True)
                    factor = sum_c[i] / cur_sum
                    c[n_functions_per_type_parallel * i:n_functions_per_type_parallel * (i + 1), :] *= factor

                run_experiments_least_squares(
                    dim=dim,
                    degree=degree,
                    w=w,
                    c=c,
                    n_parallel=n_functions_per_type_parallel,
                    n_samples=int(n_samples),
                    test_grid_seed=test_grid_seed,
                    n_test_samples=n_test_samples,
                    lb=lb,
                    up=up,
                    path=None)


if __name__ == '__main__':
    run_experiments()
