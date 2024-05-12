import os
import time

import numpy as np
import pandas as pd

from typing import Union, Callable, List

from tqdm import tqdm

from least_squares.least_squares import approximate_by_polynomial_with_least_squares
from utils.utils import max_error_function_values, min_error_function_values, l2_error_function_values

from genz.genz_functions import GenzFunctionType, get_genz_function
from grid.grid_provider import GridType, GridProvider

import platform

import datetime


def run_experiments_least_squares(dim: np.int8, degree: np.int8, w: np.ndarray, c: np.ndarray, n_parallel: np.int8,
                                  n_samples: np.int32, test_grid_seed: np.int8, n_test_samples: np.int16,
                                  lb: np.float16, up: np.float16,
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

    np.random.seed(test_grid_seed)

    grid = np.random.uniform(low=lb, high=up, size=(n_samples, dim))
    test_grid = np.random.uniform(low=lb, high=up, size=(n_test_samples, dim))

    # the following can be used as soon the Grids also support variable intervals

    # grid = GridProvider(dimension=dim).generate(GridType.RANDOM, scale=n_samples)
    # test_grid = GridProvider(dimension=dim, seed=test_grid_seed).generate(GridType.RANDOM, scale=n_test_samples)

    functions = list()
    function_names = list()
    y = np.empty(shape=(n_parallel * 6, n_test_samples), dtype=np.float64)

    for i, type in enumerate(GenzFunctionType):
        for j in range(n_parallel):
            index = i * 6 + j
            f = get_genz_function(function_type=type, d=dim, c=c[index, :], w=w[index, :])
            functions.append(f)
            y[index, :] = f(test_grid)
            function_names.append(type.name)

    f_hat = approximate_by_polynomial_with_least_squares(functions, degree=degree, include_bias=True,
                                                         self_implemented=True,
                                                         dim=dim, grid=grid)

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

    data = data._append(results, ignore_index=True)
    data.to_csv(path, sep=',', index=False)


def run_experiments():
    """
    Runs multiple experiments for least-squares with various parameter combinations
    """
    n_functions_per_type_parallel = np.int8(10)
    lb = np.float16(0.0)
    up = np.float16(1.0)
    test_grid_seed = np.int8(42)
    n_test_samples = np.int8(50)

    n_samples_list = [21, 221, 1581, 8801, 41265, 171425, 652065]

    sum_c = [np.float16(9.0), np.float16(7.25), np.float16(1.85), np.float16(7.03), np.float16(20.4), np.float16(4.3)]

    for dim in tqdm(range(10, 31), desc="Dimension"):
        for degree in tqdm(range(1, 4), desc="Degree", leave=False):
            for n_samples in tqdm(n_samples_list, desc="No_Samples", leave=False, dynamic_ncols=False):

                w = np.random.uniform(low=lb, high=up, size=(np.int8(6) * n_functions_per_type_parallel, dim))
                c = np.random.uniform(low=lb, high=up, size=(np.int8(6) * n_functions_per_type_parallel, dim))

                for i in range(6):
                    cur_slice = c[n_functions_per_type_parallel * i:n_functions_per_type_parallel * (i + 1), :]
                    cur_sum = cur_slice.sum(axis=1, keepdims=True)
                    factor = sum_c[i] / cur_sum
                    c[n_functions_per_type_parallel * i:n_functions_per_type_parallel * (i + 1), :] *= factor

                run_experiments_least_squares(
                    dim=np.int8(dim),
                    degree=np.int8(degree),
                    w=w,
                    c=c,
                    n_parallel=n_functions_per_type_parallel,
                    n_samples=np.int32(n_samples),
                    test_grid_seed=test_grid_seed,
                    n_test_samples=n_test_samples,
                    lb=lb,
                    up=up,
                    path=None)

if __name__ == '__main__':
    run_experiments()
