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
from interpolate.least_squares import approximate_by_polynomial_with_least_squares
from interpolate.smolyak import SmolyakInterpolator
from utils.utils import max_error_function_values, l2_error_function_values
from utils.utils import plot_errors


def get_no_samples(scale: int, dim: int = 10):
    # only holds for dim=10
    n_samples_list = [21, 221, 1581, 8801, 41265, 171425, 652065]  # TODO: [Jakob] Make valid function here
    return int(n_samples_list[scale - 1])


def run_experiments_smolyak(dim: int, w: np.ndarray, c: np.ndarray, scale: int, test_grid_seed: int,
                            n_test_samples: int, lb: float, ub: float, path: Union[str, None] = None):
    """
    Runs an experiment (or multiple depending on passed parameters) and appends the results to a results file
    :param dim: dimension of the grid/function
    :param w: shift-parameter for the genz-functions, can be multidimensional if multiple functions are used
    :param c: parameter for the genz-functions, can be multidimensional if multiple functions are used
    :param scale: related to the number of samples used to fit the smolyak model
    :param test_grid_seed: seed used to generate test grid
    :param n_test_samples: number of samples used to assess the quality of the fit
    :param lb: lower bound of the interval
    :param ub: upper bound of the interval
    :param path: path of the results file. If None, the default path is used
    """

    np.random.seed(test_grid_seed)

    test_grid = GridProvider(dimension=dim, lower_bound=lb, upper_bound=ub).generate(GridType.RANDOM,
                                                                                     scale=n_test_samples)

    n_function_types = int(len(GenzFunctionType))

    n_samples = get_no_samples(scale)

    si = SmolyakInterpolator(dimension=dim, scale=scale)

    function_names = list()

    ell_2_error_list = list()
    max_error_list = list()
    needed_time_list = list()

    for i, fun_type in enumerate(GenzFunctionType):
        start_time = time.time()
        f = get_genz_function(function_type=fun_type, d=dim, c=c[i], w=w[i])
        y = f(test_grid)
        function_names.append(fun_type.name)
        f_hat = si.interpolate(f)
        y_hat = f_hat(test_grid)

        ell_2_error_list.append(l2_error_function_values(y, y_hat))
        max_error_list.append(max_error_function_values(y, y_hat))

        end_time = time.time()
        needed_time = end_time - start_time
        needed_time_list.append(needed_time)

    username = os.getlogin()
    cpu = platform.processor()
    cur_datetime = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    results = list()

    for i in range(n_function_types):
        row_entry = dict()
        row_entry['dim'] = dim
        row_entry['degree'] = 0
        row_entry['w'] = w[i, :]
        row_entry['c'] = c[i, :]
        row_entry['sum_c'] = row_entry['c'].sum()
        row_entry['n_samples'] = n_samples
        row_entry['scale'] = scale
        row_entry['test_grid_seed'] = test_grid_seed
        row_entry['n_test_samples'] = n_test_samples
        row_entry['f_name'] = function_names[i]
        row_entry['l_2_error'] = ell_2_error_list[i]
        row_entry['max_error'] = max_error_list[i]
        row_entry['user'] = username
        row_entry['cpu'] = cpu
        row_entry['datetime'] = cur_datetime
        row_entry['needed_time'] = needed_time_list[i]
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


def run_experiments_least_squares(dim: int, degree: int, w: np.ndarray, c: np.ndarray, n_parallel: int, scale: int,
                                  test_grid_seed: int, n_test_samples: int, lb: float, ub: float,
                                  path: Union[str, None] = None):
    """
    Runs an experiment (or multiple depending on passed parameters) and appends the results to a results file
    :param dim: dimension of the grid/function
    :param degree: maximum degree that least-squares approximation has (sum of all exponents)
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

    n_samples = get_no_samples(scale)

    grid = GridProvider(dimension=dim, lower_bound=lb, upper_bound=ub).generate(GridType.RANDOM, scale=n_samples)
    test_grid = GridProvider(dimension=dim, lower_bound=lb, upper_bound=ub).generate(GridType.RANDOM,
                                                                                     scale=n_test_samples)

    n_function_types = int(6)

    functions = list()
    function_names = list()
    y = np.empty(shape=(n_parallel * n_function_types, n_test_samples), dtype=np.float64)

    for i, fun_type in enumerate(GenzFunctionType):
        # TODO: [Jakob] Maybe possible to vectorize
        for j in range(n_parallel):
            index = i * n_function_types + j
            f = get_genz_function(function_type=fun_type, d=dim, c=c[index, :], w=w[index, :])
            functions.append(f)
            y[index, :] = f(test_grid)
            function_names.append(fun_type.name)

    f_hat = approximate_by_polynomial_with_least_squares(functions, degree=degree, include_bias=True,
                                                         self_implemented=True, dim=dim, points=grid)

    y_hat = f_hat(test_grid)

    l_2_error = l2_error_function_values(y, y_hat)
    max_error = max_error_function_values(y, y_hat)

    end_time = time.time()
    needed_time = end_time - start_time

    username = os.getlogin()
    cpu = platform.processor()
    cur_datetime = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")

    results = list()

    for i in range(n_parallel * n_function_types):
        row_entry = dict()
        row_entry['dim'] = dim
        row_entry['degree'] = degree
        row_entry['w'] = w[i, :]
        row_entry['c'] = c[i, :]
        row_entry['sum_c'] = row_entry['c'].sum()
        row_entry['n_samples'] = n_samples
        row_entry['scale'] = scale
        row_entry['test_grid_seed'] = test_grid_seed
        row_entry['n_test_samples'] = n_test_samples
        row_entry['f_name'] = function_names[i]
        row_entry['l_2_error'] = l_2_error[i]
        row_entry['max_error'] = max_error[i]
        row_entry['user'] = username
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
    n_functions_per_type_parallel = int(25)
    n_function_types = int(6)

    lb = float(0.0)
    ub = float(1.0)
    test_grid_seed = 42
    n_test_samples = 50

    scale_range = range(1, 8)
    dim_range = range(10, 31)
    degree_range = range(1, 4)  # degree 0 means Smolyak

    n_iterations = len(scale_range) * len(dim_range) * len(degree_range)

    sum_c = [float(9.0), float(7.25), float(1.85), float(7.03), float(20.4), float(4.3)]

    pbar = tqdm(total=n_iterations, desc="Running experiments")

    for dim in dim_range:
        w = np.random.uniform(low=lb, high=ub, size=(n_function_types * n_functions_per_type_parallel, dim))
        c = np.random.uniform(low=lb, high=ub, size=(n_function_types * n_functions_per_type_parallel, dim))

        for degree in degree_range:
            for scale in scale_range:

                n_samples = get_no_samples(scale)

                pbar.set_postfix({"Dimension": dim, "Degree": degree, "Scale": scale, "n_samples": n_samples})

                for i in range(n_function_types):
                    cur_slice = c[n_functions_per_type_parallel * i:n_functions_per_type_parallel * (i + 1), :]
                    cur_sum = cur_slice.sum(axis=1, keepdims=True)
                    factor = sum_c[i] / cur_sum
                    c[n_functions_per_type_parallel * i:n_functions_per_type_parallel * (i + 1), :] *= factor

                if degree == 0:

                    w_smol = w[0::n_functions_per_type_parallel, :]
                    c_smol = c[0::n_functions_per_type_parallel, :]

                    run_experiments_smolyak(dim=dim, w=w_smol, c=c_smol, scale=scale, test_grid_seed=test_grid_seed,
                        n_test_samples=n_test_samples, lb=lb, ub=ub, path=None)
                else:
                    run_experiments_least_squares(dim=dim, degree=degree, w=w, c=c,
                        n_parallel=n_functions_per_type_parallel, scale=scale, test_grid_seed=test_grid_seed,
                        n_test_samples=n_test_samples, lb=lb, ub=ub, path=None)

                pbar.update(1)

    pbar.close()


if __name__ == '__main__':
    # run_experiments()
    plot_errors(10, GenzFunctionType.OSCILLATORY, range(1, 8))
