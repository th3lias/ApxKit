import os
from typing import Callable, Union
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from genz.genz_functions import GenzFunctionType


def l2_error_function_values(y: np.ndarray, y_hat: np.ndarray) -> Union[np.float64, np.ndarray]:
    """
    Calculates the L_2 error estimate by comparing the true function values and the approximated function values by
    calculating the mean-squared absolute difference
    :param y: true function values
    :param y_hat: approximated function values
    :return: L_2 error estimate
    """

    if y_hat.ndim == 1:
        error = np.sqrt(np.mean(np.square(np.abs(y - y_hat)))).squeeze()
    else:
        error = np.sqrt(np.mean(np.square(np.abs(y - y_hat)), axis=1)).squeeze()
    return error


def max_error_function_values(y: np.ndarray, y_hat: np.ndarray) -> Union[np.float64, np.ndarray]:
    """
    Calculates the estimated max absolute value distance by comparing the true function values and the approximated
    function values by calculating the max absolute difference
    :param y: true function values
    :param y_hat: approximated function values
    :return: error estimate
    """
    if y_hat.ndim == 1:
        error = np.max(np.abs(y_hat - y)).squeeze()
    else:
        error = np.max(np.abs(y_hat - y), axis=1).squeeze()
    return error


def min_error_function_values(y: np.ndarray, y_hat: np.ndarray) -> Union[np.float64, np.ndarray]:
    """
    Calculates the estimated min absolute value distance by comparing the true function values and the approximated
    function values by calculating the min absolute difference
    :param y: true function values
    :param y_hat: approximated function values
    :return: error estimate
    """
    if y_hat.ndim == 1:
        error = np.min(np.abs(y_hat - y)).squeeze()
    else:
        error = np.min(np.abs(y_hat - y), axis=1).squeeze()
    return error


def l2_error(f: Callable, f_hat: Callable, grid: np.ndarray) -> np.float64:
    """
    Calculates the L_2 error estimate by comparing the true function and the approximation f_hat on a test grid by
    calculating the mean-squared absolute difference
    :param f: function that should be approximated
    :param f_hat: approximation of the function
    :param grid: grid where the approximation should be compared vs the original function
    :return: error estimate
    """

    y_hat = f_hat(grid)
    y = f(grid)

    return l2_error_function_values(y=y, y_hat=y_hat)


def max_abs_error(f: Callable, f_hat: Callable, grid: np.ndarray) -> np.float64:
    """
        Calculates the estimated max absolute value distance by comparing the true function and the approximation f_hat
        on a test grid by calculating the mean-squared absolute difference
        :param f: function that should be approximated
        :param f_hat: approximation of the function
        :param grid: grid where the approximation should be compared vs the original function
        :return: error estimate
        """

    y_hat = f_hat(grid)
    y = f(grid)

    return max_error_function_values(y=y, y_hat=y_hat)


def min_abs_error(f: Callable, f_hat: Callable, grid: np.ndarray) -> np.float64:
    """
        Calculates the estimated min absolute value distance by comparing the true function and the approximation f_hat
        on a test grid by calculating the mean-squared absolute difference
        :param f: function that should be approximated
        :param f_hat: approximation of the function
        :param grid: grid where the approximation should be compared vs the original function
        :return: error estimate
        """

    y_hat = f_hat(grid)
    y = f(grid)

    return min_error_function_values(y=y, y_hat=y_hat)


def visualize_point_grid_2d(points: np.ndarray, alpha: np.float64) -> None:
    """
    Visualizes a 2D point grid in a scatter plot
    :param points: array that contains the points. Needs to be of shape (n, 2)
    :param alpha: specifies the opacity of the points
    :return:
    """
    if np.shape(points)[1] != 2:
        raise ValueError("points must be a 2-dimensional array")

    x = points[:, 0]
    y = points[:, 1]

    plt.figure(figsize=(10, 10))
    plt.scatter(x, y, color='black', alpha=alpha)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

    plt.grid(True)
    plt.show()


def visualize_point_grid_3d(points: np.ndarray, alpha: np.float64) -> None:
    """
        Visualizes a 3D point grid in a scatter plot
        :param points: array that contains the points. Needs to be of shape (n, 3)
        :param alpha: specifies the opacity of the points
        :return:
        """
    if np.shape(points)[1] != 3:
        raise ValueError("points must be a 3-dimensional array")

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, color='black', alpha=alpha, marker='o')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')

    plt.grid(True)
    plt.show()


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(f"{method.__name__}, {args}, {kw}, {te - ts}")
        return result

    return timed


@timeit
def test_function_time(func: Callable, n: int, *args, **kwargs):
    """Function for testing performance of some PyFunc. Runs the function n+1 times."""
    for i in range(n):
        func(*args, **kwargs)
    return func(*args, **kwargs)


def _remove_almost_identical_rows(arr: np.ndarray, tol=1e-8):
    """
    This method is only reference for testing purposes. It should not be used in production.
    :param arr:
    :param tol:
    :return:
    """
    unique_rows = [arr[0]]
    for row in arr[1:]:
        if not any(np.allclose(row, unique_row, atol=tol) for unique_row in unique_rows):
            unique_rows.append(row)
    return np.array(unique_rows)


def _remove_duplicates_squared_memory(arr: np.ndarray, tol: np.float32 = np.float32(1e-8)):
    """
    This method is only reference for testing purposes. It should not be used in production.
    :param arr:
    :param tol:
    :return:
    """
    if arr.size == 0:
        return arr
    diffs = np.sqrt(((arr[:, np.newaxis] - arr[np.newaxis, :]) ** 2).sum(axis=2))
    close = diffs <= tol
    not_dominated = ~np.any(np.triu(close, k=1), axis=0)
    unique_rows = arr[not_dominated]
    return unique_rows


def _remove_duplicates_linear_memory_naive(arr: np.ndarray, tol: np.float32 = np.float32(1e-8)):
    """
    This method is only reference for testing purposes. It should not be used in production.
    :param arr:
    :param tol:
    :return:
    """
    if arr.size == 0:
        return arr

    unique_rows = []
    # Iterate over each row
    for row in arr:
        # Compute the distance from the current row to all unique rows
        if unique_rows:
            diffs = np.linalg.norm(np.array(unique_rows) - row, axis=1)
            # Check if there is any row in the unique_rows close to the current row
            if not np.any(diffs <= tol):
                unique_rows.append(row)
        else:
            unique_rows.append(row)

    return np.array(unique_rows)


def plot_error_vs_scale(results: dict, scale_range: range, name: str) -> None:
    """
    Plot the results of the experiments by plotting the errors vs the scale (proportional to number of samples)
    :param results: dictionary containing the results
    :param scale_range: range of scales
    :param name: name of the experiment
    :return:
    """
    fig, axs = plt.subplots(1, 3, figsize=(18, 6))

    smolyak_min_diff = [results['smolyak'][scale]['min_diff'] for scale in scale_range]
    least_squares_min_diff = [results['least_squares'][scale]['min_diff'] for scale in scale_range]

    smolyak_max_diff = [results['smolyak'][scale]['max_diff'] for scale in scale_range]
    least_squares_max_diff = [results['least_squares'][scale]['max_diff'] for scale in scale_range]

    smolyak_ell_2 = [results['smolyak'][scale]['ell_2'] for scale in scale_range]
    least_squares_ell_2 = [results['least_squares'][scale]['ell_2'] for scale in scale_range]

    axs[0].plot(scale_range, smolyak_max_diff, label='Smolyak')
    axs[0].plot(scale_range, least_squares_max_diff, label='Least Squares')
    axs[0].set_xticks(scale_range)
    axs[0].set_title('Max (Abs) Error')
    axs[0].set_xlabel('Scale')  # TODO: Change maybe
    axs[0].set_ylabel('Max Error')
    axs[0].set_yscale('log')
    axs[0].legend()

    axs[1].plot(scale_range, smolyak_min_diff, label='Smolyak')
    axs[1].plot(scale_range, least_squares_min_diff, label='Least Squares')
    axs[1].set_xticks(scale_range)
    axs[1].set_title('Min (Abs) Error')
    axs[1].set_xlabel('Scale')  # TODO: Change maybe
    axs[1].set_ylabel('Min Error')
    axs[1].set_yscale('log')
    axs[1].legend()

    axs[2].plot(scale_range, smolyak_ell_2, label='Smolyak')
    axs[2].plot(scale_range, least_squares_ell_2, label='Least Squares')
    axs[2].set_xticks(scale_range)
    axs[2].set_title('L2 Error')
    axs[2].set_xlabel('Scale')  # TODO: Change maybe
    axs[2].set_ylabel('L2 Error')
    axs[2].set_yscale('log')
    axs[2].legend()

    fig.suptitle(name)
    plt.tight_layout()
    plt.show()


def plot_from_result_file_ls(dim: np.int8, degree: np.int8, fun_type: GenzFunctionType, path: Union[str, None] = None):
    if path is None:
        path = os.path.join("..", "results", "results_least_squares.csv")

    data = pd.read_csv(path, sep=',', header=0)

    data = data[data['f_name'] == fun_type.name]
    data = data[data['dim'] == dim]
    data = data[data['degree'] == degree]

    data_c = data.groupby('c')

    n_samples = data_c['n_samples']
    l_2_error_min_error = data_c['l_2_error_min_error']
    max_error = data_c['max_error']
    needed_time = data_c['needed_time']


def plot_data(df):
    # Extract the relevant columns from the dataframe
    dim = df['dim']
    degree = df['degree']
    w = df['w']
    c = df['c']
    sum_c = df['sum_c']
    n_samples = df['n_samples']
    test_grid_seed = df['test_grid_seed']
    n_test_samples = df['n_test_samples']
    f_name = df['f_name']
    l_2_error_min_error = df['l_2_error_min_error']
    max_error = df['max_error']
    needed_time = df['needed_time']

    # Create a subplot with 3 rows and 1 column
    fig, axs = plt.subplots(nrows=3, ncols=1)

    # Plot the l_2 error as a function of the degree on the first row
    axs[0].plot(degree, l_2_error_min_error)
    axs[0].set_title('L_2 Error vs. Degree')
    axs[0].set_xlabel('Degree')
    axs[0].set_ylabel('L_2 Error')

    # Plot the max error as a function of the dimension on the second row
    axs[1].plot(dim, max_error)
    axs[1].set_title('Max Error vs. Dimension')
    axs[1].set_xlabel('Dimension')
    axs[1].set_ylabel('Max Error')

    # Plot the needed time as a function of the number of samples on the third row
    axs[2].plot(n_samples, needed_time)
    axs[2].set_title('Needed Time vs. Number of Samples')
    axs[2].set_xlabel('Number of Samples')
    axs[2].set_ylabel('Needed Time (s)')

    # Return the figure and axis handles
    return fig, axs


def plot_errors(dimension, function_type: GenzFunctionType, scales: range,
                path: Union[str, None] = None):
    """
    Plots errors vs. number of samples for a given dimension and function_name, grouped by 'c'.

    Parameters:
    - data: DataFrame containing the dataset
    - dimension: The dimension to filter on
    - function_name: The function name to filter on
    - errors: List of errors to plot (default: ['error1', 'error2', 'error3'])
    """

    # TODO: [Jakob] Rework Docstring

    if path is None:
        path = os.path.join("..", "results", "results_numerical_experiments.csv")

    data = pd.read_csv(path, sep=',', header=0)

    # Filter data based on dimension and function_name

    filtered_data = data[(data['dim'] == dimension) & (data['f_name'] == function_type.name)]

    filtered_data.drop(['user', 'cpu', 'datetime', 'needed_time', 'sum_c', 'f_name', 'test_grid_seed'],
                       axis=1, inplace=True)

    degrees = filtered_data['degree'].unique()

    smolyak_data = filtered_data[(filtered_data['degree']) == 0]

    least_squares_data = filtered_data[(filtered_data['degree']) != 0]

    smolyak_data = smolyak_data.sort_values(by='scale')
    least_squares_data = least_squares_data.sort_values(by='scale')

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18,6))

    titles = ['Min (Abs) Error', 'Max (Abs) Error', 'L2 Error']

    errors = ['l_2_error', 'min_error', 'max_error']

    for name, group in smolyak_data.groupby('c'):
        for i, error in enumerate(errors):
            for degree in degrees:
                if degree == 0:  # TODO: [Jakob] Make it more beautiful
                    label = 'Smolyak'
                    axs[i].plot(scales, smolyak_data[smolyak_data['c'] == name][error], label=label)
                else:
                    label = f'LS degr. {degree}'
                    ls_filtered = least_squares_data[least_squares_data['c'] == name]
                    ls_filtered = ls_filtered[least_squares_data['degree']==degree]
                    axs[i].plot(scales, ls_filtered[error], label=label)
            axs[i].set_xticks(scales)
            axs[i].set_title(titles[i])
            axs[i].set_xlabel('Scale/no points')
            axs[i].set_ylabel('Error')
            axs[i].set_yscale('log')
            axs[i].legend()

    fig.suptitle(function_type.name)  # TODO [Jakob] additionally include c,w, dimension
    plt.tight_layout()
    plt.show()


# def plot_average_errors(dimension, degree, function_type: GenzFunctionType,
#                         errors=['l_2_error', 'min_error', 'max_error'], path: Union[str, None] = None):
#     """
#     Plots average errors vs. number of samples for a given dimension and function_name.
#
#     Parameters:
#     - data: DataFrame containing the dataset
#     - dimension: The dimension to filter on
#     - function_name: The function name to filter on
#     - errors: List of errors to plot (default: ['error1', 'error2', 'error3'])
#     """
#
#     if path is None:
#         path = os.path.join("..", "results", "results_numerical_experiments.csv")
#
#     data = pd.read_csv(path, sep=',', header=0)
#
#     # Filter data based on dimension and function_name
#     filtered_data = data[
#         (data['dim'] == dimension) & (data['degree'] == degree) & (data['f_name'] == function_type.name)]
#
#     filtered_data.drop(['user', 'cpu', 'datetime', 'needed_time', 'w', 'c', 'sum_c', 'f_name', 'test_grid_seed'],
#                        axis=1, inplace=True)
#
#     # Group by n_samples and calculate mean of errors
#     mean_data = filtered_data.groupby('n_samples').mean().reset_index()
#
#     plt.figure(figsize=(10, 6))
#     for error in errors:
#         plt.plot(mean_data['n_samples'], mean_data[error], label=error)
#
#     plt.yscale('log')
#     plt.xlabel('Number of Samples')
#     plt.ylabel('Average Error')
#     plt.title(f'Average Errors vs. Number of Samples (dimension={dimension}, function={function_type.name})')
#     plt.legend()
#     plt.show()
