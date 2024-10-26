from __future__ import annotations

import math
import os
import time
from typing import Callable, Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from function.type import FunctionType
from grid.grid.grid import Grid


def l2_error_function_values(y: np.ndarray, y_hat: np.ndarray) -> Union[float, np.ndarray]:
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


def max_error_function_values(y: np.ndarray, y_hat: np.ndarray) -> Union[float, np.ndarray]:
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


def l2_error(f: Callable, f_hat: Callable, grid: np.ndarray) -> float:
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


def max_abs_error(f: Callable, f_hat: Callable, grid: np.ndarray) -> float:
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


def visualize_point_grid_1d(points: Union[Grid, np.ndarray], alpha: float) -> None:
    """
    Visualizes a set of points in a histogram
    :param points: array that contains the points.
    :param alpha: specifies the opacity of the points
    :return: None
    """

    if isinstance(points, Grid):
        points = points.grid

    if len(points.shape) == 1:
        # 1D points
        plt.figure(figsize=(10, 6))
        plt.hist(points, bins=30, color='black', alpha=alpha)
        plt.xlabel('$x$')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
    else:
        raise ValueError(f"Wrong dimension of the data. Expected dimension 1, got {points.ndim}")


def visualize_point_grid_2d(points: Union[Grid, np.ndarray], alpha: float) -> None:
    """
    Visualizes a 2D point grid in a scatter plot
    :param points: array that contains the points. Needs to be of shape (n, 2)
    :param alpha: specifies the opacity of the points
    :return:
    """
    if isinstance(points, Grid):
        points = points.grid
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


def visualize_point_grid_3d(points: Union[Grid, np.ndarray], alpha: float) -> None:
    """
        Visualizes a 3D point grid in a scatter plot
        :param points: array that contains the points. Needs to be of shape (n, 3)
        :param alpha: specifies the opacity of the points
        :return:
        """

    if isinstance(points, Grid):
        points = points.grid
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


def plot_error_vs_scale(results: dict, scale_range: range, name: str) -> None:
    """
    Plot the results of the experiments by plotting the errors vs the scale (proportional to number of samples)
    :param results: dictionary containing the results
    :param scale_range: range of scales
    :param name: name of the experiment
    :return:
    """

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    smolyak_max_diff = [results['smolyak'][scale]['max_diff'] for scale in scale_range]
    least_squares_max_diff = [results['least_squares'][scale]['max_diff'] for scale in scale_range]

    smolyak_ell_2 = [results['smolyak'][scale]['ell_2'] for scale in scale_range]
    least_squares_ell_2 = [results['least_squares'][scale]['ell_2'] for scale in scale_range]

    axs[0].plot(scale_range, smolyak_max_diff, label='Smolyak')
    axs[0].plot(scale_range, least_squares_max_diff, label='Least Squares')
    axs[0].set_xticks(scale_range)
    axs[0].set_title('Max (Abs) Error')
    axs[0].set_xlabel('Scale')
    axs[0].set_ylabel('Max Error')
    axs[0].set_yscale('log')
    axs[0].legend()

    axs[1].plot(scale_range, smolyak_ell_2, label='Smolyak')
    axs[1].plot(scale_range, least_squares_ell_2, label='Least Squares')
    axs[1].set_xticks(scale_range)
    axs[1].set_title('L2 Error')
    axs[1].set_xlabel('Scale')
    axs[1].set_ylabel('L2 Error')
    axs[1].set_yscale('log')
    axs[1].legend()

    fig.suptitle(name)
    plt.tight_layout()
    plt.show()


def sample(dim: int | tuple[int], low: float = 0., high: float = 1.):
    return np.random.uniform(low=low, high=high, size=dim)


def get_next_filename(path, extension='png'):
    """ Function to get the next available filename """
    files = [f for f in os.listdir(path) if f.endswith('.' + extension)]
    numbers = [int(os.path.splitext(f)[0]) for f in files if f.split('.')[0].isdigit()]
    next_number = max(numbers, default=0) + 1
    return f"{next_number}.{extension}"


def plot_errors(dimension, seed: int, function_type: FunctionType, scales: range, multiplier_fun: Callable,
                folder_name: str, path: Union[str, None] = None, save: bool = False,
                save_path: Union[str, None] = None, same_axis_both_plots: bool = True):
    """
    Creates plots of each different c-value for a given function type, given the path of the results-csv file.
    The ell2 and the max error are plotted.

    :param dimension: dimension which should be considered from the results file
    :param seed: Representing the realization of the algorithm.
    :param function_type: Specifies which function should be considered
    :param scales: range of scales, which are considered
    :param multiplier_fun: specifies which multiplier was used to increase the number of samples in least squares
    :param folder_name: name of the folder where the results are stored
    :param path: Path of the results-csv file. If None, a default path will be used.
    :param save: Specifies whether the images should be saved. If False, the images are shown.
    :param save_path: Path where the images should be saved. If None, a default path will be used.
    :param same_axis_both_plots: Determines whether the same y-axis should be used for both error-plots
    Only used if save is True
    """

    if path is None:
        path = os.path.join("results", folder_name, "results_numerical_experiments.csv")

    if save_path is None:
        save_path = os.path.join("results", folder_name, "figures", function_type.name, f'dim{dimension}',
                                 f'seed{seed}')

    os.makedirs(save_path, exist_ok=True)

    data = pd.read_csv(path, sep=',', header=0)

    filtered_data = data[(data['dim'] == dimension) & (data['f_name'] == function_type.name)].copy()

    filtered_data.drop(['cpu', 'datetime', 'needed_time', 'sum_c', 'f_name', 'test_grid_seed'], axis=1,
                       inplace=True)

    smolyak_data = filtered_data[(filtered_data['method']) == 'Smolyak']
    least_squares_data = filtered_data[(filtered_data['method']) == 'Least_Squares']
    boolean_series = (filtered_data['grid_type'] == 'RANDOM_CHEBYSHEV').reindex(least_squares_data.index,
                                                                                fill_value=False)
    least_squares_data_chebyshev_weight = least_squares_data[boolean_series]
    boolean_series = (filtered_data['grid_type'] == 'RANDOM_UNIFORM').reindex(least_squares_data.index,
                                                                              fill_value=False)
    least_squares_data_uniform = least_squares_data[boolean_series]

    smolyak_data = smolyak_data.sort_values(by='scale')
    least_squares_data_chebyshev_weight = least_squares_data_chebyshev_weight.sort_values(by='scale')
    least_squares_data_uniform = least_squares_data_uniform.sort_values(by='scale')

    titles = ['Max (Abs) Error', 'L2 Error']

    errors = ['max_error', 'l_2_error']

    global_min_uniform, global_max_uniform = None, None
    global_min_l2, global_max_l2 = None, None

    if not smolyak_data.empty:
        grouped = smolyak_data.groupby('c')
        for name, group in grouped:
            w = group['w'].iloc[0]
            if np.isinf(group['max_error']).any() or np.isinf(group['l_2_error']).any():
                print(f"Skipping plot for {function_type.name}, c={name} and dimension {dimension} "
                      f"due to infinity values in errors.")
                continue
            fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
            for i, error in enumerate(errors):
                data_temp = smolyak_data[smolyak_data['c'] == name]
                n_points_sy = data_temp.loc[smolyak_data['seed'] == seed, 'n_samples']
                data_temp = least_squares_data_chebyshev_weight[least_squares_data_chebyshev_weight['c'] == name]
                n_points_ls = data_temp.loc[least_squares_data_chebyshev_weight['seed'] == seed, 'n_samples']
                xticklabels = [f"{scale}\n{n_points_sy.iloc[j]}\n{n_points_ls.iloc[j]}" for j, scale in
                               enumerate(scales)]

                label = 'Smolyak'
                smolyak_data_filtered = smolyak_data[smolyak_data['c'] == name]
                smolyak_plot_data = smolyak_data_filtered[smolyak_data_filtered['seed'] == seed]
                axs[i].plot(scales, smolyak_plot_data[error], label=label)

                label = 'Least Squares Uniform'
                least_squares_data_uniform_filtered = least_squares_data_uniform[
                    least_squares_data_uniform['c'] == name]
                least_squares_plot_data_uniform = least_squares_data_uniform_filtered[
                    least_squares_data_uniform_filtered['seed'] == seed]
                axs[i].plot(scales, least_squares_plot_data_uniform[error], label=label)

                label = 'Least Squares Chebyshev Weight'
                least_squares_data_chebyshev_weight_filtered = least_squares_data_chebyshev_weight[
                    least_squares_data_chebyshev_weight['c'] == name]
                least_squares_plot_data_chebyshev_weight = least_squares_data_chebyshev_weight_filtered[
                    least_squares_data_chebyshev_weight_filtered['seed'] == seed]
                axs[i].plot(scales, least_squares_plot_data_chebyshev_weight[error], label=label)

                axs[i].set_xticks(scales)
                axs[i].set_xticklabels(xticklabels)
                axs[i].set_title(titles[i])
                axs[i].set_xlabel('Scale\npoints Smolyak\npoints Least Squares')
                axs[i].set_ylabel('Error')
                axs[i].set_yscale('log')
                axs[i].legend()
                if error == 'max_error':
                    current_min_uniform, current_max_uniform = axs[i].get_ylim()
                    if global_min_uniform is None or current_min_uniform < global_min_uniform:
                        global_min_uniform = current_min_uniform
                    if global_max_uniform is None or current_max_uniform > global_max_uniform:
                        global_max_uniform = current_max_uniform
                elif error == 'l_2_error':
                    current_min_l2, current_max_l2 = axs[i].get_ylim()
                    if global_min_l2 is None or current_min_l2 < global_min_l2:
                        global_min_l2 = current_min_l2
                    if global_max_l2 is None or current_max_l2 > global_max_l2:
                        global_max_l2 = current_max_l2

            if same_axis_both_plots:
                minimum = min(global_min_uniform, global_min_l2)
                maximum = max(global_max_uniform, global_max_l2)

                global_min_uniform, global_max_uniform = minimum, maximum
                global_min_l2, global_max_l2 = minimum, maximum

            axs[0].set_ylim(global_min_uniform, global_max_uniform)
            axs[1].set_ylim(global_min_l2, global_max_l2)
            avg_c = np.mean(np.fromstring(name[1:-1], dtype=float, sep=' '))
            avg_c_str = str(int(np.round(avg_c, 0)))
            fig.suptitle(
                f'{function_type.name}; multiplier={multiplier_fun(1.0)}; dim={dimension}'
                f'; avg_c={avg_c}\nc={name}\nw={w}')
            plt.tight_layout()
            if save:
                c_folder = os.path.join(save_path, avg_c_str)
                os.makedirs(c_folder, exist_ok=True)
                filename = get_next_filename(c_folder)
                img_path = os.path.join(c_folder, filename)
                plt.savefig(img_path)
                plt.close()
            else:
                plt.show()
                plt.close()
    else:
        print("Chebyshev data is empty, this is deprecated")


#     for name, group in least_squares_data.groupby('c'):
#         w = group['w'].iloc[0]
#         if np.isinf(group['max_error']).any() or np.isinf(group['l_2_error']).any():
#             print(f"Skipping plot for {function_type.name}, c={name} and dimension {dimension} "
#                   f"due to infinity values in errors.")
#             continue
#         fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))
#         for i, error in enumerate(errors):
#             label = 'Least Squares'
#             axs[i].plot(scales, smolyak_data[smolyak_data['c'] == name][error], label=label)
#             axs[i].set_xticks(scales)
#             axs[i].set_title(titles[i])
#             axs[i].set_xlabel('Scale/no points')
#             axs[i].set_ylabel('Error')
#             axs[i].set_yscale('log')
#             axs[i].legend()
#
#         fig.suptitle(f'{function_type.name}\nc={name}\nw={w}')
#         plt.tight_layout()
#         if save:
#             filename = get_next_filename(save_path)
#             img_path = os.path.join(save_path, filename)
#             plt.savefig(img_path)
#         else:
#             plt.show()


def _comp_next(n: int, k: int, a: List[int], more, h, t) -> bool:
    """
    Helper method which is used to calculate the number of points in a chebyshev sparse grid.

    This function generates the next lexicographical composition of the integer `n` into `k` parts.
    A composition of `n` is a way of writing `n` as the sum of `k` non-negative integers.

    based on method comp_next in
    https://people.math.sc.edu/Burkardt/cpp_src/sandia_rules/sandia_rules.cpp

    Parameters:
    :param n: The integer to be composed.
    :param k: The number of parts in the composition.
    :param a: The current composition (to be modified in place).
    :param more: A flag indicating if there are more compositions to generate.
    :param h: A helper list used for intermediate calculations (modified in place).
    :param t: A helper list used for intermediate calculations (modified in place).

    Returns:
    :return: True if the composition `a` is not the final composition of `n` into `k` parts, False otherwise.
    """

    if not more:
        t[0] = n
        h[0] = 0
        a[0] = n
        for i in range(1, k):
            a[i] = 0
    else:
        if t[0] > 1:
            h[0] = 0
        h[0] += 1
        t[0] = a[h[0] - 1]
        a[h[0] - 1] = 0
        a[0] = t[0] - 1
        a[h[0]] += 1

    return a[k - 1] != n


def calculate_num_points(scale: int, dimension: int) -> int:
    """
    Calculates the number of points in a sparse chebyshev grid
    based on
    https://people.math.sc.edu/Burkardt/presentations/sgmga_counting.pdf

    Parameters:
    :param scale: The fineness parameter of the chebyshev sparse grid
    :param dimension: The dimension of the chebyshev sparse grid

    Returns:
    :return: The number of points in the chebyshev grid
    """

    array = [0] * (scale + 1)
    array[0] = 1
    array[1] = 2
    j = 1
    for i in range(2, scale + 1):
        j *= 2
        array[i] = j

    level = [0] * dimension
    no_points = 0

    for i in range(scale + 1):
        more = False
        h = [0]
        t = [0]

        while True:
            more = _comp_next(i, dimension, level, more, h, t)
            v = 1
            for dim in range(dimension):
                v *= array[level[dim]]
            no_points += v
            if not more:
                break

    return no_points


def find_degree(scale: int, dimension: int):
    cheby_basis_size = calculate_num_points(scale, dimension)
    degree = 1

    normal_basis_size = math.comb(dimension + degree, dimension)

    while normal_basis_size < cheby_basis_size:
        degree += 1
        normal_basis_size = math.comb(dimension + degree, dimension)

    return degree