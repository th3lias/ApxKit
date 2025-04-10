from __future__ import annotations

import glob
import math
import os
import time
from typing import Callable, Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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


def reformat_old_file_to_new_one(path: str, old: bool) -> None:
    df = pd.read_csv(path, header=0, decimal='.', sep=',')

    new_column_order = ["dim", "scale", "method", "w", "c", "sum_c", "grid_type", "basis_type", "method_type",
                        "n_samples", "seed", "f_name", "ell_2_error", "ell_infty_error", "datetime", "needed_time"]
    old_column_order = ["dim", "method", "w", "c", "sum_c", "grid_type", "basis_type", "method_type", "n_samples",
                        "scale", "seed", "f_name", "ell_2_error", "ell_infty_error", "cpu", "datetime", "needed_time"]

    if old:
        # TODO: do this with the 3 files from the server (30.03, 31.03, 02.04)
        # reorder the columns
        df.drop(columns=['cpu'], inplace=True)
        df = df[new_column_order]

        # change numerics for c and w
        df['w'] = df['w'].apply(lambda x: ','.join(x.replace('\n', '').split()) if isinstance(x, str) else x)
        df['c'] = df['c'].apply(lambda x: ','.join(x.replace('\n', '').split()) if isinstance(x, str) else x)

        # change grid_type from RANDOM_CHEBYSHEV zu SPARSE

        def replace_name(x: str):
            if x == "CHEBYSHEV":
                return "SPARSE"
            if x == "RANDOM_CHEBYSHEV":
                return "CHEBYSHEV"
            if x == "RANDOM_UNIFORM":
                return "UNIFORM"
            return x

        df['grid_type'] = df['grid_type'].apply(lambda x: replace_name(x))

        # replace NUMPY_LSTSQ with SCIPY_LSTSQ_GELSY
        df['method_type'] = df['method_type'].apply(lambda x: x.replace("NUMPY_LSTSQ", "SCIPY_LSTSQ_GELSY"))

    else:

        def replace_name_new(method: str, grid_type: str):
            if method == 'Smolyak' and grid_type == 'CHEBYSHEV':
                return "SPARSE"
            return grid_type

        df['grid_type'] = df.apply(lambda x: replace_name_new(x.method, x.grid_type), axis=1)

    # save the dataframe
    df.to_csv(path, index=False, sep=',', decimal='.', header=True)


def combine_result_files_to_combined_one(folder_path: str, output_file_path: str = None):
    if output_file_path is None:
        output_file_path = os.path.join(folder_path, "combined_results_numerical_experiments.csv")

    data_frames = []

    search_pattern = os.path.join(folder_path, '**', 'results_numerical_experiments.csv')
    csv_files = glob.glob(search_pattern, recursive=True)

    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path)

            data_frames.append(df)
        except Exception as e:
            print(f"Could not process {file_path}: {e}")

    if data_frames:
        combined_df = pd.concat(data_frames, ignore_index=True)
        try:
            combined_df.to_csv(output_file_path, index=False)
            print(f"Combined CSV file saved to {output_file_path}")
        except Exception as e:
            print(f"Could not save the combined file: {e}")
    else:
        print("No files found to combine.")


if __name__ == '__main__':
    # path = r"C:\Users\jakob\OneDrive - Johannes Kepler Universität Linz\Studium\JKU\cur_sem\_Student_Assistant\Assistance_Mario\SS24\Forschung\backup_results"
    # #
    # combine_result_files_to_combined_one(path)

    # path = r"C:\Users\jakob\OneDrive - Johannes Kepler Universität Linz\Studium\JKU\cur_sem\_Student_Assistant\Assistance_Mario\SS24\Forschung\backup_results\30_03_2025_13_47_35\results_numerical_experiments.csv"
    # old = False

    # reformat_old_file_to_new_one(path, old)

    raise ValueError("This file is not meant to be run directly. Please use the appropriate files.")
