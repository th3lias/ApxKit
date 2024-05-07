from typing import Callable
import time

import matplotlib.pyplot as plt
import numpy as np


def l2_error(f: Callable, f_hat: Callable, grid: np.ndarray) -> np.float64:
    """
    Calculates the ell_2 error estimate by comparing the true function and the approximation f_hat on a test grid by
    calculating the mean-squared absolute difference
    :param f: function that should be approximated
    :param f_hat: approximation of the function
    :param grid: grid where the approximation should be compared vs the original function
    :return: error estimate
    """

    y_hat = f_hat(grid)
    y = f(grid)

    error = np.sqrt(np.mean(np.square(np.abs(y - y_hat)))).squeeze()

    return error


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

    error = np.max(np.abs(y_hat - y)).squeeze()

    return error


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
        print(f"{method.__name__}, {args}, {kw}, {te-ts}")
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


def plot_results(results: dict, scale_range: range, name: str) -> None:
    """
    Plot the results of the experiments
    :param results: dictionary containing the results
    :param scale_range: range of scales
    :param name: name of the experiment
    :return:
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    scales = scale_range
    smolyak_max_diff = [results['smolyak'][scale]['max_diff'] for scale in scales]
    least_squares_max_diff = [results['least_squares'][scale]['max_diff'] for scale in scales]

    smolyak_ell_2 = [results['smolyak'][scale]['ell_2'] for scale in scales]
    least_squares_ell_2 = [results['least_squares'][scale]['ell_2'] for scale in scales]

    axs[0].plot(scales, smolyak_max_diff, label='Smolyak')
    axs[0].plot(scales, least_squares_max_diff, label='Least Squares')
    axs[0].set_xticks(scale_range)
    axs[0].set_title('Max (Abs) Error')
    axs[0].set_xlabel('Scale')
    axs[0].set_ylabel('Max Error')
    axs[0].set_yscale('log')
    axs[0].legend()

    axs[1].plot(scales, smolyak_ell_2, label='Smolyak')
    axs[1].plot(scales, least_squares_ell_2, label='Least Squares')
    axs[1].set_xticks(scale_range)
    axs[1].set_title('L2 Error')
    axs[1].set_xlabel('Scale')
    axs[1].set_ylabel('L2 Error')
    axs[1].set_yscale('log')
    axs[1].legend()

    fig.suptitle(name)
    plt.tight_layout()
    plt.show()
