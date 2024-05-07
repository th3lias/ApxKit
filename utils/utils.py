from typing import Callable
import time

import matplotlib.pyplot as plt
import numpy as np


def ell_2_error_estimate(f: Callable, f_hat: Callable, d: np.int8, no_samples: np.int16,
                         lower_bound: np.float64 = np.float64(-1.0),
                         upper_bound: np.float64 = np.float64(1.0)) -> np.float64:
    """
    Calculates the ell_2 error estimate by sampling no_samples points in [lower_bound, upper_bound]^d and
    calculating the mean-squared absolute difference
    :param f: function that should be approximated
    :param f_hat: approximation of the function
    :param d: dimension of the function
    :param no_samples: number of samples
    :param lower_bound: lower bound of the rectangle to sample from
    :param upper_bound: upper bound of the rectangle to sample from
    :return: error estimate
    """
    points = np.random.uniform(low=lower_bound, high=upper_bound, size=(no_samples, d))

    y_hat = f_hat(points)
    y = f(points)

    error = np.sqrt(np.mean(np.square(np.abs(y - y_hat)))).squeeze()

    return error


def max_abs_error(f: Callable, f_hat: Callable, d: np.int8, no_samples: np.int16,
                  lower_bound: np.float64 = np.float64(-1.0),
                  upper_bound: np.float64 = np.float64(1.0)) -> np.float64:
    """
        Calculates the estimated max absolute value distance by sampling no_samples points in
        [lower_bound, upper_bound]^d and calculating the max absolute value difference at those points
        :param f: function that should be approximated
        :param f_hat: approximation of the function
        :param d: dimension of the function
        :param no_samples: number of samples
        :param lower_bound: lower bound of the rectangle to sample from
        :param upper_bound: upper bound of the rectangle to sample from
        :return: error estimate
        """

    points = np.random.uniform(low=lower_bound, high=upper_bound, size=(no_samples, d))
    y_hat = f_hat(points)
    y = f(points)

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


def _remove_duplicates_squared_memory(arr: np.ndarray, tol: np.float32=np.float32(1e-8)):
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
