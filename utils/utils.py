from __future__ import annotations

import glob
import math
import os
import time
from typing import Callable, Union, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ast

from grid.grid.grid import Grid


def l2_error_function_values(y: np.ndarray, y_hat: np.ndarray) -> Union[float, np.ndarray]:
    """
    Calculates the ell_2 error estimate by comparing the true function values and the approximated function values by
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
    Calculates the ell_2 error estimate by comparing the true function and the approximation f_hat on a test grid by
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
