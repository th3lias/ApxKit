"""
Utility functions: error metrics, point counting for Smolyak grids, and helpers.
"""
from __future__ import annotations

import math
import os
import time
from collections.abc import Callable

import numpy as np


def l2_error_function_values(y: np.ndarray, y_hat: np.ndarray) -> float | np.ndarray:
    """Estimate the ℓ₂ error from function values: √(mean((y − ŷ)²)).

    For 2-D inputs, reduction is over axis 0 (points), yielding one error per function.
    """
    if y_hat.ndim == 1:
        return np.sqrt(np.mean(np.square(np.abs(y - y_hat)))).squeeze()
    # axis=0: average over points → one error per function column
    return np.sqrt(np.mean(np.square(np.abs(y - y_hat)), axis=0)).squeeze()


def max_error_function_values(y: np.ndarray, y_hat: np.ndarray) -> float | np.ndarray:
    """Estimate the ℓ∞ error from function values: max(|y − ŷ|)."""
    if y_hat.ndim == 1:
        return np.max(np.abs(y_hat - y)).squeeze()
    return np.max(np.abs(y_hat - y), axis=1).squeeze()


def l2_error(f: Callable, f_hat: Callable, grid: np.ndarray) -> float:
    """ℓ₂ error between a function and its approximation on a test grid."""
    return l2_error_function_values(y=f(grid), y_hat=f_hat(grid))


def max_abs_error(f: Callable, f_hat: Callable, grid: np.ndarray) -> float:
    """ℓ∞ error between a function and its approximation on a test grid."""
    return max_error_function_values(y=f(grid), y_hat=f_hat(grid))


def timeit(method):
    """Decorator that prints execution time of the wrapped function."""
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print(f"{method.__name__}, {args}, {kw}, {te - ts}")
        return result
    return timed


@timeit
def test_function_time(func: Callable, n: int, *args, **kwargs):
    """Run *func* n+1 times and return the last result (for benchmarking)."""
    for i in range(n):
        func(*args, **kwargs)
    return func(*args, **kwargs)


def _remove_almost_identical_rows(arr: np.ndarray, tol=1e-8):
    """Remove near-duplicate rows (reference implementation for testing only)."""
    unique_rows = [arr[0]]
    for row in arr[1:]:
        if not any(np.allclose(row, unique_row, atol=tol) for unique_row in unique_rows):
            unique_rows.append(row)
    return np.array(unique_rows)


def sample(dim: int | tuple[int], low: float = 0., high: float = 1.):
    return np.random.uniform(low=low, high=high, size=dim)


def get_next_filename(path, extension='png'):
    """Return the next available numbered filename in *path*."""
    files = [f for f in os.listdir(path) if f.endswith('.' + extension)]
    numbers = [int(os.path.splitext(f)[0]) for f in files if f.split('.')[0].isdigit()]
    next_number = max(numbers, default=0) + 1
    return f"{next_number}.{extension}"


def _comp_next(n: int, k: int, a: list[int], more, h, t) -> bool:
    """
    Generate the next lexicographical composition of *n* into *k* parts.

    Based on ``comp_next`` in
    https://people.math.sc.edu/Burkardt/cpp_src/sandia_rules/sandia_rules.cpp
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


def calculate_num_points(dim: int, scale: int) -> int:
    """
    Number of points in a Clenshaw-Curtis Smolyak sparse grid.

    Based on https://people.math.sc.edu/Burkardt/presentations/sgmga_counting.pdf
    """
    array = [0] * (scale + 1)
    array[0] = 1
    array[1] = 2
    j = 1
    for i in range(2, scale + 1):
        j *= 2
        array[i] = j

    level = [0] * dim
    no_points = 0

    for i in range(scale + 1):
        more = False
        h = [0]
        t = [0]

        while True:
            more = _comp_next(i, dim, level, more, h, t)
            v = 1
            for d in range(dim):
                v *= array[level[d]]
            no_points += v
            if not more:
                break

    return no_points


def find_degree(scale: int, dimension: int):
    """Find the total polynomial degree matching the Smolyak basis size."""
    cheby_basis_size = calculate_num_points(dimension, scale)
    degree = 1

    normal_basis_size = math.comb(dimension + degree, dimension)

    while normal_basis_size < cheby_basis_size:
        degree += 1
        normal_basis_size = math.comb(dimension + degree, dimension)

    return degree
