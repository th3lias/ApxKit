from __future__ import annotations

import numpy as np

from fit.least_squares import LeastSquares
from function.f import Function
from grid.grid.grid import Grid
from grid.grid.random_grid import RandomGrid


# TODO [Jakob] Check this.

def interpolate_and_evaluate(function: Function, training_grid: Grid, points: RandomGrid):
    """
    This function interpolates the given function using a standard grid. The function is assumed to have the same
    properties as the grid. Interpolation fails otherwise. RandomGrid has to have the same dimensionality as the function.
    """
    assert points.grid.input_dim == function.dim, "The dimensionality of the function and the grid do not match."
    assert training_grid.input_dim == function.dim, "The dimensionality of the function and the grid do not match."
    fitter = LeastSquares(grid=training_grid)
    model = fitter.fit(function)
    interpolated = model.__call__(points.grid)
    return interpolated


def interpolate_and_evaluate_list(functions: list[Function], training_grid: Grid, points: RandomGrid):
    """
    This function interpolates the given functions using a standard grid. The functions are assumed to have the same
    properties as the grid. Interpolation fails otherwise. RandomGrid has to have the same dimensionality as the functions.
    """
    assert points.grid.input_dim == functions[0].dim, "The dimensionality of the functions and the grid do not match."
    assert training_grid.input_dim == functions[0].dim, "The dimensionality of the functions and the grid do not match."
    fitter = LeastSquares(grid=training_grid)
    l2_losses = list()
    abs_losses = list()  # TODO[Jakob] Optimize?? Calls the function $n$ times. Maybe this is slow. Unless it is optimized, such that only the fitter step takes long
    for function in functions:
        model = fitter.fit(function)
        y = function.__call__(points.grid)
        interpolated = model.__call__(points.grid)
        fitter.fitted = False  # TODO[Jakob] This is a bit hacky. We don't have a fitted argument here in Least Squares Class.
        l2_losses.append((np.square(y - interpolated)).mean())
        abs_losses.append(np.max(np.abs(y - interpolated)))
    return l2_losses, abs_losses
