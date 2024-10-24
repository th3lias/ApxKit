from __future__ import annotations

from fit.smolyak import Smolyak
from function.f import Function
from grid.grid.random_grid import RandomGrid
from grid.grid.rule_grid import RuleGrid


def interpolate_and_evaluate(function: Function, training_grid: RuleGrid, points: RandomGrid):
    """
    This function interpolates the given function using a standard grid. The function is assumed to have the same
    properties as the grid. Interpolation fails otherwise. RandomGrid has to have the same dimensionality as the function.
    """
    assert points.grid.input_dim == function.dim, "The dimensionality of the function and the grid do not match."
    assert training_grid.input_dim == function.dim, "The dimensionality of the function and the grid do not match."
    fitter = Smolyak(grid=training_grid)
    model = fitter.fit(function)
    interpolated = model.__call__(points.grid)
    return interpolated


def interpolate_and_evaluate_list(functions: list[Function], training_grid: RuleGrid, points: RandomGrid):
    """
    This function interpolates the given functions using a standard grid. The functions are assumed to have the same
    properties as the grid. Interpolation fails otherwise. RandomGrid has to have the same dimensionality as the functions.
    """
    assert points.grid.input_dim == functions[0].dim, "The dimensionality of the functions and the grid do not match."
    assert training_grid.input_dim == functions[0].dim, "The dimensionality of the functions and the grid do not match."
    fitter = Smolyak(grid=training_grid)
    l2_losses = list()
    abs_losses = list()
    for function in functions:
        model = fitter.fit(function)
        y = function.__call__(points.grid)
        interpolated = model.__call__(points.grid)
        l2_losses.append(((y - interpolated) ** 2).mean())
        abs_losses.append(max(abs(y - interpolated)))
    return l2_losses, abs_losses
