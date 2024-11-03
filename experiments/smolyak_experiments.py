from __future__ import annotations

from fit.smolyak import Smolyak
from function.f import Function
from function.provider import ParametrizedFunctionProvider
from function.type import FunctionType
from grid.grid.random_grid import RandomGrid
from grid.grid.rule_grid import RuleGrid

import numpy as np # TODO [Jakob] I am unhappy with the current error calculation. Maybe we could like make here a "quality assessment wrapper function/class".

from grid.provider.rule_grid_provider import RuleGridProvider


# TODO: Add docstring
# TODO: Inconsistent. Why do we return only the interpolated here and in the method where we accept a list of functions, we return the losses
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


# TODO: Add docstring
def interpolate_and_evaluate_list(functions: list[Function], training_grid: RuleGrid, points: RandomGrid):
    """
    This function interpolates the given functions using a standard grid. The functions are assumed to have the same
    properties as the grid. Interpolation fails otherwise. RandomGrid has to have the same dimensionality as the functions.
    """
    assert points.input_dim == functions[0].dim, "The dimensionality of the functions and the grid do not match."
    assert training_grid.input_dim == functions[0].dim, "The dimensionality of the functions and the grid do not match."
    fitter = Smolyak(grid=training_grid)
    l2_losses = list()
    abs_losses = list()
    for function in functions:# TODO[Jakob] Optimize?? Calls the function $n$ times. Maybe this is slow. Unless it is optimized, such that only the fitter step takes long
        model = fitter.fit(function)
        y = function.__call__(points.grid)
        interpolated = model.__call__(points.grid)
        l2_losses.append((np.square(y - interpolated)).mean())
        abs_losses.append(np.max(np.abs(y - interpolated)))
        fitter.fitted = False
    return l2_losses, abs_losses

class FitnessTrainer:
    def __init__(self, scales: list[int], dims: list[int]):
        self.scales = scales
        self.dims = dims

    def workout_in_the_smolyak_gym(self, functions: list[Function]) -> list[tuple[int, int, float, float]]:
        """
            Send a couple of newbies to the Smolyak Gym to train.
        """
        losses = list()
        for scale in self.scales:
            for dim in self.dims:
                test_grid = np.random.uniform(0, 1, (1000, dim))
                rule_grid = RuleGridProvider(input_dim=dim).generate(scale)
                for function in functions:
                    fitter = Smolyak(grid=rule_grid)
                    model = fitter.fit(function)
                    l2, l_inf = self.calculate_strength_gain(function, model, test_grid)
                    losses.append((function.name, scale, dim, l2, l_inf))
                del test_grid, rule_grid
        return losses

    def workout_in_the_least_squares_gym(self, functions: list[Function]):
        pass

    @staticmethod
    def calculate_strength_gain(function: Function, model: Function, barbells: np.array) -> tuple[float, float]:
        y = function(barbells)
        y_hat = model(barbells)
        l2_loss = (np.square(y - y_hat)).mean()
        abs_loss = np.max(np.abs(y - y_hat))
        return l2_loss, abs_loss


###########################################
# Script Part
###########################################

if __name__ == "__main__":
    dims = [1, 2, 3]
    scales = [2, 3, 4, 5]
    types = [e.name for e in FunctionType]
    functions = list()
    for t in types:
        for d in dims:
            c = np.random.uniform(-1, 1, (1, d))
            w = np.random.uniform(0, 1, (1, d))
            functions.append(ParametrizedFunctionProvider().get_function(FunctionType[t], d, c, w))
    trainer = FitnessTrainer(scales, dims)
    results = trainer.workout_in_the_smolyak_gym(functions)
    print(results)
