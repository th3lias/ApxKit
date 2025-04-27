from typing import Callable
import numpy as np

from function.model.model import Model


class LeastSquaresModel(Model):
    def __init__(self, f: Callable, dim: int, upper: float, lower: float):
        super(LeastSquaresModel, self).__init__(f, dim, upper, lower)
        self.beta = None
        self.grid = None
        self.evaluate = None
        self.kwargs = None

    def set_solution(self, solution: np.ndarray):
        self.beta = solution

    def set_grid(self, grid: np.ndarray):
        self.grid = grid

    def set_kwargs(self, **kwargs):
        self.kwargs = kwargs

    def set_evaluate(self, evaluate: Callable):
        """
            Takes a Callable with parameters x, grid, beta and possibly some kwargs.
            Specifies how the model is evaluated on its grid and coefficients.
        """
        self.evaluate = evaluate

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.evaluate(x, self.grid, self.beta, **self.kwargs)
