#  Created 2024. (Elias Mindlberger)
from typing import Callable

from function.model.model import Model
from function.model import Model
from grid.grid.grid import Grid


class Fitter:
    """
    Abstract fitter class that defines the interface for all fitters. Currently, we only support fitting functions
    which map to one output, i.e. output_dim=1!
    """

    def __init__(self, grid: Grid):
        self.grid = grid
        self.scale = grid.scale
        self.dim = grid.input_dim

    def fit(self, f: list[Callable]) -> Model:
        raise NotImplementedError("The method `fit` must be implemented by the subclass.")
