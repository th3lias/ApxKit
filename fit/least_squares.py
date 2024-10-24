#  Created 2024. (Elias Mindlberger)
from typing import Callable

from fit.fitter import Fitter
from function.model.least_squares_model import LeastSquaresModel
from grid.grid.grid import Grid


class LeastSquares(Fitter):
    def __init__(self, grid: Grid):
        super().__init__(grid)

    def fit(self, f: list[Callable]) -> LeastSquaresModel:
        return
