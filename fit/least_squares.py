from fit.fitter import Fitter
from grid.grid.grid import Grid


class LeastSquares(Fitter):
    def __init__(self, grid: Grid):
        super().__init__(grid)
