from typing import Callable

from grid.grid import Grid


class Interpolator:
    def __init__(self, grid: Grid):
        self.grid = grid

    def interpolate(self, f: Callable) -> Callable:
        raise NotImplementedError
