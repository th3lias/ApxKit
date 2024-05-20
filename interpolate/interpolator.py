from typing import Callable, Union, List

from grid.grid import Grid


class Interpolator:
    def __init__(self, grid: Grid):
        self.grid = grid

    def interpolate(self, f: Union[Callable, List[Callable]]) -> Callable:
        raise NotImplementedError

    def set_grid(self, grid: Grid):
        self.grid = grid
