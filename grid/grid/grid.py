import numpy as np
from TasmanianSG import TasmanianSparseGrid
from dataclasses import dataclass

from grid.rule.rule import GridRule


@dataclass
class Grid:
    """Base wrapper around a point set used for interpolation or evaluation."""

    input_dim: int
    scale: int
    grid: np.ndarray | TasmanianSparseGrid
    rule: GridRule
    lower_bound: float = 0.
    upper_bound: float = 1.

    def get_num_points(self):
        raise NotImplementedError("Must be implemented in subclasses.")

    def __array__(self):
        """Return grid points as a numpy array."""
        if isinstance(self.grid, TasmanianSparseGrid):
            return self.grid.getPoints()
        elif isinstance(self.grid, np.ndarray):
            return self.grid
        else:
            raise ValueError(f'Grid has type {type(self.grid)}, which is not supported')
