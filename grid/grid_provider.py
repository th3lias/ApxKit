"""
Provides sparse grids and random grids.
"""
import numpy as np
from grid.grid_type import GridType


class GridProvider:
    def __init__(self,
                 dimension: np.int8,
                 upper_bound: np.ndarray,
                 lower_bound: np.ndarray,
                 seed: np.int8 = None):
        self.dim = dimension
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.seed = seed

    def generate(self, grid_type: GridType) -> np.ndarray:
        pass
