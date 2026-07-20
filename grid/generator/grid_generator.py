import numpy as np
import os

from grid.grid.grid import Grid


class GridGenerator:
    """Abstract base for grid generators."""

    def __init__(self, name: str, abbr_name: str):
        self.name = name
        self.abbr_name = abbr_name

    def get_grid(self, **kwargs) -> Grid:
        # TODO: Look for saved grids, such that results stay consistent.
        # TODO: If no grid is found, generate a new one and save it for future use.
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _load_grid_if_available(self, filename: str) -> np.ndarray | None:
        if os.path.isfile(filename):
            return np.load(filename)['arr_0']
        else:
            return None
