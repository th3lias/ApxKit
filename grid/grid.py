from typing import Union

import numpy as np

from grid.grid_type import GridType


class Grid:
    """ Wrapper for an Array type used for interpolation. """

    def __init__(self, dim: int, scale: int, grid: Union[np.ndarray], grid_type: GridType, # jnp.ndarrary,
                 lower_bound: float = -1., upper_bound: float = 1.):
        self.dim = dim
        self.scale = scale
        self.grid = grid
        self.grid_type = grid_type
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_num_points(self):
        return self.grid.shape[0]

    # def jax(self):
    #     self.grid = jnp.array(self.grid)
    #
    # def numpy(self):
    #     assert isinstance(self.grid, jnp.ndarray), "Grid is already a numpy array"
    #     self.grid = np.asarray(self.grid)

    # def get_grid(self) -> Union[jnp.ndarray, np.ndarray]:
    #     return self.grid

    def __eq__(self, other):
        return (self.dim, self.scale, self.grid_type, self.lower_bound, self.upper_bound) == (
            other.dim, other.scale, other.grid_type, other.lower_bound, other.upper_bound)

    def __hash__(self):
        return hash((self.dim, self.scale, self.grid_type, self.lower_bound, self.upper_bound))
