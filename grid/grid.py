from typing import Union

import jax.numpy as jnp
import numpy as np

from grid.rule.rule import GridRule


# import numpy as jnp


class Grid:
    """ Wrapper for an Array type used for interpolation. """

    def __init__(self, input_dim: int, output_dim: int, scale: int, grid: Union[jnp.ndarray, np.ndarray],
                 grid_type: GridRule, lower_bound: float = 0., upper_bound: float = 1.):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scale = scale
        self.grid = grid
        self.grid_type = grid_type
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_num_points(self):
        return self.grid.shape[0]

    def jax(self):
        self.grid = jnp.array(self.grid)

    def numpy(self):
        assert isinstance(self.grid, jnp.ndarray), "Grid is already a numpy array"
        self.grid = np.asarray(self.grid)

    def vstack(self, other):
        self.grid = np.vstack((self.grid, other.grid))
        self.scale += other.scale
        return self

    def __eq__(self, other):
        return (self.input_dim, self.output_dim, self.scale, self.grid_type, self.lower_bound, self.upper_bound) == (
            other.input_dim, other.output_dim, other.scale, other.grid_type, other.lower_bound, other.upper_bound)

    def __hash__(self):
        return hash((self.input_dim, self.output_dim, self.scale, self.grid_type, self.lower_bound, self.upper_bound))
