#  Created 2024. (Elias Mindlberger)

import jax.numpy as jnp
import numpy as np
from TasmanianSG import TasmanianSparseGrid

from grid.rule.rule import GridRule


class Grid:
    """ Wrapper for an Array type used for interpolation. """

    # Perhaps there is a need to include this as a Wrapper for TasmanianSparseGrids as well.
    def __init__(self, input_dim: int, output_dim: int, scale: int,
                 grid: jnp.ndarray | np.ndarray | TasmanianSparseGrid, rule: GridRule, lower_bound: float = 0.,
                 upper_bound: float = 1.):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.scale = scale
        self.grid = grid
        self.rule = rule
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_num_points(self):
        raise NotImplementedError("Method not implemented for this grid type.")

    def jax(self):
        """
        This method has different implementations depending on the type of the grid.
        The implementation is thus delegated.
        """
        raise NotImplementedError("Method not implemented for this grid type.")

    def numpy(self):
        """
        This method has different implementations depending on the type of the grid.
        The implementation is thus delegated.
        """
        raise NotImplementedError("Method not implemented for this grid type.")

    def vstack(self, other):
        """
        This method may be implemented when grid is of type np.ndarray or jnp.ndarray.
        """
        raise NotImplementedError("Method not implemented for this grid type.")

    def __eq__(self, other):
        return (self.input_dim, self.output_dim, self.scale, self.grid_type, self.lower_bound, self.upper_bound) == (
            other.input_dim, other.output_dim, other.scale, other.grid_type, other.lower_bound, other.upper_bound)

    def __hash__(self):
        return hash((self.input_dim, self.output_dim, self.scale, self.grid_type, self.lower_bound, self.upper_bound))
