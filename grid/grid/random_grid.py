#  Created 2024. (Elias Mindlberger)
import numpy as np

from grid.rule.random_grid_rule import RandomGridRule
from grid.grid.grid import Grid


class RandomGrid(Grid):
    def __init__(self, input_dim: int, output_dim: int, scale: int, grid: np.ndarray | jnp.ndarray,
                 rule: RandomGridRule, lower_bound: float = 0., upper_bound: float = 1., seed=None):
        super().__init__(input_dim, output_dim, scale, grid, rule, lower_bound, upper_bound)
        self.seed = seed if seed else None

    def get_num_points(self):
        return self.grid.shape[0]

    def vstack(self, other):
        self.grid = np.vstack((self.grid, other.grid))
        self.scale += other.scale
        return self

    def __eq__(self, other):
        """
        This method is used to compare two RandomGrid objects. The probability that two random grids are equal is
        0. Thus, we here only check if the two objects have the same properties. For checking equality in values, use
        the method `is_equal` instead.
        """
        return super().__eq__(other)
