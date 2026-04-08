import numpy as np

from grid.rule.random_grid_rule import RandomGridRule
from grid.grid.grid import Grid


class RandomGrid(Grid):
    """Grid backed by a randomly sampled numpy array."""

    def __init__(self, input_dim: int, scale: int, n_points: int, grid: np.ndarray,
                 rule: RandomGridRule, lower_bound: float = 0., upper_bound: float = 1., seed: int = None):
        super().__init__(input_dim, scale, grid, rule, lower_bound, upper_bound)
        self.n_points = n_points

    def get_num_points(self):
        return self.n_points

    def vstack(self, other):
        """Append another RandomGrid's points to this one (in-place)."""
        if not isinstance(other, RandomGrid):
            raise ValueError("Cannot stack RandomGrid with non-RandomGrid object.")
        self.grid = np.vstack((self.grid, other.grid))
        self.n_points += other.n_points
        return self
