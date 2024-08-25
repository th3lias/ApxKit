#  Created 2024. (Elias Mindlberger)
from grid.grid import Grid
from grid.rule.random_grid_rule import RandomGridRule


class RandomGrid(Grid):
    def __init__(self, input_dim: int, output_dim: int, scale: int, grid, grid_type: RandomGridRule,
                 lower_bound: float = 0., upper_bound: float = 1., seed=None):
        super().__init__(input_dim, output_dim, scale, grid, grid_type, lower_bound, upper_bound)
        self.seed = seed if seed else None

    def __eq__(self, other):
        """
        This method is used to compare two RandomGrid objects. The probability that two random grids are equal is
        0. Thus, we here only check if the two objects have the same properties. For checking equality in values, use
        the method `is_equal` instead.
        """
        return super().__eq__(other)

    def is_equal(self, other):
        """
        This method is used to compare two RandomGrid objects. The probability that two random grids are equal is
        0. Thus, we here only check if the two objects have the same properties. For checking equality in values, use
        the method `is_equal` instead.
        """
        return super().__eq__(other) and (self.grid == other.grid).all()
