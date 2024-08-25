#  Created 2024. (Elias Mindlberger)
import numpy as np
from TasmanianSG import TasmanianSparseGrid

from grid.rule.rule_grid_rule import RuleGridRule
from grid.src.grid import Grid


class RuleGrid(Grid):
    def __init__(self, input_dim: int, output_dim: int, scale: int, grid: TasmanianSparseGrid, rule: RuleGridRule,
                 lower_bound: float = 0., upper_bound: float = 1.):
        super().__init__(input_dim, output_dim, scale, grid, rule, lower_bound, upper_bound)

    def get_num_points(self):
        """Wrapper function for the getNumPoints method of the TasmanianSparseGrid object."""
        return self.grid.getNumPoints()

    def get_needed_points(self) -> np.ndarray:
        """Wrapper function for the getNeededPoints method of the TasmanianSparseGrid object."""
        return self.grid.getNeededPoints()

    def load_needed_values(self, llf_vals: np.ndarray) -> None:
        """Wrapper function for the loadNeededValues method of the TasmanianSparseGrid object."""
        self.grid.loadNeededValues(llf_vals)

    def __eq__(self, other):
        """
        This method is used to compare two RandomGrid objects. The probability that two random grids are equal is
        0. Thus, we here only check if the two objects have the same properties. For checking equality in values, use
        the method `equals` instead.
        """
        return super().__eq__(other)

    def equals(self, other):
        return self.__eq__(other) and np.all(self.grid == other.grid)
