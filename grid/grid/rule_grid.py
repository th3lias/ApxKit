import numpy as np
from TasmanianSG import TasmanianSparseGrid

from grid.rule.rule_grid_rule import RuleGridRule
from grid.grid.grid import Grid


class RuleGrid(Grid):
    def __init__(self, input_dim: int, output_dim: int, scale: int, grid: TasmanianSparseGrid, rule: RuleGridRule,
                 lower_bound: float = 0., upper_bound: float = 1.):
        super().__init__(input_dim, output_dim, scale, grid, rule, lower_bound, upper_bound)
        self.needed_points = None

    def get_num_points(self):
        """Wrapper function for the getNumPoints method of the TasmanianSparseGrid object."""
        return self.grid.getNumPoints()

    def get_needed_points(self) -> np.ndarray:
        """Wrapper function for the getNeededPoints method of the TasmanianSparseGrid object."""
        if self.needed_points is None:
            self.needed_points = self.grid.getNeededPoints()
        return self.needed_points

    def load_needed_values(self, llf_vals: np.ndarray) -> None:
        """Wrapper function for the loadNeededValues method of the TasmanianSparseGrid object."""
        self.grid.loadNeededValues(llf_vals)

    def __eq__(self, other):
        return super().__eq__(other)

    def set_domain_transform(self, domain_transform):
        self.grid.setDomainTransform(domain_transform)
