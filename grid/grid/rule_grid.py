import numpy as np
from TasmanianSG import TasmanianSparseGrid

from grid.rule.rule_grid_rule import RuleGridRule
from grid.grid.grid import Grid


class RuleGrid(Grid):
    """Grid backed by a TasmanianSparseGrid (deterministic rule-based points)."""

    def __init__(self, input_dim: int, output_dim: int, scale: int, grid: TasmanianSparseGrid, rule: RuleGridRule,
                 lower_bound: float = 0., upper_bound: float = 1.):
        super().__init__(input_dim, scale, grid, rule, lower_bound, upper_bound)
        self.output_dim = output_dim
        self.needed_points = None

    def get_num_points(self):
        return self.grid.getNumPoints()

    def get_needed_points(self) -> np.ndarray:
        if self.needed_points is None:
            self.needed_points = self.grid.getNeededPoints()
        return self.needed_points

    def load_needed_values(self, llf_vals: np.ndarray) -> None:
        self.grid.loadNeededValues(llf_vals)

    def set_domain_transform(self, domain_transform):
        self.grid.setDomainTransform(domain_transform)
