#  Created 2024. (Elias Mindlberger)
import numpy as np

from grid.provider.grid_provider import GridProvider
from grid.rule.random_grid_rule import RandomGridRule
from grid.grid.grid import Grid
from grid.grid.random_grid import RandomGrid
from grid.utils import sample_chebyshev_univariate
from utils.utils import calculate_num_points


class RandomGridProvider(GridProvider):
    def __init__(self, input_dim: int, output_dim: int = 1, lower_bound: float = 0., upper_bound: float = 1.,
                 seed: int = None, rule: RandomGridRule = RandomGridRule.UNIFORM, multiplier: float = 1.):
        super().__init__(input_dim, output_dim, lower_bound, upper_bound)
        self.seed = seed if seed else None
        self.rng = np.random.default_rng(seed=self.seed)
        self.rule = rule
        self.multiplier = multiplier

    def generate(self, scale: int) -> RandomGrid:
        n_points = int(calculate_num_points(scale, self.input_dim) * self.multiplier)
        if self.rule == RandomGridRule.UNIFORM:
            return RandomGrid(self.input_dim, self.output_dim, scale, self._generate_uniform(n_points), rule=self.rule,
                              seed=self.seed)
        elif self.rule == RandomGridRule.CHEBYSHEV:
            return RandomGrid(self.input_dim, self.output_dim, scale, self._generate_chebyshev(n_points),
                              rule=self.rule, seed=self.seed)

    def increase_scale(self, current_grid: Grid, delta: int = 1, keep_old_pts: bool = True) -> Grid:
        target_no_points = int(
            calculate_num_points(scale=current_grid.scale + delta, dimension=self.input_dim) * self.multiplier)
        if self.rule == RandomGridRule.UNIFORM:
            if keep_old_pts:
                new_pts = self._generate_uniform(target_no_points - current_grid.get_num_points())
                return current_grid.vstack(Grid(self.input_dim, self.output_dim, delta, new_pts, rule=self.rule))
            else:
                old_scale = current_grid.scale
                del current_grid
                return Grid(self.input_dim, self.output_dim, old_scale + delta,
                            self._generate_uniform(target_no_points), rule=self.rule)
        elif self.rule == RandomGridRule.CHEBYSHEV:
            if keep_old_pts:
                new_pts = self._generate_chebyshev(target_no_points - current_grid.get_num_points())
                return current_grid.vstack(Grid(self.input_dim, self.output_dim, delta, new_pts, rule=self.rule))
            else:
                old_scale = current_grid.scale
                del current_grid
                return Grid(self.input_dim, self.output_dim, old_scale + delta,
                            self._generate_chebyshev(target_no_points), rule=self.rule)

    def _generate_uniform(self, num_points: int):
        return self.rng.uniform(low=self.lower_bound, high=self.upper_bound, size=(num_points, self.input_dim))

    def _generate_chebyshev(self, num_points: int):
        grid_points = np.empty(shape=(num_points, self.input_dim))
        for i in range(self.input_dim):
            grid_points[:, i] = sample_chebyshev_univariate(num_points, self.lower_bound, self.upper_bound)
        return grid_points
