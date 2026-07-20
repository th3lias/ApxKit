import hashlib
import os.path
from collections.abc import Callable

import numpy as np

from grid.generator.random_grid_generator import RandomGridGenerator
from grid.grid.random_grid import RandomGrid
from grid.rule.random_grid_rule import RandomGridRule
from utils.utils import calculate_num_points


class ChebyshevGridGenerator(RandomGridGenerator):
    """
    Generate random grids with points drawn from the Chebyshev (arcsine) distribution.

    Uses inverse-transform sampling: uniform draws in [-π/2, π/2] mapped
    through sin(·) yield the arcsine distribution on [-1, 1].
    """

    def __init__(self, seed=42, multiplier_fun: Callable = lambda x: x):
        super().__init__("CHEBYSHEV", "CS", seed)
        self._cache = dict()
        self._multiplier_fun = multiplier_fun

        self.used_grids_filepath = os.path.join("used_grids", "chebyshev_grid_")

    def _generate_seed(self, n_points: int, dim: int, scale: int, lower_bound: float = 0.,
                       upper_bound: float = 1.) -> int:
        """Deterministic seed derived from all grid parameters via SHA-256."""
        key = f"{self.seed}-{self._reshuffle_count}-{n_points}-{dim}-{scale}-{lower_bound}-{upper_bound}"
        hash_digest = hashlib.sha256(key.encode()).hexdigest()
        return int(hash_digest[:16], 16) % (2 ** 32)

    def get_grid(self, dim: int, scale: int, lower_bound: float = 0., upper_bound: float = 1.) -> RandomGrid:
        store = False
        stored_grid = None
        if lower_bound == 0. and upper_bound == 1.0:
            stored_grid = self._load_grid_if_available(self.used_grids_filepath + f"dim{dim}_scale{scale}.npz")

        if stored_grid is not None:
            print("Loaded_Random Grid")
            return RandomGrid(dim, scale, calculate_num_points(dim=dim, scale=scale), stored_grid, RandomGridRule.CHEBYSHEV, lower_bound, upper_bound, self.seed)
        else:
            store = True

        # Reuse a cached lower-scale grid and extend it if available
        if not scale == 1:
            n_points_scale_minus_one = int(self._multiplier_fun(calculate_num_points(dim=dim, scale=scale - 1)))
            key_scale_minus_one = (n_points_scale_minus_one, dim, scale - 1, lower_bound, upper_bound)
            if key_scale_minus_one in self._cache:
                return self._increase_scale(dim, scale - 1, 1, lower_bound, upper_bound)

        n_points = int(self._multiplier_fun(calculate_num_points(dim=dim, scale=scale)))
        key = (n_points, dim, scale, lower_bound, upper_bound)
        if key not in self._cache:
            np.random.seed(self._generate_seed(*key))
            array = self._generate_chebyshev_points(n_points, dim, lower_bound, upper_bound)
            grid = RandomGrid(dim, scale, n_points, array, RandomGridRule.CHEBYSHEV, lower_bound, upper_bound,
                              self.seed)
            self._cache[key] = grid
            self._current_config = key
        return self._cache[key]

    def _increase_scale(self, dim: int, scale: int, delta: int, lower: float = 0.0, upper: float = 1.0) -> RandomGrid:
        new_total_points = int(self._multiplier_fun(calculate_num_points(dim=dim, scale=scale + delta)))
        n_existing = int(self._multiplier_fun(calculate_num_points(dim=dim, scale=scale)))
        difference = new_total_points - n_existing
        new_key = (new_total_points, dim, scale + delta, lower, upper)
        base_key = (n_existing, dim, scale, lower, upper)

        if base_key not in self._cache:
            raise ValueError(f"Base grid with key {base_key} not found in cache. Cannot increase scale.")

        existing_grid = self._cache[base_key]

        if new_key in self._cache:
            new_grid = self._cache[new_key]
            if self._is_subset(existing_grid.grid, new_grid.grid):
                return new_grid

        # Append freshly drawn points to the existing grid
        np.random.seed(self._generate_seed(*new_key))
        new_points = self._generate_chebyshev_points(difference, dim, lower, upper)
        all_points = np.vstack([existing_grid.grid, new_points])
        n_points = all_points.shape[0]
        new_grid = RandomGrid(dim, scale + delta, n_points, all_points, RandomGridRule.CHEBYSHEV, lower, upper,
                              self.seed)
        self._cache[new_key] = new_grid
        self._current_config = new_key
        return new_grid

    @staticmethod
    def _generate_chebyshev_points(num_points: int, dim: int, lower_bound: float = 0.0, upper_bound: float = 1.0):
        grid_points = np.empty(shape=(num_points, dim))
        for i in range(dim):
            grid_points[:, i] = ChebyshevGridGenerator._sample_chebyshev_univariate(num_points, lower_bound,
                                                                                    upper_bound)
        return grid_points

    @staticmethod
    def _sample_chebyshev_univariate(num_points: int, lower_bound: float = 0.0, upper_bound: float = 1.0) -> np.ndarray:
        """Inverse-transform sampling: CDF is arcsin, inverse is sin."""
        points = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=num_points)
        return ChebyshevGridGenerator._rescale(grid=np.sin(points), lower_bound=lower_bound, upper_bound=upper_bound)

    @staticmethod
    def _rescale(grid: np.ndarray, lower_bound: float, upper_bound: float) -> np.ndarray:
        """Rescale points from [-1, 1] to [lower_bound, upper_bound]."""
        if lower_bound == -1. and upper_bound == 1.:
            return grid
        grid = (grid + 1) / 2
        return grid * (upper_bound - lower_bound) + lower_bound
