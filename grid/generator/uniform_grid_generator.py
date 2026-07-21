import hashlib

import os

import numpy as np
from collections.abc import Callable

from grid.generator.random_grid_generator import RandomGridGenerator
from grid.grid.random_grid import RandomGrid
from grid.rule.random_grid_rule import RandomGridRule
from utils.utils import calculate_num_points


class UniformGridGenerator(RandomGridGenerator):
    """Generate random grids with uniformly distributed points."""

    def __init__(self, seed=42, multiplier_fun: Callable = lambda x: x, store_path:str = None):
        super().__init__("UNIFORM", "UF", seed)
        self._cache = dict()
        self._multiplier_fun = multiplier_fun

        self.used_grids_filepath = store_path

    def _generate_seed(self, n_points: int, dim: int, scale, lower_bound: float = 0., upper_bound: float = 1.) -> int:
        """Deterministic seed derived from all grid parameters via SHA-256."""
        key = f"{self.seed}-{self._reshuffle_count}-{n_points}-{dim}-{scale}-{lower_bound}-{upper_bound}"
        hash_digest = hashlib.sha256(key.encode()).hexdigest()
        return int(hash_digest[:16], 16) % (2 ** 32)

    def get_grid(self, dim: int, scale: int, lower_bound: float = 0., upper_bound: float = 1.) -> RandomGrid:
        stored_grid = None

        n_points = int(self._multiplier_fun(calculate_num_points(dim=dim, scale=scale)))

        if lower_bound == 0. and upper_bound == 1.0:
            stored_grid = self._load_grid_if_available(self.used_grids_filepath + f"dim{dim}_scale{scale}.npz")

        if stored_grid is not None:
            return RandomGrid(dim, scale, n_points, stored_grid,
                              RandomGridRule.UNIFORM, lower_bound, upper_bound, self.seed)

        # Reuse a cached lower-scale grid and extend it if available
        if not scale == 1:
            n_points_scale_minus_one = int(self._multiplier_fun(calculate_num_points(dim=dim, scale=scale - 1)))
            key_scale_minus_one = (n_points_scale_minus_one, dim, scale - 1, lower_bound, upper_bound)
            if key_scale_minus_one in self._cache:
                increased_grid = self._increase_scale(dim, scale - 1, 1, lower_bound, upper_bound)

                # store grid as it was not on the drive yet
                if self.used_grids_filepath is not None:
                    np.savez(self.used_grids_filepath + f"dim{dim}_scale{scale}.npz", increased_grid.grid)
                    print(f"Stored a new grid: {self.name} with dim {dim} and scale {scale}.")
                return increased_grid


        key = (n_points, dim, scale, lower_bound, upper_bound)
        if key not in self._cache:
            np.random.seed(self._generate_seed(*key))
            array = np.random.uniform(low=lower_bound, high=upper_bound, size=(n_points, dim))
            grid = RandomGrid(dim, scale, n_points, array, RandomGridRule.UNIFORM, lower_bound, upper_bound, self.seed)
            self._cache[key] = grid
            self._current_config = key

            # store grid as it was not on the drive yet
            if self.used_grids_filepath is not None:
                np.savez(self.used_grids_filepath + f"dim{dim}_scale{scale}.npz", self._cache[key].grid)
                print(f"Stored a new grid: {self.name} with dim {dim} and scale {scale}.")
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
        new_points = np.random.uniform(low=lower, high=upper, size=(difference, dim))
        all_points = np.vstack([existing_grid.grid, new_points])
        n_points = all_points.shape[0]
        new_grid = RandomGrid(dim, scale + delta, n_points, all_points, RandomGridRule.UNIFORM, lower, upper, self.seed)
        self._cache[new_key] = new_grid
        self._current_config = new_key

        return new_grid
