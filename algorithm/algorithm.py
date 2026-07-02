from __future__ import annotations

import numpy as np

from basis.basis_generator import BasisGenerator
from function.f import Function
from grid.generator.grid_generator import GridGenerator
from grid.grid.grid import Grid
from solver.solver import Solver


class Algorithm:
    """
    Base class for approximation algorithms.

    An algorithm combines a BasisGenerator, GridGenerator, and Solver to
    construct a polynomial (or other) approximation of one or more functions.
    """
    def __init__(self,
                 name: str,
                 abbr_name: str,
                 basis_generator: BasisGenerator,
                 grid_generator: GridGenerator | None,
                 solver: Solver):
        self.abbr_name = abbr_name
        self.name = name
        self.basis_generator = basis_generator
        self.grid_generator = grid_generator
        self.solver = solver

        self.grid = None
        self.basis = None
        self.coeff = None

    def fit(self, dim: int, scale: int, f: list[Function], lower: float = 0.0, upper: float = 1.0):
        raise NotImplementedError("Subclasses should implement this method")

    def evaluate(self, grid: Grid):
        """Evaluate the fitted approximant at the given grid points."""
        raise NotImplementedError("Subclasses should implement this method")

    def get_n_points(self):
        if self.grid is None:
            raise ValueError("Grid has not been generated yet. Call fit() first.")
        return self.grid.get_num_points()

    def save_coefficients(self, results_path:str, dim:int, scale:int):
        raise NotImplementedError("Subclasses should implement this method")

    @staticmethod
    def _calculate_y(f: list[Function], grid: Grid):
        """Evaluate functions at grid points, returning an (n_points, n_functions) array."""
        data = np.array(grid)
        return np.column_stack([f_i(data) for f_i in f])
