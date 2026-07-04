import os

import numpy as np

from algorithm import Algorithm
from basis import BasisGenerator
from function import Function
from grid import Grid
from grid.generator import RandomGridGenerator
from solver import Solver


class LeastSquaresAlgorithm(Algorithm):
    """Unweighted least-squares approximation on a random grid."""

    def __init__(self, basis_generator: BasisGenerator, grid_generator: RandomGridGenerator, solver: Solver):
        super().__init__(
            name="Least_Squares",
            abbr_name="LS",
            basis_generator=basis_generator,
            grid_generator=grid_generator,
            solver=solver
        )

    def fit(self, dim: int, scale: int, f: list[Function], lower: float = 0.0,
            upper: float = 1.0) -> None:
        self.grid = self.grid_generator.get_grid(dim=dim, scale=scale, lower_bound=lower, upper_bound=upper)
        self.basis = self.basis_generator.create_basis(self.grid, scale)

        y = self._calculate_y(f, self.grid)
        self.coeff = self.solver.solve(self.basis.basis, y)
        self.basis = None  # free memory after fitting

    def evaluate(self, grid: Grid, scale: int):
        test_basis = self.basis_generator.create_basis(grid, scale)
        return test_basis.basis @ self.coeff

    def save_coefficients(self, results_path: str, dim: int, scale: int):
        # TODO: Maybe shift this one layer up to the Algorithm base class, since it's almost the same for all
        if self.coeff is None:
            raise ValueError("Coefficients have not been computed yet. Call fit() first.")

        filename = os.path.join("coefficients", f"LS_coefficients_d{dim}_s{scale}.npz")
        path = os.path.join(results_path.replace("results_numerical_experiments.csv", ""), filename)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, coeff=self.coeff)
