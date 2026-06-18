import numpy as np

from algorithm.algorithm import Algorithm
from basis.basis_generator import BasisGenerator
from function.f import Function
from grid.generator.grid_generator import GridGenerator
from grid.grid.grid import Grid
from solver.solver import Solver


class WeightedLeastSquaresAlgorithm(Algorithm):
    """
    Importance-weighted least-squares approximation.

    Points are drawn from the Chebyshev (arcsine) density; weighting
    corrects the empirical sum to approximate the L²(Lebesgue) norm.
    """

    def __init__(self, basis_generator: BasisGenerator, grid_generator: GridGenerator, solver: Solver):
        super().__init__(
            name="Weighted_Least_Squares",
            abbr_name="wLS",
            basis_generator=basis_generator,
            grid_generator=grid_generator,
            solver=solver
        )

    def fit(self, dim: int, scale: int, f: list[Function], lower: float = 0.0,
            upper: float = 1.0) -> None:
        self.grid = self.grid_generator.get_grid(dim=dim, scale=scale, lower_bound=lower, upper_bound=upper)
        self.basis = self.basis_generator.create_basis(self.grid)

        y = self._calculate_y(f, self.grid)
        weight = self._get_weights_for_weighted_ls()

        # Apply √(1/μ) weighting to both sides of Az = b
        x_poly = self.basis.basis * weight[:, np.newaxis]
        y_prime = y * weight[:, np.newaxis]

        self.coeff = self.solver.solve(x_poly, y_prime)
        self.basis = None  # free memory after fitting

    def evaluate(self, grid: Grid):
        test_basis = self.basis_generator.create_basis(grid)
        return test_basis.basis @ self.coeff

    def _get_weights_for_weighted_ls(self):
        """
        Compute importance-sampling weights √(1/μ(x)) for Chebyshev-distributed points.

        The Chebyshev density is μ(x) = ∏ 1/(π√(1−xᵢ²)).  The square-root
        is taken because both the basis matrix and the RHS are multiplied by it.
        """
        points = np.array(self.grid)
        # Map [0,1]^d → [-1,1]^d if needed
        if self.grid.lower_bound == 0.0 and self.grid.upper_bound == 1.0:
            points = 2 * points - 1
        elif self.grid.lower_bound != -1.0 or self.grid.upper_bound != 1.0:
            raise ValueError("The Chebyshev rule only supports the range [-1, 1] or [0, 1]")
        return np.sqrt(np.prod(np.pi / np.polynomial.chebyshev.chebweight(points), axis=1))
