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

    def fit(self, dim: int, scale: int, f: Function | list[Function], lower: float = 0.0,
            upper: float = 1.0) -> None:
        self.grid = self.grid_generator.get_grid(dim=dim, scale=scale, lower_bound=lower, upper_bound=upper)
        self.basis = self.basis_generator.create_basis(self.grid)

        if isinstance(f, Function):
            f = [f]

        y = self._calculate_y(f, self.grid)
        self.coeff = self.solver.solve(self.basis.basis, y)
        self.basis = None  # free memory after fitting

    def evaluate(self, grid: Grid):
        test_basis = self.basis_generator.create_basis(grid)
        return test_basis.basis @ self.coeff
