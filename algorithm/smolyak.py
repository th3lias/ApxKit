import numpy as np

from algorithm.algorithm import Algorithm
from basis.basis_generator import BasisGenerator
from function.f import Function
from grid.generator.rule_grid_generator import RuleGridGenerator
from grid.generator.selection_strategy import SelectionStrategy
from grid.grid.random_grid import RandomGrid
from grid.rule.rule_grid_rule import RuleGridRule
from grid.rule.sparse_grid_type import SparseGridType
from solver.solver import Solver


class SmolyakAlgorithm(Algorithm):
    """Smolyak sparse-grid interpolation via the Tasmanian library."""

    def __init__(self, basis_generator: BasisGenerator, grid_generator: RuleGridGenerator, solver: Solver):
        super().__init__(
            name="Smolyak_Algorithm",
            abbr_name="SA",
            basis_generator=basis_generator,
            grid_generator=grid_generator,
            solver=solver
        )

    def fit(self, dim: int, scale: int, f: list[Function], lower: float = 0.0, upper: float = 1.0):
        self.grid = self.grid_generator.get_grid(input_dim=dim, scale=scale, lower=lower, upper=upper,
                                                  strategy=SelectionStrategy.LEVEL,
                                                  rule=RuleGridRule.CLENSHAW_CURTIS,
                                                  sparse_grid_type=SparseGridType.STANDARD_GLOBAL)
        self.basis = None  # Tasmanian handles the basis internally

        model_values = self._calculate_y(f, self.grid)
        self.grid.load_needed_values(model_values)
        # TODO: Place a "_save_coefficients"-call method somewhere here

    def evaluate(self, grid: RandomGrid):
        test_array = np.array(grid)
        return self.grid.grid.evaluateBatch(test_array)
