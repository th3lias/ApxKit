import numpy as np
from TasmanianSG import TasmanianSparseGrid

from grid.generator.grid_generator import GridGenerator
from grid.generator.selection_strategy import SelectionStrategy
from grid.rule.sparse_grid_type import SparseGridType
from grid.rule.rule_grid_rule import RuleGridRule
from grid.grid.rule_grid import RuleGrid


class RuleGridGenerator(GridGenerator):
    """
    Deterministic grid generator backed by Tasmanian sparse grids.

    Supports global, wavelet, and local-polynomial grid types.
    """

    def __init__(self, output_dim: int = 1):
        super().__init__("SPARSE", "SPARSE")
        self.output_dim = output_dim

    def get_grid(self, input_dim: int, scale: int, lower: float, upper: float, strategy: SelectionStrategy,
                 rule: RuleGridRule, sparse_grid_type: SparseGridType) -> RuleGrid:
        domain_transform = self._compute_domain_transform(lower, upper, input_dim)

        match sparse_grid_type:
            case SparseGridType.STANDARD_GLOBAL:
                grid = self.generate_global_grid(input_dim, scale, lower, upper, strategy, rule)
            case SparseGridType.WAVELET:
                grid = self.generate_wavelet_grid(input_dim, scale, lower, upper, rule)
            case SparseGridType.LOCAL_POLYNOMIAL:
                grid = self.generate_local_polynomial_grid(input_dim, scale, lower, upper, rule)
            case _:
                raise ValueError("Invalid Tasmanian grid type")

        grid.set_domain_transform(domain_transform)
        return grid

    def generate_global_grid(self, input_dim: int, scale: int, lower: float, upper: float,
                             strategy: SelectionStrategy = SelectionStrategy.LEVEL,
                             rule: RuleGridRule = RuleGridRule.CLENSHAW_CURTIS) -> RuleGrid:
        grid = TasmanianSparseGrid()
        grid.makeGlobalGrid(iDimension=input_dim, iOutputs=self.output_dim, iDepth=scale,
                            sType=strategy.value, sRule=rule.value)
        return RuleGrid(input_dim, self.output_dim, scale, grid, rule, lower, upper)

    def generate_wavelet_grid(self, input_dim: int, scale: int, lower: float, upper: float,
                              rule: RuleGridRule = RuleGridRule.CLENSHAW_CURTIS) -> RuleGrid:
        grid = TasmanianSparseGrid()
        grid.makeWaveletGrid(iDimension=input_dim, iOutputs=self.output_dim, iDepth=scale)
        return RuleGrid(input_dim, self.output_dim, scale, grid, rule, lower, upper)

    def generate_local_polynomial_grid(self, input_dim: int, scale: int, lower: float, upper: float,
                                       rule: RuleGridRule = RuleGridRule.CLENSHAW_CURTIS) -> RuleGrid:
        grid = TasmanianSparseGrid()
        grid.makeLocalPolynomialGrid(iDimension=input_dim, iOutputs=self.output_dim, iDepth=scale)
        return RuleGrid(input_dim, self.output_dim, scale, grid, rule, lower, upper)

    @staticmethod
    def _compute_domain_transform(lower: float, upper: float, dim: int):
        """Build the (dim, 2) domain-transform array for TasmanianSG."""
        domain = np.array([[lower, upper]])
        return np.full((dim, 2), domain)
