import numpy as np
from TasmanianSG import TasmanianSparseGrid

from grid.provider.grid_provider import GridProvider
from grid.provider.selection_strategy import SelectionStrategy
from grid.rule import TasmanianGridType
from grid.rule.rule_grid_rule import RuleGridRule
from grid.grid.grid import Grid
from grid.grid.rule_grid import RuleGrid


class RuleGridProvider(GridProvider):
    """
    A grid provider that generates a grid based on a given 1-dimensional grid rule.
    Allowed grid rules can be found in the GridRule enum and it's descendants.
    """

    def __init__(self,
                 input_dim: int,
                 output_dim: int = 1,
                 lower_bound: float = 0.,
                 upper_bound: float = 1.,
                 strategy: SelectionStrategy = SelectionStrategy.LEVEL,
                 rule: RuleGridRule = RuleGridRule.CLENSHAW_CURTIS,
                 tasmanian_type: TasmanianGridType = TasmanianGridType.STANDARD_GLOBAL):
        """
        Takes the same parameters as the GridProvider class and additionally a grid rule.
        :param strategy: The selection strategy
        :param rule: The grid rule to use for generating the grid.
        """
        super().__init__(input_dim, output_dim, lower_bound, upper_bound)
        self.strategy = strategy
        self.rule = rule
        self.tasmanian_type = tasmanian_type

    def generate(self, scale: int) -> RuleGrid:
        #  Set the domain transform -> in most cases this transforms the domain to [0, 1]^d instead of [-1, 1]^d
        domain_transform = self._compute_domain_transform()
        match self.tasmanian_type:
            case TasmanianGridType.STANDARD_GLOBAL:
                grid = self.generate_global_grid(scale)
            case TasmanianGridType.WAVELET:
                grid = self.generate_wavelet_grid(scale)
            case TasmanianGridType.LOCAL_POLYNOMIAL:
                grid = self.generate_local_polynomial_grid(scale)
            case _:
                raise ValueError("Invalid Tasmanian grid type")
        grid.set_domain_transform(domain_transform)
        return grid

    def generate_global_grid(self, scale: int) -> RuleGrid:
        grid = TasmanianSparseGrid()
        grid.makeGlobalGrid(iDimension=self.input_dim, iOutputs=self.output_dim, iDepth=scale,
                            sType=self.strategy.value, sRule=self.rule.value)
        return RuleGrid(self.input_dim, self.output_dim, scale, grid, self.rule, self.lower_bound, self.upper_bound)

    def generate_wavelet_grid(self, scale: int) -> RuleGrid:
        grid = TasmanianSparseGrid()
        grid.makeWaveletGrid(iDimension=self.input_dim, iOutputs=self.output_dim, iDepth=scale)
        return RuleGrid(self.input_dim, self.output_dim, scale, grid, self.rule, self.lower_bound, self.upper_bound)

    def generate_local_polynomial_grid(self, scale: int) -> RuleGrid:
        grid = TasmanianSparseGrid()
        grid.makeLocalPolynomialGrid(iDimension=self.input_dim, iOutputs=self.output_dim, iDepth=scale)
        return RuleGrid(self.input_dim, self.output_dim, scale, grid, self.rule, self.lower_bound, self.upper_bound)

    def increase_scale(self, current_grid: Grid, delta: int) -> RuleGrid:
        """
        Naive implementation of increasing the scale of a grid. Perhaps there is a direct way to do this in Tasmanian.
        """
        old_scale = current_grid.scale
        del current_grid
        return self.generate(old_scale + delta)

    def _compute_domain_transform(self):
        """
            Compute the domain transformation for the grid.
            According to TasmanianSG documentation:
            llfTransform: a 2-D numpy.ndarray of size iDimension X 2
                  transform specifies the lower and upper bound
                  of the domain in each direction.
        """
        domain = np.array([[self.lower_bound, self.upper_bound]])
        return np.full((self.input_dim, 2), domain)
