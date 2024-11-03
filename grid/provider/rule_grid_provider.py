from TasmanianSG import TasmanianSparseGrid

from grid.provider.grid_provider import GridProvider
from grid.provider.selection_strategy import SelectionStrategy
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
                 rule: RuleGridRule = RuleGridRule.CLENSHAW_CURTIS):
        """
        Takes the same parameters as the GridProvider class and additionally a grid rule.
        :param strategy: The selection strategy
        :param rule: The grid rule to use for generating the grid.
        """
        super().__init__(input_dim, output_dim, lower_bound, upper_bound)
        self.strategy = strategy
        self.rule = rule

    def generate(self, scale: int) -> RuleGrid:
        grid = TasmanianSparseGrid()
        grid.makeGlobalGrid(iDimension=self.input_dim, iOutputs=self.output_dim, iDepth=scale,
                            sType=self.strategy.value, sRule=self.rule.value)
        return RuleGrid(self.input_dim, self.output_dim, scale, grid, self.rule, self.lower_bound, self.upper_bound)

    def increase_scale(self, current_grid: Grid, delta: int) -> RuleGrid:
        """
        Naive implementation of increasing the scale of a grid. Perhaps there is a direct way to do this in Tasmanian.
        """
        old_scale = current_grid.scale
        del current_grid
        return self.generate(old_scale + delta)
        # TODO[Jakob] Check whether we can increase the scale without generating a new one
