#  Created 2024. (Elias Mindlberger)
from typing import Callable

from fit.fitter import Fitter
from function.smolyak_model import SmolyakModel
from function.model.smolyak_model import SmolyakModel
from grid.grid.rule_grid import RuleGrid


class Smolyak(Fitter):
    def __init__(self, grid: RuleGrid):
        super().__init__(grid)
        self.grid = grid  # We assign the grid in here separately to emphasize that it is of type RuleGrid.
        self.fitted = False

    def fit(self, f: Callable) -> SmolyakModel:
        """
            f is a function that takes a numpy array of shape (n, d) as input and returns a numpy array of shape (n, 1)
            as output. The input array contains n points in the d-dimensional input space. The output array contains
            the corresponding function values.
        """
        model_values = self._compute_values(f)
        self.grid.load_needed_values(model_values.reshape(-1, 1))
        self.fitted = True
        return SmolyakModel(f=f, dim=self.grid.input_dim, upper=self.grid.upper_bound, lower=self.grid.lower_bound,
                            tasmanian=self.grid.grid)

    def _compute_values(self, f: Callable):
        """
            f is a function that takes a numpy array of shape (n, d) as input and returns a numpy array of shape either
            (n, ) or (n, 1) as output. The input array contains n points in R^d. The output array contains the
            corresponding function values.
        """
        return f(self.grid.get_needed_points())
