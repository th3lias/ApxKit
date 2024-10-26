from typing import Callable

from fit.fitter import Fitter
from function.model.least_squares_model import LeastSquaresModel
from grid.grid.grid import Grid


class LeastSquares(Fitter):
    def __init__(self, grid: Grid):
        super().__init__(grid)

    def fit(self, f: Callable) -> LeastSquaresModel:
        """
            f is a function that takes a numpy array of shape (n, d) as input and returns a numpy array of shape (n, 1)
            as output. The input array contains n points in the d-dimensional input space. The output array contains
            the corresponding function values.
        """
        # model_values = self._compute_values(f)
        # self.grid.load_needed_values(model_values.reshape(-1, 1))
        # self.fitted = True
        return LeastSquaresModel(f=f, dim=self.grid.input_dim, upper=self.grid.upper_bound, lower=self.grid.lower_bound)

    def _compute_values(self, f: Callable):
        """
            f is a function that takes a numpy array of shape (n, d) as input and returns a numpy array of shape either
            (n, ) or (n, 1) as output. The input array contains n points in R^d. The output array contains the
            corresponding function values.
        """
        return f(self.grid.get_num_points())
