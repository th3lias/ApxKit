from fit.fitter import Fitter
from function.f import Function
from function.model.least_squares_model import LeastSquaresModel
from grid.grid.random_grid import RandomGrid
from typing import Union, List

# TODO: Not functional yet

class LeastSquaresFitter(Fitter):  # TODO[Elias] Rework this to fit the new signature
    def __init__(self, dim: int):
        super(LeastSquaresFitter, self).__init__(dim=dim)

    def fit(self, f: Union[Function, List[Function]],
            grid: RandomGrid) -> LeastSquaresModel:  # TODO[Elias] All that should happen here is to create the model and solve the linear least squares problem in a polynomial basis
        """
            f is a function that takes a numpy array of shape (n, d) as input and returns a numpy array of shape (n, 1)
            as output. The input array contains n points in the d-dimensional input space. The output array contains
            the corresponding function values.
        """
        assert self.is_fittable(f), "The function is not fittable by this model."
        return LeastSquaresModel(f=f, dim=self.grid.input_dim, upper=self.grid.upper_bound, lower=self.grid.lower_bound)

#   When we implement the functionality to support wavelet bases, the question is how to design the wavelets.
#   The idea is to take, as usual, wavelets with thin support around the grid points.
#   But this becomes nontrivial when we have random grids.
#   Do we need to save the grid in order to evaluate the interpolant later?
