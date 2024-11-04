from fit.fitter import Fitter
from function.f import Function
from function.model.least_squares_model import LeastSquaresModel
from grid.grid.random_grid import RandomGrid


class LeastSquaresFitter(Fitter): #TODO[Elias] Rework this to fit the new signature
    def __init__(self, dim: int):
        super(LeastSquaresFitter, self).__init__(dim=dim)

    def fit(self, f: Function, grid: RandomGrid) -> LeastSquaresModel: #TODO[Elias] All that should happen here is to create the model and solve the linear least squares problem in a polynomial basis
        """
            f is a function that takes a numpy array of shape (n, d) as input and returns a numpy array of shape (n, 1)
            as output. The input array contains n points in the d-dimensional input space. The output array contains
            the corresponding function values.
        """
        assert self.is_fittable(f), "The function is not fittable by this model."
        # model_values = self._compute_values(f)
        # self.grid.load_needed_values(model_values.reshape(-1, 1))
        # self.fitted = True
        return LeastSquaresModel(f=f, dim=self.grid.input_dim, upper=self.grid.upper_bound, lower=self.grid.lower_bound)
