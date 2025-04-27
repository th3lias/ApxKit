import scipy

from fit.fitter import Fitter
from function.f import Function
from function.model.least_squares_model import LeastSquaresModel
from grid.grid.random_grid import RandomGrid
from typing import Union, List, Callable
from fit.basis_f import grid_basis_gaussian, compute_epsilon, evaluate_gaussian_rbf

from grid.provider.random_grid_provider import RandomGridProvider


class LeastSquaresFitter(Fitter):
    def __init__(self, dim: int):
        super(LeastSquaresFitter, self).__init__(dim=dim)
        self.basis_f = None

    def set_basis_f(self, basis_f: Callable):
        """
            Takes a Callable with parameters x, grid, beta and possibly some kwargs.
            Specifies how the basis is computed on its grid and coefficients.
        """
        self.basis_f = basis_f

    def fit(self, f: Union[Function, List[Function]],
            grid: RandomGrid, **kwargs) -> LeastSquaresModel:
        """
            f is a function that takes a numpy array of shape (n, d) as input and returns a numpy array of shape (n, 1)
            as output. The input array contains n points in the d-dimensional input space. The output array contains
            the corresponding function values.
        """
        assert self.is_fittable(f), "The function is not fittable by this model."
        assert self.basis_f is not None, "The basis function is not set."
        vandermonde = self.basis_f(grid.grid, grid.grid, **kwargs)
        y = f(grid.grid)
        # TODO: Either make it variable or remove all other possibilities of method selection as soon as we only keep lstsq with GELSY driver
        result, _, _, _ = scipy.linalg.lstsq(vandermonde, y)
        model = LeastSquaresModel(f=f, dim=grid.input_dim, upper=grid.upper_bound, lower=grid.lower_bound)
        model.set_solution(result)
        model.set_grid(grid.grid)
        model.set_kwargs(**kwargs)
        return model


#   When we implement the functionality to support wavelet bases, the question is how to design the wavelets.
#   The idea is to take, as usual, wavelets with thin support around the grid points.
#   But this becomes nontrivial when we have random grids.
#   Do we need to save the grid in order to evaluate the interpolant later?

#  Experimental
if __name__ == "__main__":
    least_squares_fitter = LeastSquaresFitter(dim=2)
    grid_provider = RandomGridProvider(input_dim=2, output_dim=1, lower_bound=0., upper_bound=1.)
    grid = grid_provider.generate(scale=2)
    f = Function(lambda x: x[:, 0] ** 2 + x[:, 1] ** 2, dim=2)
    least_squares_fitter.set_basis_f(grid_basis_gaussian)
    eps = compute_epsilon(grid_centers=grid.grid, c=1.0)
    lsq_model = least_squares_fitter.fit(f=f, grid=grid, epsilon=eps)
    lsq_model.set_evaluate(evaluate_gaussian_rbf)
    print(lsq_model.beta)
    test_grid = grid_provider.generate(scale=2)
    print(test_grid.grid.shape)
    result = lsq_model(test_grid.grid)
    print(result)
