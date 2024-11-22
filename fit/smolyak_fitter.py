import numpy as np
from numpy import dtype
from typing import Union, List

from fit.fitter import Fitter
from function.f import Function
from function.model.smolyak_model import SmolyakModel
from grid.grid.rule_grid import RuleGrid
from grid.provider.rule_grid_provider import RuleGridProvider


class SmolyakFitter(Fitter):
    def __init__(self, input_dim: int):
        super(SmolyakFitter, self).__init__(dim=input_dim)

    def fit(self, f: Union[Function, List[Function]], grid: RuleGrid) -> SmolyakModel:
        """
            f is a function or a list of such functions that takes a numpy array of shape (n, d) as input and returns a
            numpy array of shape (n, 1) as output. The input array contains n points in the d-dimensional input space.
            The output array contains the corresponding function values.
        """

        if isinstance(f, Function):
            f = [f]

        assert self.is_fittable(f), "At least one of the provided functions is not fittable by this model."
        model_values = self._calculate_y(f, grid)
        grid.load_needed_values(model_values)
        return SmolyakModel(f=f, dim=grid.input_dim, upper=grid.upper_bound, lower=grid.lower_bound,
                            tasmanian=grid.grid)

    def _calculate_y(self, f: List[Function], grid: RuleGrid) -> np.ndarray:
        """
        Calculates the function values for the given grid points
        :param f: function that need to be approximated on the same points
        :param grid: RuleGrid
        :return: function values
        """

        return np.hstack([func(grid.get_needed_points()).reshape(-1, 1) for func in f])


if __name__ == '__main__':
    # Dummy test for the fitter
    fun = Function(lambda x: x[0] + x[1], 2)
    rule_grid = RuleGridProvider(2).generate(2)
    fitter = SmolyakFitter(2)
    model = fitter.fit(fun, rule_grid)
    test_array = np.array([[1, 2], [3, 4], [5, 7], [10, 10]], dtype=dtype(float))  # only works when dtype is set
    print(test_array.shape)
    print(model(test_array))  # the model assumes dimensions (n, input_dim)
    print(fun(test_array.T))  # a usual numeric function assumes dimensions (input_dim, n)
