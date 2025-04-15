from function.f import Function
from function.model.model import Model
from grid.grid.grid import Grid
from typing import List, Union


class Fitter:
    """
        Abstract fitter class that defines the interface for all fitters. Currently, we only support fitting functions
        which map to one output, i.e. output_dim=1!
    """

    def __init__(self, dim: int):
        self.dim = dim

    def fit(self, f: Function, grid: Grid, **kwargs) -> Model:
        """
            Fits the function on a given grid of points.
        """
        raise NotImplementedError("The method `fit` must be implemented by the subclass.")

    def is_fittable(self, f: Union[Function, List[Function]]) -> bool:
        """
            Checks if the model is able to compute an approximation for the given function.
        """
        if isinstance(f, Function):
            f = [f]
        for function in f:
            if function.dim != self.dim:
                return False

        return True
