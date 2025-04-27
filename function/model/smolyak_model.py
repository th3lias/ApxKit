from typing import Callable, Union, List

import numpy as np
from TasmanianSG import TasmanianSparseGrid

from function.model.model import Model


class SmolyakModel(Model):
    def __init__(self, f: Union[Callable, List[Callable]], dim: int, upper: float, lower: float,
                 tasmanian: TasmanianSparseGrid = None,
                 fitted: bool = False):
        """
            In this case, the SmolyakModel contains a 'tasmanian: TasmanianSparseGrid' parameter, which persists
            the grid which was used for fitting. The __call__ method then acts as a wrapper for tasmanian.evaluate.
        """
        super().__init__(f, dim, upper, lower)
        self.tasmanian = tasmanian
        self.fitted = fitted

    def __call__(self, x: np.ndarray):
        """
            Executes 'evaluate' if 'x' is a 1d-numpy array and 'evaluateBatch' if 'x' is a 2d-numpy array.
            !
                Before calling this method, the user should call 'is_fitted' to avoid repeated checking in here.
            !
        """
        x = x.reshape(-1, self.dim)
        return self.tasmanian.evaluate(x) if x.ndim == 1 else self.tasmanian.evaluateBatch(x).T

    def is_fitted(self):
        return self.fitted
