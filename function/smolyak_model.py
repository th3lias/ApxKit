from typing import Callable

import numpy as np
from TasmanianSG import TasmanianSparseGrid

from function.model import Model


class SmolyakModel(Model):
    def __init__(self, f: Callable, dim: int, upper: float, lower: float, tasmanian: TasmanianSparseGrid = None):
        """
            In this case, the SmolyakModel contains a 'tasmanian: TasmanianSparseGrid' parameter, which persists
            the grid which was used for fitting. The __call__ method then acts as a wrapper for tasmanian.evaluate.
        """
        super().__init__(f, dim, upper, lower)
        self.tasmanian = tasmanian

    def __call__(self, x: np.ndarray):
        """
            Executes 'evaluate' if 'x' is a 1d-numpy array and 'evaluateBatch' if 'x' is a 2d-numpy array.
        """
        assert self.tasmanian is not None, "Tasmanian Grid not found, has this model been fitted?"
        return self.tasmanian.evaluate(x) if x.ndim == 1 else self.tasmanian.evaluateBatch(x)
