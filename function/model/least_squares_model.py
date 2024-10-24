from typing import Callable

import numpy as np

from function.model.model import Model


class LeastSquaresModel(Model):
    def __init__(self, f: Callable, dim: int, upper: float, lower: float):
        super().__init__(f, dim, upper, lower)

    def __call__(self, x: np.ndarray):
        pass
