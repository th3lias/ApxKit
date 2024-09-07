#  Created 2024. (Elias Mindlberger)
from typing import Callable

import numpy as np

from function.f import Function


class ParametrizedFunction(Function):
    """
        Callable wrapper with parameters c and w.
    """
    def __init__(self, f: Callable, dim: int, upper: float, lower: float, w: np.ndarray, c: np.ndarray):
        super().__init__(f, dim, upper, lower)
        self.w = w
        self.c = c

    def __call__(self, *args, **kwargs):
        super.__call__(*args, **kwargs)
