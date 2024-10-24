from typing import Callable

import numpy as np

from function.f import Function


class ParametrizedFunction(Function):
    """
        Callable wrapper with parameters c and w.
    """
    def __init__(self, f: Callable, dim: int, w: np.ndarray, c: np.ndarray, upper: float = 1.0, lower: float = 0.0,
                 name: str = "unknown"):
        super().__init__(f, dim, upper, lower)
        self.w = w
        self.c = c
        self.name = name
