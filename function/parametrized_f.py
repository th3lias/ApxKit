from collections.abc import Callable
import numpy as np

from function.f import Function


class ParametrizedFunction(Function):
    """Function with additional parameter vectors c and w."""

    def __init__(self, f: Callable, dim: int, c: np.ndarray, w: np.ndarray, upper: float = 1.0, lower: float = 0.0,
                 name: str = "unknown"):
        super().__init__(f, dim, upper, lower, name)
        self.c = c
        self.w = w
