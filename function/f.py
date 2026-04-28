from collections.abc import Callable

import numpy as np


class Function(Callable):
    """Callable wrapper representing a function f: R^dim → R."""

    def __init__(self, f: Callable, dim: int, upper: float = 1.0, lower: float = 0.0, name: str = "unknown"):
        self.f = f
        self.dim = dim
        self.upper = upper
        self.lower = lower
        self.name = name

    def __call__(self, x: np.ndarray):
        return self.f(x)
