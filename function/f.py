#  Created 2024. (Elias Mindlberger)
from typing import Callable

import numpy as np


class Function(Callable):
    def __init__(self, f: Callable, dim: int, upper: float, lower: float):
        """
            This is a wrapper for callables which represents functions from R^m to R^n.
        """
        self.f = f
        self.dim = dim
        self.upper = upper
        self.lower = lower
    def __call__(self, x: np.ndarray):
        """
            Wrapper function for some numeric Callable.
        """
        return self.f(x)
