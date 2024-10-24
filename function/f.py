#  Created 2024. (Elias Mindlberger)
from typing import Callable

import numpy as np


class Function(Callable):
    """
        This is a wrapper for callables which represents functions from R^m to R^n.
    """
    def __init__(self, f: Callable, dim: int, upper: float = 1.0, lower: float = 0.0):
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
