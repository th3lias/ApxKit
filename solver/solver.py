import numpy as np


class Solver:
    """Abstract base for linear-system solvers used by approximation algorithms."""

    def __init__(self, name: str, abbr_name: str):
        self.name = name
        self.abbr_name = abbr_name

    def solve(self, A: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Solve the (possibly overdetermined) system Az = y for z."""
        raise NotImplementedError("Subclasses should implement this method")
