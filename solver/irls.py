import numpy as np
from numpy import array, diag, dot, maximum, repeat
from numpy.linalg import inv

from solver.solver import Solver


class IterativeReweightedLeastSquares(Solver):
    """
    Iteratively Reweighted Least Squares (IRLS) solver.

    Based on https://github.com/aehaynes/IRLS/blob/master/irls.py
    """

    def __init__(self, max_iter: int, tolerance: float, d: float):
        super().__init__("Iterative_Reweighted_Least_Squares", "IRLS")
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.d = d  # small constant to avoid division by zero

    def solve(self, A: np.ndarray, y: np.ndarray) -> np.ndarray:
        n, p = A.shape
        delta = array(repeat(self.d, n)).reshape(1, n)
        w = repeat(1, n)
        W = diag(w)
        B = dot(inv(A.T.dot(W).dot(A)), (A.T.dot(W).dot(y)))

        for _ in range(self.max_iter):
            _B = B
            _w = abs(y - A.dot(B)).T
            w = float(1) / maximum(delta, _w)
            W = diag(w[0])
            B = dot(inv(A.T.dot(W).dot(A)), (A.T.dot(W).dot(y)))
            if np.sum(abs(B - _B)) < self.tolerance:
                return B
        return B
