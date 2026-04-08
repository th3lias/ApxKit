import numpy as np
import torch

from solver.solver import Solver


class ConjugateGradient_NE(Solver):
    """
    Conjugate-gradient solver on the normal equations (A^T A x = A^T y).

    Better suited for well-conditioned systems but requires forming A^T A.
    """

    def __init__(self, max_iter: int, tolerance: float, device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__("Conjugate_Gradient_Normal_Equation", "CGNE")
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.device = device
        self.dtype = dtype

    def solve(self, A: np.ndarray, y: np.ndarray) -> np.ndarray:
        A = torch.tensor(A, device=self.device, dtype=self.dtype)
        y = torch.tensor(y, device=self.device, dtype=self.dtype)

        b = A.T @ y
        x = torch.zeros_like(b)
        r = b.clone()
        p = r.clone()
        rs_old = torch.sum(r * r, dim=0)

        for _ in range(self.max_iter):
            Ap = A.T @ (A @ p)
            alpha = rs_old / torch.sum(p * Ap, dim=0)

            x = x + alpha * p
            r = r - alpha * Ap

            rs_new = torch.sum(r * r, dim=0)
            if torch.sqrt(rs_new).max() < self.tolerance:
                break

            p = r + (rs_new / rs_old) * p
            rs_old = rs_new

        return x.cpu().numpy()
