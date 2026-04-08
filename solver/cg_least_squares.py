import numpy as np
import torch

from solver.solver import Solver


class ConjugateGradient_LS(Solver):
    """
    Conjugate-gradient least-squares (CGLS) solver.

    Solves Az = y in the least-squares sense without forming A^T A,
    which is advantageous for ill-conditioned systems.
    """

    def __init__(self, max_iter: int, tolerance: float, device: torch.device, dtype: torch.dtype = torch.float32):
        super().__init__("Conjugate_Gradient_Least_Squares", "CGLS")
        self.max_iter = max_iter
        self.tolerance = tolerance
        self.device = device
        self.dtype = dtype

    def solve(self, A: np.ndarray, y: np.ndarray) -> np.ndarray:
        A = torch.tensor(A, device=self.device, dtype=self.dtype)
        y = torch.tensor(y, device=self.device, dtype=self.dtype)

        m, n = A.shape
        _, k = y.shape

        x = torch.zeros((n, k), device=self.device, dtype=self.dtype)
        d = y - A @ x          # residual in observation space
        p = A.T @ d             # search direction in coefficient space
        s = p.clone()

        for i in range(self.max_iter):
            if torch.norm(s, dim=0).max().float() < self.tolerance:
                break

            q = A @ p
            numerator = torch.sum(s * s, dim=0)
            denominator = torch.sum(q * q, dim=0)
            alpha = numerator / denominator

            x.add_(alpha * p)
            d.sub_(alpha * q)
            s = A.T @ d

            beta = torch.sum(s * s, dim=0) / numerator
            p.mul_(beta).add_(s)

        return x.cpu().numpy()
