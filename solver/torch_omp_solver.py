import math

import numpy as np
import torch

from algorithm.omp import ChebyshevPyKeopsSystem
# from basis.clenshaw_curtis_pykeops_basis_generator import ClenshawCurtisPyKeopsBasisGenerator
from solver.solver import Solver

# TODO: Requires_Grad could probably be turned off everywhere.

class TorchOMPSolver(Solver):
    """
    Orthogonal Matching Pursuit (OMP) Solver using PyTorch + PyKeOps.

    Does not materialize the dense
    system matrix A. Instead, the forward/adjoint/normalization transforms of the
    Chebyshev dictionary (see `ClenshawCurtisPyKeopsBasisGenerator`) are evaluated
    lazily on the fly, which drastically reduces memory usage for large candidate
    dictionaries / point sets. Only the single newly-selected atom is ever built
    explicitly, since the incremental Cholesky update needs the actual atom values.

    Note: unlike the base `Solver.solve(A: np.ndarray, y: np.ndarray)` signature,
    `solve` here takes a `ChebyshevPyKeopsSystem` (see omp.py) in place of
    the dense matrix `A`, since PyKeOps needs the lazy operators rather than a
    materialized array.
    """

    def __init__(self,
                 num_iters: int,
                 tol: float,
                 device: torch.device,
                 name: str = "Torch_OMP_Solver",
                 abbr_name: str = "OMP",
                 ):
        super().__init__(name=name, abbr_name=abbr_name)
        self.num_iters = num_iters
        self.tol = tol

        self.device = device
        self.dtype = torch.float64 if self.device.type == "cpu" else torch.float32
        if self.device.type == "mps":
            print("Warning: MPS backend is not tested. Consider using CPU or CUDA if available.")

    def solve(self,
              system: ChebyshevPyKeopsSystem,
              y: np.ndarray):
        """
        Main interface mapping to the ApxKit Solver pipeline.

        Args:
            system: A `ChebyshevPyKeopsSystem` bundling the lazy forward/adjoint/
                normalization operators together with the candidate indices and the
                (arccos-transformed) sample points. Built by
                `OMPPyKeops._build_system`.
            y_np: Target function values, shape (n_points, n_functions).

        Returns:
            Coefficients, shape (n_indices, n_functions), as a NumPy array.
        """

        y = torch.from_numpy(y).to(dtype=self.dtype, device=self.device)

        num_indices = system.indices.shape[0]
        n_funcs = y.shape[1]
        eps = torch.finfo(self.dtype).eps

        coeff = torch.zeros((num_indices, n_funcs), dtype=self.dtype, device=self.device)

        # Column normalization, computed once via the lazy reduction (no dense matrix needed).
        # normalization() returns squared column norms, so take the square root here to match
        # the 'diag' quantity used by TorchOMPSolver.
        squared_norms = system.normalization(system.indices, system.norm_coeffs).flatten()
        normalization = torch.sqrt(torch.clamp(squared_norms, min=eps))

        for col in range(n_funcs):
            coeff[:, col] = self._omp_solve_single(system, y[:, col:col + 1], normalization).flatten()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return coeff.cpu().numpy()

    def _omp_solve_single(self,
                          system:ChebyshevPyKeopsSystem,
                          b: torch.Tensor,
                          normalization: torch.Tensor) -> torch.Tensor:
        """
        Runs OMP for a single right-hand side.

        Uses the same incremental-Cholesky recursion as `TorchOMPSolver._omp_solve_single`,
        but replaces every dense matrix slice (A[:, S], A.T @ res, ...) by a call to the
        lazy PyKeOps forward/adjoint operators, restricted to the relevant subset of
        candidate indices where needed.

        Args:
            system: `ChebyshevPyKeopsSystem` describing the lazy Chebyshev dictionary.
            b: Target values for one function, shape (n_points, 1).
            normalization: Precomputed dictionary column norms, shape (n_indices,).

        Returns:
            Coefficient vector, shape (n_indices, 1).
        """

        N = b.shape[0]
        M = system.indices.shape[0]
        eps = torch.finfo(self.dtype).eps

        num_iters = min(self.num_iters, M, N // 2)
        selected_indices = torch.zeros(num_iters, device=self.device, dtype=torch.long)
        L = torch.zeros((num_iters, num_iters), device=self.device, dtype=self.dtype)
        rhs = torch.zeros((num_iters, 1), device=self.device, dtype=self.dtype)

        diag = torch.clamp(normalization, min=eps)
        res = b.clone()

        actual_iters = num_iters
        out = None
        for j in range(num_iters):
            # Correlate the current residual with every candidate atom via the lazy adjoint
            # transform. This replaces the dense 'corr = A.T @ res'.
            corr = system.adjoint(res, system.indices, system.norm_coeffs)
            z_2 = torch.abs(corr.flatten() / diag)

            if j > 0:
                z_2[selected_indices[:j]] = -1.0

            # Find maximum correlation index
            max_ind = torch.argmax(z_2)
            selected_indices[j] = max_ind

            # Explicitly build only the single newly-selected column (needed for the
            # Cholesky diagonal update and the right-hand-side dot product below).
            new_col = ClenshawCurtisPyKeopsBasisGenerator.extract_column(system.points_acos,
                                                                         system.indices,
                                                                         system.norm_coeffs,
                                                                         max_ind)

            # Recursive construction of Cholesky decomposition
            if j == 0:
                L[0, 0] = torch.linalg.vector_norm(new_col)
            else:
                # Correlate the new atom with the previously selected atoms only, via the
                # adjoint transform restricted to that subset of indices.
                selected_so_far = selected_indices[:j]
                c = system.adjoint(new_col, system.indices[selected_so_far], system.norm_coeffs[selected_so_far])
                v = torch.linalg.solve_triangular(L[:j, :j], c, upper=False)
                L[j, :j] = v.T
                L[j, j] = torch.sqrt(
                    torch.clamp(diag[max_ind] ** 2 - torch.sum(v ** 2), min=eps ** 2)
                )

            rhs[j, :] = torch.dot(new_col.flatten(), b.flatten())

            # Solve the linear system over the selected columns via Cholesky factor
            out = torch.cholesky_solve(rhs[:j + 1, :], L[:j + 1, :j + 1], upper=False)

            # Reconstruct the residual using only the selected atoms, via the forward
            # transform restricted to that subset of indices.
            selected_so_far = selected_indices[:j + 1]
            res = b - system.forward(out, system.indices[selected_so_far], system.norm_coeffs[selected_so_far])


            residual = torch.linalg.vector_norm(res) / math.sqrt(N)

            if residual < self.tol:
                actual_iters = j + 1
                break

        x = torch.zeros((M, 1), dtype=self.dtype, device=self.device)
        if actual_iters > 0:
            x.flatten()[selected_indices[:actual_iters]] = out[:actual_iters].flatten()

        return x
