import math
import torch
from solver.solver import Solver
import numpy as np


class TorchOMPSolver(Solver):
    """
    Orthogonal Matching Pursuit (OMP) Solver using PyTorch for GPU acceleration.
    Solves Az = y column-by-column using incremental Cholesky updates.
    """

    def __init__(self, num_iters: int, tol: float, device: torch.device,
                 name: str = "Torch_OMP_Solver", abbr_name: str = "OMP"):
        super().__init__(name=name, abbr_name=abbr_name)
        self.num_iters = num_iters
        self.tol = tol


        self.device = device
        self.dtype = torch.float64 if self.device.type == "cpu" else torch.float32
        if self.device.type == "mps":
            print("Warning: MPS backend is not tested. Consider using CPU or CUDA if available.")

    def solve(self, A_np, y_np):
        """
        Main interface mapping to the ApxKit Solver pipeline.
        Converts NumPy arrays to PyTorch float32 tensors, executes OMP on the target device,
        and returns coefficients back as a NumPy array.
        """
        # Convert arrays to 32-bit float tensors on the requested device
        A = torch.from_numpy(A_np).to(dtype=self.dtype, device=self.device)
        y = torch.from_numpy(y_np).to(dtype=self.dtype, device=self.device)

        M = A.shape[1]
        n_funcs = y.shape[1]

        coeff = torch.zeros((M, n_funcs), dtype=self.dtype, device=self.device)

        # Precompute the column normalization vector to match the reference setup
        # Equates to the 'normalization' parameter in your original function
        normalization = torch.sqrt((A ** 2).sum(dim=0, keepdim=True))

        # TODO: Parallelize this loop using multiprocessing
        for col in range(n_funcs):
            coeff[:, col] = self._omp_solve_single(A, y[:, col:col + 1], normalization).flatten()

        # delete cuda memory
        del A, y, normalization
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return coeff.cpu().numpy()

    def _omp_solve_single(self, A: torch.Tensor, b: torch.Tensor, normalization: torch.Tensor) -> torch.Tensor:
        N, M = A.shape
        eps = torch.finfo(A.dtype).eps

        num_iters = min(self.num_iters, M, N // 2) # TODO: Change this. It is now hardcoded to the same number as in LS (E(q,d))
        selected_indices = torch.zeros(num_iters, device=self.device, dtype=torch.long)
        L = torch.zeros((num_iters, num_iters), device=self.device, dtype=self.dtype)
        rhs = torch.zeros((num_iters, 1), device=self.device, dtype=self.dtype)

        diag = torch.clamp(normalization.flatten(), min=eps)
        z_2 = torch.empty_like(diag)
        res = b.clone()

        actual_iters = num_iters
        for j in range(num_iters):
            # Compute and mask normalized correlations
            # This replicates: torch.abs(AT(res).flatten() / diag, out=z_2)
            corr = A.T @ res
            torch.abs(corr.flatten() / diag, out=z_2)

            if j > 0:
                z_2[selected_indices[:j]] = -1.0

            # Find maximum correlation index
            max_ind = torch.argmax(z_2)
            selected_indices[j] = max_ind

            # Extract the newly selected column (acting as col_extractor)
            new_col = A[:, max_ind:max_ind + 1]

            # Recursive construction of Cholesky decomposition
            if j == 0:
                L[0, 0] = torch.linalg.vector_norm(new_col)
            else:
                # Replicates the sub-matrix correlation logic using matrix multiplication
                c = A[:, selected_indices[:j]].T @ new_col
                v = torch.linalg.solve_triangular(L[:j, :j], c, upper=False)
                L[j, :j] = v.T
                L[j, j] = torch.sqrt(
                    torch.clamp(diag[max_ind] ** 2 - torch.sum(v ** 2), min=eps ** 2)
                )

            rhs[j, :] = torch.dot(new_col.flatten(), b.flatten())

            # Solve the linear system over the selected columns via Cholesky factor
            out = torch.cholesky_solve(rhs[:j + 1, :], L[:j + 1, :j + 1], upper=False)
            res = b - A[:, selected_indices[:j + 1]] @ out

            residual = torch.linalg.vector_norm(res) / math.sqrt(N)
            # if j % 100 == 0:
            #     print(f"Iteration: {j + 1}  Residual: {residual.item():.6e}")

            if residual < self.tol:
                actual_iters = j + 1
                break

        x = torch.zeros((M, 1), dtype=self.dtype, device=self.device)
        if actual_iters > 0:
            x.flatten()[selected_indices[:actual_iters]] = out[:actual_iters].flatten()

        return x
