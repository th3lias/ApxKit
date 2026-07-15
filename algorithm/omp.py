import os
from enum import Enum

import numpy as np
import torch
from algorithm.algorithm import Algorithm
from basis.basis_generator import BasisGenerator
from function.f import Function
from grid.generator.grid_generator import GridGenerator
from grid.grid.grid import Grid
from solver.solver import Solver
from collections.abc import Callable
from utils.utils import calculate_num_points


class IndexSetType(Enum):
    HYPERBOLIC = 1
    TOTAL_DEGREE = 2

class OMP(Algorithm):
    """
    OMP Algorithm using customizable multidimensional index sets (Hyperbolic Cross, Full Degree, Total Degree).
    """

    # TODO: Allow to use basis generator by using A = self.basis_generator.get_matrix(self.grid, self._indices, self._norm_coeffs) or similar. Adapt this everywhere.
    def __init__(self, basis_generator: BasisGenerator, grid_generator: GridGenerator, solver: Solver, device: torch.device,
                 hc_bandwidth: int | None, index_set_type:IndexSetType, bandwidth_multiplier_function: Callable,
                 name:str = "Orthogonal_Matching_Pursuit", abbr_name: str = "OMP"):
        """
        Args:
            basis_generator: Framework basis generator.
            grid_generator: Framework grid generator.
            solver: The underlying linear solver.
            device: Computing device ('cpu', 'cuda', 'mps').
            hc_bandwidth: Explicit maximum degree (R / J). If None, it is calculated from grid points.
            index_set_type: Type of candidate index pool: 'hyperbolic', 'total_degree', or 'full_degree'.
            bandwidth_multiplier_function: Scale up the candidate space size beyond the minimal envelope (e.g., 1.5 or 2.0).
        """
        super().__init__(
            name=name,
            abbr_name=abbr_name,
            basis_generator=basis_generator,
            grid_generator=grid_generator,
            solver=solver
        )

        self.device = device
        self.dtype = torch.float64 if self.device.type == "cpu" else torch.float32
        if self.device.type == "mps":
            print("Warning: MPS backend is not tested. Consider using CPU or CUDA if available.")

        self.hc_bandwidth = hc_bandwidth
        self.bandwidth_multiplier_function = bandwidth_multiplier_function
        self.index_set_type = index_set_type

        self._indices = None
        self._norm_coeffs = None
        self._lower = None
        self._upper = None

    def fit(self, dim: int, scale: int, f: list[Function], lower: float = 0.0, upper: float = 1.0):
        self._lower = lower
        self._upper = upper

        # 1. Fetch grid and normalize coordinates to [-1, 1]
        self.grid = self.grid_generator.get_grid(dim=dim, scale=scale, lower_bound=lower, upper_bound=upper)
        points = np.array(self.grid)
        points_norm = 2.0 * (points - lower) / (upper - lower) - 1.0  # transform to [-1, 1] for Chebyshev evaluation

        # 2. Derive Base Bandwidth
        if self.hc_bandwidth is None:
            base_R = OMP._find_hc_bandwidth(dim, calculate_num_points(dim, scale))
        else:
            base_R = self.hc_bandwidth

        # Apply multiplier to scale up the dictionary size
        effective_R = int(np.ceil(self.bandwidth_multiplier_function(base_R)))

        # 3. Generate the selected pool of candidate indices
        if self.index_set_type == IndexSetType.HYPERBOLIC:
            self._indices = self._hyp_cross(dim, effective_R)
        elif self.index_set_type == IndexSetType.TOTAL_DEGREE:
            self._indices = self._total_degree_cross(dim, effective_R)
        else:
            raise ValueError(f"Unknown index_set_type: {self.index_set_type}.")

        self._norm_coeffs = (np.sqrt(2) ** np.clip(self._indices, 0, 1).sum(axis=1)).astype(np.float64)

        # 4. Materialize dense Chebyshev Matrix (pure PyTorch tensor transformation)
        A = self._chebyshev_matrix(points_norm, self._indices, self._norm_coeffs, self.device, self.dtype)
        y = self._calculate_y(f, self.grid)

        # 5. Hand system matrix over to the abstract solver pipeline
        self.coeff = self.solver.solve(A, y)

        del A, y

    def evaluate(self, grid: Grid, scale) -> np.ndarray:
        """Evaluates the fitted Chebyshev approximation model on target validation points."""
        points = np.array(grid)
        points_norm = 2.0 * (points - self._lower) / (self._upper - self._lower) - 1.0
        A_test = self._chebyshev_matrix(points_norm, self._indices, self._norm_coeffs, self.device, self.dtype)
        return A_test @ self.coeff

    def save_coefficients(self, results_path: str, dim: int, scale: int):
        if self.coeff is None:
            raise ValueError("Coefficients have not been computed yet. Call fit() first.")

        # TODO: Make this more dynamic
        name = self.abbr_name
        filename = os.path.join("coefficients", f"{name}_coefficients_d{dim}_s{scale}.npz")
        path = os.path.join(results_path.replace("results_numerical_experiments.csv", ""), filename)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, coeff=self.coeff)

    @staticmethod
    def _hyp_cross(dim: int, R: int) -> np.ndarray:
        """Builds multidimensional Hyperbolic Cross index configurations: prod(max(1, |k_i|)) <= R"""
        if dim == 1:
            return np.arange(0, R + 1, dtype=np.int32).reshape(-1, 1)
        out = []
        for k in range(0, R + 1):
            sub = OMP._hyp_cross(dim - 1, R // max(1, abs(k)))
            block = np.empty((len(sub), dim), dtype=np.int32)
            block[:, 0] = k
            block[:, 1:] = sub
            out.append(block)
        return np.vstack(out)

    @staticmethod
    def _total_degree_cross(dim: int, R: int) -> np.ndarray:
        """Builds a Total Degree space (Linear Cross): sum(k_i) <= R"""
        if dim == 1:
            return np.arange(0, R + 1, dtype=np.int32).reshape(-1, 1)
        out = []
        for k in range(0, R + 1):
            sub = OMP._total_degree_cross(dim - 1, R - k)
            block = np.empty((len(sub), dim), dtype=np.int32)
            block[:, 0] = k
            block[:, 1:] = sub
            out.append(block)
        return np.vstack(out)

    @staticmethod
    # TODO: Utilize basis_generator.create_basis() here
    def _chebyshev_matrix(points_normalized: np.ndarray, indices: np.ndarray, norm_coeffs: np.ndarray,
                          device: torch.device, dtype: torch.dtype, save_memory:bool=True) -> np.ndarray:
        """Materializes the tensor evaluations mapping your multi-index system explicitly.
        If save_memory is True, we loop over the dimension, which creates a smaller intermediate tensor and reduces
        memory usage. If False, we create a larger tensor in one go, which may be faster but uses more memory.

        """
        pts = torch.from_numpy(points_normalized).to(dtype=dtype, device=device)
        idx = torch.from_numpy(indices).to(dtype=dtype, device=device)
        coeffs = torch.from_numpy(norm_coeffs).to(dtype=dtype, device=device)

        eps = torch.finfo(dtype).eps

        if save_memory:
            num_samples = pts.shape[0]
            num_indices = idx.shape[0]
            num_dimensions = pts.shape[1]

            # Initialize the matrix with ones
            mat = torch.ones((num_samples, num_indices), dtype=dtype, device=device)

            for d in range(num_dimensions):
                pts_d = pts[:, d]  # Shape: (num_samples,)
                idx_d = idx[:, d]  # Shape: (num_indices,)

                # Compute 1D Chebyshev component for this dimension
                acos_d = torch.acos(torch.clamp(pts_d, -1.0 + eps, 1.0 - eps))
                angle_d = acos_d[:, None] * idx_d[None, :]  # Shape: (num_samples, num_indices)

                # Multiply in-place into our running product
                mat *= torch.cos(angle_d)

            mat *= coeffs[None, :]

            return mat.cpu().numpy()


        samples_acos = torch.acos(torch.clamp(pts, -1.0 + eps, 1.0 - eps))

        cosine_argument = samples_acos[:, None, :] * idx[None, :, :]
        mat = torch.prod(torch.cos(cosine_argument), dim=2)
        mat *= coeffs[None, :]

        return mat.cpu().numpy()

    @staticmethod
    def _hyp_cross_size(dim: int, R: int) -> int:
        if dim == 1:
            return R + 1
        total = 0
        for k in range(R + 1):
            total += OMP._hyp_cross_size(dim - 1, R // max(1, abs(k)))
        return total

    @staticmethod
    def _find_hc_bandwidth(dim: int, target_m: int) -> int:
        hi = 1
        while OMP._hyp_cross_size(dim, hi) < target_m:
            hi *= 2
        lo = hi // 2
        while lo < hi:
            mid = (lo + hi) // 2
            if OMP._hyp_cross_size(dim, mid) >= target_m:
                hi = mid
            else:
                lo = mid + 1
        return lo
