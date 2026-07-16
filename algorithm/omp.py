import os
from collections.abc import Callable
from dataclasses import dataclass
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

# TODO: deepinv as solver??
# TODO: Go through all files and format them inkl. removal of non-used imported libraries

class IndexSetType(Enum):
    HYPERBOLIC = 1
    TOTAL_DEGREE = 2

@dataclass
class ChebyshevPyKeopsSystem:
    """
    Bundles everything the PyKeOps-based OMP solver needs to evaluate the Chebyshev
    dictionary lazily, without ever materializing the dense (n_points x n_indices) matrix.

    Attributes:
        forward: Lazy KeOps reduction implementing A @ x (coefficients -> function values).
            Called as forward(x, indices, norm_coeffs).
        adjoint: Lazy KeOps reduction implementing A.T @ x (function values -> coefficients).
            Called as adjoint(x, indices, norm_coeffs).
        normalization: Lazy KeOps reduction returning sum_i A_ij^2 for every candidate
            column j. Called as normalization(indices, norm_coeffs).
        points_acos: arccos of the normalized sample points, shape (n_points, dim). Needed
            to explicitly extract single dictionary columns during the Cholesky update.
        indices: Candidate multi-indices, shape (n_indices, dim).
        norm_coeffs: Chebyshev normalization prefactors, shape (n_indices, 1).
    """
    forward: Callable
    adjoint: Callable
    normalization: Callable
    points_acos: torch.Tensor
    indices: torch.Tensor
    norm_coeffs: torch.Tensor


class OMP(Algorithm):
    """
    OMP Algorithm using a PyKeOps-backed lazy Chebyshev dictionary.

    Functionally equivalent to `OMP`, but never materializes the dense
    (n_points x n_indices) Chebyshev matrix. Instead, the forward/adjoint transforms
    are evaluated on the fly by PyKeOps,
    trading a bit of extra compute for a large reduction in memory usage. This makes
    it feasible to use much larger candidate dictionaries and/or point sets than the
    dense `OMP` variant.
    """

    def __init__(self,
                 basis_generator: BasisGenerator,
                 grid_generator: GridGenerator,
                 solver: Solver,
                 device: torch.device,
                 hc_bandwidth: int | None,
                 index_set_type:IndexSetType,
                 bandwidth_multiplier_function: Callable,
                 name:str = "Orthogonal_Matching_Pursuit",
                 abbr_name: str = "OMP"):
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

        self._indices = None # np.ndarray, kept around for saving/inspection
        self._indices_t = None  # torch tensor version, used by the PyKeOps operators (t stands for tensor)
        self._norm_coeffs_t = None  # torch tensor version, shape (n_indices, 1) (t stands for tensor)
        self._dim = None
        self._lower = None
        self._upper = None

    def fit(self,
            dim: int,
            scale: int,
            f: list[Function],
            lower: float = 0.0,
            upper: float = 1.0):

        self._lower = lower
        self._upper = upper
        self._dim = dim

        # 1. Fetch grid and calculate arccos of the [-1, 1]-normalized coordinates
        self.grid = self.grid_generator.get_grid(dim=dim, scale=scale, lower_bound=lower, upper_bound=upper)
        points_acos = self._points_to_acos(np.array(self.grid), lower, upper) # TODO: Check if correctly transformed to [-1, 1] for Chebyshev evaluation

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

        norm_coeffs = (np.sqrt(2) ** np.clip(self._indices, 0, 1).sum(axis=1)).astype(np.float64)

        self._indices_t = torch.from_numpy(self._indices).to(dtype=self.dtype, device=self.device)
        self._norm_coeffs_t = torch.from_numpy(norm_coeffs).to(dtype=self.dtype, device=self.device).unsqueeze(1)

        # 4. Build the lazy Chebyshev system (no dense matrix is ever created here)
        system = self._build_system(points_acos)
        y = self._calculate_y(f, self.grid)

        # 5. Hand system matrix over to the abstract solver pipeline
        self.coeff = self.solver.solve(system, y)

    def evaluate(self,
                 grid: Grid,
                 scale) -> np.ndarray:
        """Evaluates the fitted Chebyshev approximation model on target validation points."""

        points_acos = self._points_to_acos(np.array(grid), self._lower, self._upper)
        system = self._build_system(points_acos)

        coeff_t = torch.from_numpy(self.coeff).to(dtype=self.dtype, device=self.device)
        values = system.forward(coeff_t, self._indices_t, self._norm_coeffs_t) # forward transform: A @ coeff -> values
        return values.cpu().numpy()

    def save_coefficients(self, results_path: str, dim: int, scale: int):
        if self.coeff is None:
            raise ValueError("Coefficients have not been computed yet. Call fit() first.")

        # TODO: Make this more dynamic
        name = self.abbr_name
        filename = os.path.join("coefficients", f"{name}_coefficients_d{dim}_s{scale}.npz")
        path = os.path.join(results_path.replace("results_numerical_experiments.csv", ""), filename)

        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(path, coeff=self.coeff)

    def _points_to_acos(self, points: np.ndarray, lower: float, upper: float) -> torch.Tensor:
        """Normalizes physical grid points to [-1, 1] and returns their arccos, ready to be
        fed into the Chebyshev PyKeOps operators."""
        points_norm = 2.0 * (points - lower) / (upper - lower) - 1.0
        pts = torch.from_numpy(points_norm).to(dtype=self.dtype, device=self.device)
        eps = torch.finfo(self.dtype).eps
        return torch.acos(torch.clamp(pts, -1.0 + eps, 1.0 - eps))

    def _build_system(self, points_acos: torch.Tensor) -> ChebyshevPyKeopsSystem:
        """Builds the lazy forward/adjoint/normalization operators bound to the given
        (arccos-transformed) sample points, and bundles them together with the current
        candidate indices into a ChebyshevPyKeopsSystem."""
        forward = self.basis_generator.build_forward_operator(points_acos, self._dim)
        adjoint = self.basis_generator.build_adjoint_operator(points_acos, self._dim)
        normalization = self.basis_generator.build_normalization_operator(points_acos, self._dim)

        return ChebyshevPyKeopsSystem(
            forward=forward,
            adjoint=adjoint,
            normalization=normalization,
            points_acos=points_acos,
            indices=self._indices_t,
            norm_coeffs=self._norm_coeffs_t,
        )

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
                          device: torch.device, dtype: torch.dtype, save_memory: bool = True) -> np.ndarray:
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
