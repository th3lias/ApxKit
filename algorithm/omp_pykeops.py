import os

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
from pykeops.torch import Vi, Vj
from deepinv.optim.utils import least_squares
import math


# TODO: deepinv as solver??


class OMPPyKeops(Algorithm):
    """
    OMP Algorithm using customizable multidimensional index sets (Hyperbolic Cross, Full Degree, Total Degree).
    """

    def __init__(self, basis_generator: BasisGenerator, grid_generator: GridGenerator, solver: Solver, device: torch.device,
                 hc_bandwidth: int | None, index_set_type, bandwidth_multiplier_function: Callable,
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
        self.index_set_type = index_set_type.lower()
        self.bandwidth_multiplier_function = bandwidth_multiplier_function

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
            base_R = OMPPyKeops._find_hc_bandwidth(dim, calculate_num_points(dim, scale))
        else:
            base_R = self.hc_bandwidth

        # Apply multiplier to scale up the dictionary size
        effective_R = int(np.ceil(self.bandwidth_multiplier_function(base_R)))

        # 3. Generate the selected pool of candidate indices
        if self.index_set_type == "hyperbolic":
            self._indices = self._hyp_cross(dim, effective_R)
        elif self.index_set_type == "total_degree":
            self._indices = self._total_degree_cross(dim, effective_R)
        else:
            raise ValueError(
                f"Unknown index_set_type: {self.index_set_type}. Choose 'hyperbolic', 'total_degree', or 'full_degree'.")

        self._norm_coeffs = (np.sqrt(2) ** np.clip(self._indices, 0, 1).sum(axis=1)).astype(np.float64)

        # Generate Lazy Matrix


        # 4. Materialize dense Chebyshev Matrix (pure PyTorch tensor transformation)
        # A = self._chebyshev_matrix(points_norm, self._indices, self._norm_coeffs, self.device, self.dtype)
        # y = self._calculate_y(f, self.grid)

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
                          device: torch.device, dtype: torch.dtype) -> np.ndarray:
        """Materializes the tensor evaluations mapping your multi-index system explicitly."""
        pts = torch.from_numpy(points_normalized).to(dtype=dtype, device=device)
        idx = torch.from_numpy(indices).to(dtype=dtype, device=device)
        coeffs = torch.from_numpy(norm_coeffs).to(dtype=dtype, device=device)

        eps = torch.finfo(dtype).eps
        samples_acos = torch.acos(torch.clamp(pts, -1.0 + eps, 1.0 - eps))

        angle = samples_acos[:, None, :] * idx[None, :, :]
        mat = torch.prod(torch.cos(angle), dim=2) # TODO: remove dim but loop
        mat *= coeffs[None, :]

        return mat.cpu().numpy()

    @staticmethod
    def aChebyshev_eval(p_acos, D):
        # Taken from https://github.com/Zeppo1994/SparseRecovery/blob/main/algorithms/OMP.py
        # Adjoint Chebyshev Transform (multidimensional)
        # x : tensor of type torch.Tensor and shape (N,1), real valued
        # p, k : tensors of type torch.Tensor and shapes (N,D), (M,D)
        k_i = Vi(1, D)  # (M, 1, D) LazyTensor
        pre_i = Vi(2, 1)  # (M, 1, 1) LazyTensor
        p_acos_j = Vj(p_acos)  # (1, N, D) LazyTensor
        x_j = Vj(0, 1)  # (1, N, 1) LazyTensor

        tmp = (k_i[:, :, 0] * p_acos_j[:, :, 0]).cos()
        for d in range(D - 1):
            tmp *= (k_i[:, :, d + 1] * p_acos_j[:, :, d + 1]).cos()
        return (pre_i * tmp * x_j).sum_reduction(dim=1, use_double_acc=True)

    @staticmethod
    def Chebyshev_eval(p_acos, D):
        # Taken from https://github.com/Zeppo1994/SparseRecovery/blob/main/algorithms/OMP.py
        # Chebyshev Transform (multidimensional)
        # x : tensor of type torch.Tensor and shape (N,1), real valued
        # p, k : tensors of type torch.Tensor and shapes (N,D), (M,D)
        k_j = Vj(1, D)  # (1, M, D) LazyTensor
        pre_j = Vj(2, 1)  # (1, M, 1) LazyTensor
        p_acos_i = Vi(p_acos)  # (N, 1, D) LazyTensor
        x_j = Vj(0, 1)  # (1, M, 1) LazyTensor

        tmp = (k_j[:, :, 0] * p_acos_i[:, :, 0]).cos()
        for d in range(D - 1):
            tmp *= (k_j[:, :, d + 1] * p_acos_i[:, :, d + 1]).cos()
        return (pre_j * tmp * x_j).sum_reduction(dim=1, use_double_acc=True)

    @staticmethod
    def normalization_Chebyhsev(p_acos, k, pre, D):
        # Taken from https://github.com/Zeppo1994/SparseRecovery/blob/main/algorithms/OMP.py
        # normalization of matrix columns
        k_i = Vi(k)  # (M, 1, D) LazyTensor
        pre_i = Vi(pre)  # (M, 1, 1) LazyTensor
        p_acos_j = Vj(p_acos)  # (1, N, D) LazyTensor

        tmp = (k_i[:, :, 0] * p_acos_j[:, :, 0]).cos()
        for d in range(D - 1):
            tmp *= (k_i[:, :, d + 1] * p_acos_j[:, :, d + 1]).cos()
        return ((pre_i * tmp) ** 2).sum_reduction(dim=1, use_double_acc=True)

    @staticmethod
    def aCosine_eval(p, D):
        # Taken from https://github.com/Zeppo1994/SparseRecovery/blob/main/algorithms/OMP.py
        # Adjoint Cosine transform (multidimensional)
        # x : tensor of type torch.Tensor and shape (N,1), real valued
        # p, k : tensors of type torch.Tensor and shapes (N,D), (M,D)
        k_i = Vi(1, D)  # (M, 1, D) LazyTensor
        k_i = math.pi * k_i
        pre_i = Vi(2, 1)  # (M, 1, 1) LazyTensor
        p_j = Vj((p + 1) / 2)  # (N, 1, D) LazyTensor
        x_j = Vj(0, 1)  # (1, N, 1) LazyTensor

        tmp = (k_i[:, :, 0] * p_j[:, :, 0]).cos()
        for d in range(D - 1):
            tmp *= (k_i[:, :, d + 1] * p_j[:, :, d + 1]).cos()
        return (pre_i * tmp * x_j).sum_reduction(dim=1, use_double_acc=True)

    @staticmethod
    def Cosine_eval(p, D):
        # Taken from https://github.com/Zeppo1994/SparseRecovery/blob/main/algorithms/OMP.py
        # Cosine transform (multidimensional)
        # x : tensor of type torch.Tensor and shape (N,1), real valued
        # p, k : tensors of type torch.Tensor and shapes (N,D), (M,D)
        k_j = Vj(1, D)  # (1, M, D) LazyTensor
        k_j = math.pi * k_j
        pre_j = Vj(2, 1)  # (1, M, 1) LazyTensor
        p_i = Vi((p + 1) / 2)  # (N, 1, D) LazyTensor
        x_j = Vj(0, 1)  # (1, M, 1) LazyTensor

        tmp = (k_j[:, :, 0] * p_i[:, :, 0]).cos()
        for d in range(D - 1):
            tmp *= (k_j[:, :, d + 1] * p_i[:, :, d + 1]).cos()
        return (pre_j * tmp * x_j).sum_reduction(dim=1, use_double_acc=True)

    @staticmethod
    def _hyp_cross_size(dim: int, R: int) -> int:
        if dim == 1:
            return R + 1
        total = 0
        for k in range(R + 1):
            total += OMPPyKeops._hyp_cross_size(dim - 1, R // max(1, abs(k)))
        return total

    @staticmethod
    def _find_hc_bandwidth(dim: int, target_m: int) -> int:
        hi = 1
        while OMPPyKeops._hyp_cross_size(dim, hi) < target_m:
            hi *= 2
        lo = hi // 2
        while lo < hi:
            mid = (lo + hi) // 2
            if OMPPyKeops._hyp_cross_size(dim, mid) >= target_m:
                hi = mid
            else:
                lo = mid + 1
        return lo
