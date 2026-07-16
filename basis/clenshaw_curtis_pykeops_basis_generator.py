from __future__ import annotations

import torch
from pykeops.torch import Vi, Vj

from basis.basis import Basis
from basis import BasisGenerator
from grid.grid.grid import Grid

class ClenshawCurtisPyKeopsBasisGenerator(BasisGenerator):
    """
    PyKeOps-backed basis generator for the (tensorized) Chebyshev / Clenshaw-Curtis
    polynomial basis.

    Instead of materializing the dense (n_points x n_indices) Vandermonde-type matrix,
    this generator builds *lazy* KeOps reductions for:
      - the forward transform  A:   coefficients   -> function values,
      - the adjoint transform  A^T: function values -> coefficient-space correlations,
      - the column normalization sum_i A_ij^2, needed by matching-pursuit-style solvers.

    Each of the three is built once per sample-point set (via build_forward_operator /
    build_adjoint_operator / build_normalization_operator) and can then be called
    repeatedly with different (sub-)sets of candidate indices. This is exactly the
    access pattern needed by the PyKeOps OMP solver (full candidate set for the
    matching step, small subsets of already-selected atoms for the Cholesky update).
    """

    def __init__(self,
                 device: torch.device,
                 name: str = "Clenshaw_Curtis_PyKeops_Basis_Generator",
                 abbr_name: str = "CS"):
        """
        Args:
            device: Computing device ('cpu', 'cuda', 'mps').
            name: Human readable name of the basis generator.
            abbr_name: Short name of the basis generator.
        """
        super().__init__(name=name, abbr_name=abbr_name)
        self.device = device

    def create_basis(self,
                     grid: Grid,
                     scale: int) -> Basis:
        """Not supported for the PyKeOps generator.

        Materializing a dense Basis here would defeat the entire purpose of the lazy
        evaluation this generator exists for. Use build_forward_operator /
        build_adjoint_operator / build_normalization_operator instead, which is what
        OMPPyKeops.fit()/evaluate() do.
        """
        raise NotImplementedError(
            "ClenshawCurtisPyKeopsBasisGenerator never materializes a dense basis matrix. "
            "Use build_forward_operator/build_adjoint_operator/build_normalization_operator instead."
        )

    def _basis_rule(self):
        raise NotImplementedError(
            "ClenshawCurtisPyKeopsBasisGenerator does not use a dense basis rule."
        )

    # The following methods are taken and adapted from:
    # https://github.com/Zeppo1994/SparseRecovery/blob/main/algorithms/OMP.py

    @staticmethod
    def build_forward_operator(points_acos: torch.Tensor, dim: int):
        # points_acos: (N_points, dim) -> Vi (dimension i, since output is on grid points)
        p_acos_i = Vi(points_acos)

        def forward_op(coeffs: torch.Tensor, indices: torch.Tensor, norm_coeffs: torch.Tensor):
            # coeffs: (N_indices, 1) -> Vj (dimension j)
            c_j = Vj(coeffs)
            # indices: (N_indices, dim) -> Vj (dimension j)
            k_j = Vj(indices)
            # norm_coeffs: (N_indices, 1) -> Vj (dimension j)
            pre_j = Vj(norm_coeffs)

            tmp = (k_j[:, :, 0] * p_acos_i[:, :, 0]).cos()
            for d in range(dim - 1):
                tmp = tmp * (k_j[:, :, d + 1] * p_acos_i[:, :, d + 1]).cos()

            # We reduce over Vj (dim=1) to yield an output of size (N_points, 1)
            return ((pre_j * tmp) * c_j).sum_reduction(dim=1, use_double_acc=True)

        return forward_op

    @staticmethod
    def build_adjoint_operator(points_acos: torch.Tensor, dim: int):
        # points_acos: (N_points, dim) -> Vj (dimension j)
        p_acos_j = Vj(points_acos)

        def adjoint_op(y: torch.Tensor, indices: torch.Tensor, norm_coeffs: torch.Tensor):
            # y (values): (N_points, 1) -> Vj (dimension j)
            y_j = Vj(y)
            # indices: (N_indices, dim) -> Vi (dimension i)
            k_i = Vi(indices)
            # norm_coeffs: (N_indices, 1) -> Vi (dimension i)
            pre_i = Vi(norm_coeffs)

            tmp = (k_i[:, :, 0] * p_acos_j[:, :, 0]).cos()
            for d in range(dim - 1):
                tmp = tmp * (k_i[:, :, d + 1] * p_acos_j[:, :, d + 1]).cos()

            # We reduce over Vj (dim=1) to yield an output of size (N_indices, 1)
            return ((pre_i * tmp) * y_j).sum_reduction(dim=1, use_double_acc=True)

        return adjoint_op

    @staticmethod
    def build_normalization_operator(points_acos: torch.Tensor, dim: int):
        # points_acos: (N_points, dim) -> Vj (dimension j)
        p_acos_j = Vj(points_acos)

        def normalization_op(indices: torch.Tensor, norm_coeffs: torch.Tensor):
            # indices: (N_indices, dim) -> Vi (dimension i)
            k_i = Vi(indices)
            # norm_coeffs: (N_indices, 1) -> Vi (dimension i)
            pre_i = Vi(norm_coeffs)

            tmp = (k_i[:, :, 0] * p_acos_j[:, :, 0]).cos()
            for d in range(dim - 1):
                tmp = tmp * (k_i[:, :, d + 1] * p_acos_j[:, :, d + 1]).cos()

            # Sum over Vj (dim=1) to get the norm of each candidate column (N_indices, 1)
            return ((pre_i * tmp) ** 2).sum_reduction(dim=1, use_double_acc=True)

        return normalization_op

    @staticmethod
    def extract_column(points_acos: torch.Tensor, indices: torch.Tensor, norm_coeffs: torch.Tensor,
                       index: int) -> torch.Tensor:
        """
        Explicitly materializes a single Chebyshev dictionary column (dense, length n_points).

        The OMP solver's incremental Cholesky update needs the actual newly-selected atom
        (not just correlations with it). Since only one column is ever built at a time,
        memory usage stays negligible regardless of the total dictionary size.

        Args:
            points_acos: arccos of the normalized sample points, shape (n_points, dim).
            indices: Candidate multi-indices, shape (n_indices, dim).
            norm_coeffs: Chebyshev normalization prefactors, shape (n_indices, 1).
            index: Row of `indices`/`norm_coeffs` identifying the atom to extract.

        Returns:
            The dictionary column, shape (n_points, 1).
        """
        atom_index = indices[index]  # shape (dim,)
        prefactor = norm_coeffs[index]  # shape (1,)
        return prefactor * torch.cos(atom_index * points_acos).prod(dim=-1, keepdim=True)