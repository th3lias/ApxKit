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
                 abbr_name: str = "CC_PyKeops"):
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
        """
        Builds the lazy forward Chebyshev transform, bound to a fixed set of sample points.

        The returned callable maps coefficients to function values:
            values_i = sum_j pre_j * prod_d cos(k_jd * acos(points_i)_d) * x_j
        i.e. `values = A @ x`, where A is the (never materialized) Chebyshev dictionary.

        Args:
            points_acos: arccos of the normalized sample points, shape (n_points, dim).
            dim: Number of spatial dimensions (D).

        Returns:
            A callable operator(x, indices, norm_coeffs) -> values, shape (n_points, 1).
            `x`, `indices` and `norm_coeffs` may have any (matching) number of rows, i.e.
            the operator can be called with the full candidate set or any subset of it.
        """
        k_j = Vj(1, dim)  # candidate multi-indices, one per dictionary atom (column)
        pre_j = Vj(2, 1)  # per-atom Chebyshev normalization prefactor
        p_acos_i = Vi(points_acos)  # fixed sample points (row variable)
        x_j = Vj(0, 1)  # coefficient vector, one entry per atom

        tmp = (k_j[:, :, 0] * p_acos_i[:, :, 0]).cos()
        for d in range(dim - 1):
            tmp = tmp * (k_j[:, :, d + 1] * p_acos_i[:, :, d + 1]).cos()

        return (pre_j * tmp * x_j).sum_reduction(dim=1, use_double_acc=True)

    @staticmethod
    def build_adjoint_operator(points_acos: torch.Tensor, dim: int):
        """
        Builds the lazy adjoint Chebyshev transform, bound to a fixed set of sample points.

        The returned callable maps function values to coefficient-space correlations:
            coeffs_j = sum_i pre_j * prod_d cos(k_jd * acos(points_i)_d) * x_i
        i.e. `coeffs = A.T @ x`.

        Args:
            points_acos: arccos of the normalized sample points, shape (n_points, dim).
            dim: Number of spatial dimensions (D).

        Returns:
            A callable operator(x, indices, norm_coeffs) -> coeffs, shape (n_indices, 1).
            `indices`/`norm_coeffs` may be the full candidate set or any subset of it;
            `x` must always be aligned with the (fixed) sample points.
        """
        k_i = Vi(1, dim)  # candidate multi-indices (now the output/row variable)
        pre_i = Vi(2, 1)
        p_acos_j = Vj(points_acos)  # fixed sample points (column variable)
        x_j = Vj(0, 1)  # function values, one entry per sample point

        tmp = (k_i[:, :, 0] * p_acos_j[:, :, 0]).cos()
        for d in range(dim - 1):
            tmp = tmp * (k_i[:, :, d + 1] * p_acos_j[:, :, d + 1]).cos()

        return (pre_i * tmp * x_j).sum_reduction(dim=1, use_double_acc=True)

    @staticmethod
    def build_normalization_operator(points_acos: torch.Tensor, dim: int):
        """
        Builds the lazy column-normalization reduction, bound to a fixed set of sample points.

        The returned callable computes the squared L2 norm of every dictionary column
        without ever forming the columns themselves:
            norm_j = sum_i (pre_j * prod_d cos(k_jd * acos(points_i)_d))^2

        Args:
            points_acos: arccos of the normalized sample points, shape (n_points, dim).
            dim: Number of spatial dimensions (D).

        Returns:
            A callable operator(indices, norm_coeffs) -> norm, shape (n_indices, 1).
            Note this returns the squared norm; take a square root to get the norm itself.
        """

        k_i = Vi(0, dim)  # If you absolutely must use manual indices, you must start from 0!
        pre_i = Vi(1, 1)  # 0, 1, and then p_acos_j will auto-bind to 2.
        p_acos_j = Vj(points_acos)

        tmp = (k_i[:, :, 0] * p_acos_j[:, :, 0]).cos()
        for d in range(dim - 1):
            tmp = tmp * (k_i[:, :, d + 1] * p_acos_j[:, :, d + 1]).cos()

        return ((pre_i * tmp) ** 2).sum_reduction(dim=1, use_double_acc=True)

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