"""
Faber (hat-function) basis with Smolyak-type level structure.

Uses the same multi-index selection rule as the Clenshaw-Curtis Smolyak
basis (``d ≤ Σ i_j ≤ d + scale``) but replaces Chebyshev polynomials
with hierarchical equidistant hat (tent) functions.

At each 1-D Smolyak level *i* the grid has ``m_i(i)`` equidistant knots.
Only the knots that are **new** at level *i* (absent at level *i − 1*)
introduce new hat functions.  This gives exactly the same cardinality
per level as the Chebyshev basis, so the total number of multivariate
basis functions equals ``calculate_num_points(dim, scale)``.
"""

from functools import reduce
from operator import mul

import numpy as np

from basis.basis import Basis
from basis.basis_generator import BasisGenerator
from basis.smolyak_indexing import SmolyakIndexing
from grid.grid.grid import Grid


class FaberBasisGenerator(SmolyakIndexing, BasisGenerator):
    """
    Smolyak-structured hierarchical hat-function basis.

    Per dimension, Smolyak level *i* contributes hat functions at
    the *new* equidistant knots of the ``m_i(i)``-point grid on
    ``[lower, upper]``.  The multivariate basis is the union of
    tensor products across all multi-indices ``(i_1, …, i_d)`` with
    ``d ≤ Σ i_j ≤ d + scale``.

    Smolyak multi-index helpers (_smolyak_idx, _poly_idx, _phi_chain,
    _m_i, _calculate_basis_indices) are inherited from SmolyakIndexing.

    The Vandermonde matrix is **dense**, shape ``(N_pts, N_basis)``
    where ``N_basis = calculate_num_points(dim, scale)``.
    """

    def __init__(self):
        super().__init__("Faber Hat Basis (Smolyak)", "FabS")

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def create_basis(self, grid: Grid, scale :int) -> Basis:
        """
        Build the Vandermonde matrix of hierarchical hat functions
        evaluated at all grid points.

        Parameters
        ----------
        grid : Grid
            Evaluation grid.  ``grid.scale`` determines the basis
            resolution; ``grid.lower_bound`` / ``grid.upper_bound``
            set the domain.

        Returns
        -------
        Basis
            Dense Vandermonde of shape ``(N_pts, N_basis)``.
        """
        dim = grid.input_dim
        lower = grid.lower_bound
        upper = grid.upper_bound
        pts = np.asarray(grid)                       # (N, dim)
        N = pts.shape[0]

        # 1. Smolyak-structured basis indices --------------------------
        b_idx = self.calculate_basis_indices(dim, scale)

        # 2. Hierarchical hat-function parameters ----------------------
        n_1d = self._m_i(scale + 1)                  # total 1-D functions
        knots, spacings = self._build_hat_params(scale + 1, lower, upper)
        assert len(knots) == n_1d

        # 3. Precompute all 1-D hat values -----------------------------
        #    hat_vals[j, d, :] = j-th hat at all points in dimension d
        hat_vals = np.empty((n_1d, dim, N))
        for j in range(n_1d):
            for d in range(dim):
                hat_vals[j, d, :] = np.maximum(
                    0.0,
                    1.0 - np.abs(pts[:, d] - knots[j]) / spacings[j],
                )

        # 4. Assemble Vandermonde via tensor products ------------------
        n_basis = len(b_idx)
        V = np.empty((N, n_basis))
        for col, comb in enumerate(b_idx):
            V[:, col] = reduce(
                mul,
                [hat_vals[comb[d] - 1, d, :] for d in range(dim)],
            )

        return Basis(V)


    # ------------------------------------------------------------------
    # Hierarchical hat-function parameters
    # ------------------------------------------------------------------

    @staticmethod
    def _build_hat_params(
        n: int, lower: float, upper: float,
    ) -> tuple[list[float], list[float]]:
        """
        For Smolyak levels 1 … *n*, compute the knot position and
        knot spacing of every **new** hat function.

        At level *i* the equidistant grid has ``m_i(i)`` knots with
        spacing ``h_i = (upper − lower) / (m_i(i) − 1)``.  The new
        knots are those not present at any previous level.

        Returns
        -------
        knots : list[float]
            Knot positions, length ``m_i(n)``.
        spacings : list[float]
            Knot spacings, one per knot.
        """
        knots: list[float] = []
        spacings: list[float] = []
        prev: set[float] = set()

        for lev in range(1, n + 1):
            m = FaberBasisGenerator._m_i(lev)
            if m == 1:
                h = upper - lower
                all_k = [(lower + upper) / 2.0]
            else:
                h = (upper - lower) / (m - 1)
                all_k = [lower + j * h for j in range(m)]

            new_k = sorted(
                k for k in all_k
                if not any(abs(k - p) < 1e-14 for p in prev)
            )
            for k in new_k:
                knots.append(k)
                spacings.append(h)
            prev.update(all_k)

        return knots, spacings
