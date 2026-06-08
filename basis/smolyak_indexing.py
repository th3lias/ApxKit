"""
Smolyak multi-index machinery shared by every basis generator that uses
the Clenshaw-Curtis level selection rule  (d ≤ Σ i_j ≤ d + scale).

Provides the five static helpers as a mixin so that concrete generators
(Chebyshev polynomials, Faber hats, …) can inherit them without
duplicating code.
"""

from itertools import product

from basis.partition import Partition


class SmolyakIndexing:
    """Mixin that supplies Smolyak-level index helpers."""

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    @staticmethod
    def calculate_basis_indices(dim: int, scale: int) -> list[tuple[int]]:
        """Return the full list of per-function index tuples for the
        Smolyak basis of dimension *dim* and depth *scale*."""
        idx = SmolyakIndexing._smolyak_idx(dim, scale)
        return SmolyakIndexing._poly_idx(dim, scale, idx)

    # ------------------------------------------------------------------
    # Multi-index enumeration
    # ------------------------------------------------------------------

    @staticmethod
    def _smolyak_idx(dim: int, scale: int) -> list[list[int]]:
        r"""
        All multi-indices :math:`(i_1, \ldots, i_d)` with :math:`i_j \geq 1`
        and :math:`d \leq \sum i_j \leq d + \text{scale}`.
        """
        if not isinstance(scale, int):
            raise ValueError(f"scale must be int, got {type(scale)}")
        idx_list: list[list[int]] = []
        for q in range(dim, scale + dim + 1):
            p = Partition(dim, q, limit=1)
            idx_list.extend(p.get_all_partitions())
        return idx_list

    @staticmethod
    def _poly_idx(dim: int, scale: int, idx: list[list[int]]) -> list[tuple[int]]:
        """Expand Smolyak multi-indices into per-function index tuples
        via the Cartesian product of the ``_phi_chain`` sets."""
        if not isinstance(scale, int):
            raise ValueError(f"scale must be int, got {type(scale)}")
        aphi = SmolyakIndexing._phi_chain(scale + 1)
        polys: list[tuple[int]] = []
        for el in idx:
            sets = [aphi[i] for i in el]
            polys.extend(list(product(*sets)))
        return polys

    # ------------------------------------------------------------------
    # Level → 1-D function-index mapping
    # ------------------------------------------------------------------

    @staticmethod
    def _phi_chain(n: int) -> dict[int, list[int] | range]:
        """
        Level *i* → list of global 1-D function indices introduced at
        that level.

        Cardinalities:
            level 1 → 1,  level 2 → 2,  level i (i ≥ 3) → 2^(i-2).

        Cumulative total up to level *n* equals ``_m_i(n)``.
        """
        chain: dict[int, list[int] | range] = {}
        if n >= 1:
            chain[1] = [1]
        if n >= 2:
            chain[2] = [2, 3]
        cur = 4
        for i in range(3, n + 1):
            end = 2 ** (i - 1) + 1
            chain[i] = list(range(cur, end + 1))
            cur = end + 1
        return chain

    # ------------------------------------------------------------------
    # Knot-count formula
    # ------------------------------------------------------------------

    @staticmethod
    def _m_i(i: int) -> int:
        r"""
        Number of 1-D knots at Smolyak level *i*.

        .. math::
            m_i = \begin{cases}
                i          & i < 2 \\
                2^{i-1}+1  & i \geq 2
            \end{cases}
        """
        if i < 0:
            raise ValueError("i must be non-negative")
        if i < 2:
            return i
        return 2 ** (i - 1) + 1

