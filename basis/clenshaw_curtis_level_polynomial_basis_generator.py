from functools import reduce
from operator import mul

import numpy as np
import os

from basis.basis import Basis
from basis.basis_generator import BasisGenerator
from basis.smolyak_indexing import SmolyakIndexing

from grid.grid.grid import Grid


class ClenshawCurtisLevelPolynomialBasisGenerator(SmolyakIndexing, BasisGenerator):
    """
    Chebyshev polynomial basis with Smolyak-level index selection.

    Creates a basis that is exact on the Smolyak index set E(q, d).
    Multi-index helpers are inherited from ``SmolyakIndexing``.
    """

    def __init__(self, store_indices: bool = True):
        super().__init__("CHEBYSHEV", "CS")
        self._b_idx = None
        self.store_indices = store_indices

    def create_basis(self, grid: Grid, path: str = None) -> Basis:
        dim = grid.input_dim
        scale = grid.scale

        self._b_idx = self._load_basis_indices_if_existent(dim, scale, path=path)

        data = np.array(grid)

        if self._b_idx is None:
            self._b_idx = self._calculate_basis_indices(dim, scale)
            if self.store_indices:
                self._save_basis_indices(self._b_idx, dim, scale, path=path)

        # Precompute all 1-D Chebyshev polynomials up to level scale+1
        cheby_polynomials = self._cheby2n(data.T, self._m_i(scale + 1))
        n_polys = len(self._b_idx)
        npts = data.shape[0]
        basis_array = np.empty(shape=(npts, n_polys))

        # Assemble multivariate basis via tensor products of 1-D polynomials
        for ind, comb in enumerate(self._b_idx):
            basis_array[:, ind] = reduce(mul, [cheby_polynomials[comb[i] - 1, i, :] for i in range(dim)])

        return Basis(basis_array)

    @staticmethod
    def _cheby2n(x, n):
        """
        Evaluate L²-normalised Chebyshev polynomials T_0 … T_n at *x*.

        Input *x* is assumed to lie in [0, 1] and is mapped to [-1, 1]
        internally.  T_0 = 1 is left unscaled; T_k (k ≥ 1) are multiplied
        by √2 so that each has unit L²[0,1] norm.

        Returns an ``(n+1, *x.shape)`` array.
        """
        x = 2 * x - 1  # map [0,1] → [-1,1]
        x = np.asarray(x)
        dim = x.shape
        results = np.zeros((n + 1,) + dim)
        results[0, ...] = np.ones(dim)
        results[1, ...] = x
        for i in range(2, n + 1):
            results[i, ...] = 2 * x * results[i - 1, ...] - results[i - 2, ...]
        # Normalise T_k (k ≥ 1) to unit L²[0,1] norm
        results[1:, ...] *= np.sqrt(2)
        return results

    @staticmethod
    def _load_basis_indices_if_existent(dim: int, scale: int, path=None):
        """Load precomputed basis indices from disk, or return None."""
        if path is None:
            path = os.path.join('indices')
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, f'dim{dim}_scale{scale}.npy')
        try:
            return np.load(path, allow_pickle=True)
        except FileNotFoundError:
            return None

    @staticmethod
    def _save_basis_indices(_b_idx, dim: int, scale: int, path=None):
        """Persist basis indices to disk (skips if file already exists)."""
        if path is None:
            path = os.path.join('indices')
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, f'dim{dim}_scale{scale}.npy')
        if not os.path.exists(path):
            np.save(path, _b_idx, allow_pickle=True)

