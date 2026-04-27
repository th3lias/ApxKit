"""
Riesz basis R_d = {1} ∪ {C_k, S_k : k ∈ Z^d, k ≻ 0} for ReLU-friendly
function approximation.

The basis consists of piecewise-linear analogues of cos/sin (Eq. 1 in
[Schneider, Vybíral 2024]):

    C(x) = 4|x − 1/2| − 1      (interpolates cos 2πx at {0, 1/4, 1/2, 3/4, 1})
    S(x) = |2 − 4|x − 1/4|| − 1 (interpolates sin 2πx at the same points)

extended periodically and lifted to R^d via C_k(x) = C(k·x), S_k(x) = S(k·x).

References
----------
[53] C. Schneider, J. Vybíral.  A multivariate Riesz basis of ReLU neural
     networks.  Appl. Comput. Harmon. Anal. 68 (2024), 101605.
[1]  C. Schneider, M. Ullrich, J. Vybíral.  Nonlocal techniques for the
     analysis of deep ReLU neural network approximations.  arXiv:2504.04847, 2025.
"""

from __future__ import annotations

import numpy as np

from basis.basis import Basis
from basis.basis_generator import BasisGenerator
from grid.grid.grid import Grid


# ---------------------------------------------------------------------------
# Univariate basis function evaluation
# ---------------------------------------------------------------------------

def eval_C(t: np.ndarray) -> np.ndarray:
    """Evaluate C(t) = 4|frac(t) − 1/2| − 1 (periodic, period 1)."""
    t = np.asarray(t, dtype=float)
    frac = t - np.floor(t)
    return 4.0 * np.abs(frac - 0.5) - 1.0


def eval_S(t: np.ndarray) -> np.ndarray:
    """Evaluate S(t) = |2 − 4|frac(t) − 1/4|| − 1 (periodic, period 1)."""
    t = np.asarray(t, dtype=float)
    frac = t - np.floor(t)
    return np.abs(2.0 - 4.0 * np.abs(frac - 0.25)) - 1.0


def eval_C_k(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Evaluate C_k(x) = C(k·x) for x ∈ R^{N×d}, k ∈ Z^d."""
    return eval_C(x @ np.asarray(k, dtype=float))


def eval_S_k(x: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Evaluate S_k(x) = S(k·x) for x ∈ R^{N×d}, k ∈ Z^d."""
    return eval_S(x @ np.asarray(k, dtype=float))


# ---------------------------------------------------------------------------
# Basis generator
# ---------------------------------------------------------------------------

class RieszBasisGenerator(BasisGenerator):
    """
    Build the Vandermonde matrix for the Riesz basis R_d.

    Columns: ``[1, C_{k_0}, S_{k_0}, C_{k_1}, S_{k_1}, …]``
    where the k_i are drawn from the supplied ``index_set``.

    In 1-D with ``max_freq=K`` the natural index set is
    ``[(1,), (2,), …, (K,)]``, giving ``2K + 1`` basis functions.

    Parameters
    ----------
    index_set : list[tuple[int, ...]]
        Frequency multi-indices k ∈ Z^d with k ≻ 0.
    """

    def __init__(self, index_set: list[tuple[int, ...]]):
        super().__init__("Riesz Basis", "Riesz")
        self.index_set = index_set

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def create_basis(self, grid: Grid) -> Basis:
        """
        Build the dense Vandermonde matrix of shape ``(N, 2·#I + 1)``.

        Parameters
        ----------
        grid : Grid
            Evaluation grid whose ``.grid`` holds the point array.

        Returns
        -------
        Basis
            Dense Vandermonde matrix.
        """
        pts = np.asarray(grid)                       # (N, dim)
        n = pts.shape[0]
        m = len(self.index_set)
        V = np.ones((n, 2 * m + 1))
        for i, k_tuple in enumerate(self.index_set):
            k = np.array(k_tuple, dtype=float)
            t = pts @ k                              # (N,)
            V[:, 1 + 2 * i] = eval_C(t)
            V[:, 2 + 2 * i] = eval_S(t)
        return Basis(V)

    @property
    def n_basis(self) -> int:
        """Number of basis functions (including the constant)."""
        return 2 * len(self.index_set) + 1

    # ------------------------------------------------------------------
    # Coefficient helpers
    # ------------------------------------------------------------------

    def coefficients_to_dict(
        self, coefficients: np.ndarray,
    ) -> tuple[float, dict[tuple[int, ...], tuple[float, float]]]:
        """
        Convert a flat coefficient vector into the format expected by
        ``coefficients_to_network``.

        Parameters
        ----------
        coefficients : (2·#I + 1,) or (2·#I + 1, 1) array

        Returns
        -------
        alpha_0 : float
            Constant term.
        coeff_dict : dict
            ``{k_tuple: (alpha_k, beta_k)}`` for every k with a nonzero pair.
        """
        c = np.asarray(coefficients).ravel()
        alpha_0 = float(c[0])
        coeff_dict: dict[tuple[int, ...], tuple[float, float]] = {}
        for i, k_tuple in enumerate(self.index_set):
            a = float(c[1 + 2 * i])
            b = float(c[2 + 2 * i])
            if abs(a) > 0 or abs(b) > 0:
                coeff_dict[k_tuple] = (a, b)
        return alpha_0, coeff_dict
