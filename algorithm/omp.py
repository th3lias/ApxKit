"""
Orthogonal Matching Pursuit (OMP) with Tchebychev basis.

Ported from SparseRecovery/algorithms/OMP.py (Tchebychev path) and adapted to
the ApxKit Algorithm interface.  The core algorithm (incremental Cholesky OMP)
is unchanged; the main differences from the SparseRecovery version are:

  * Pure numpy/scipy instead of torch + PyKeOps.  The Tchebychev evaluation
    matrix is materialised explicitly. For very large N or M, the PyKeOps
    lazy-tensor version in SparseRecovery is more memory-efficient.

  * Index-set generation is encapsulated in the class rather than being
    computed by the caller.  In SparseRecovery the caller builds the index
    tensor and passes it directly to OMP.Tchebychev(); here the equivalent
    parameter is ``hc_bandwidth`` (called ``M`` / ``J`` in SparseRecovery).

  * Domain mapping: ApxKit functions live on [lower, upper] (default [0, 1]);
    points are mapped to [-1, 1] internally before Tchebychev evaluation.
"""
import math

import numpy as np
from scipy.linalg import cho_solve, solve_triangular

from algorithm.algorithm import Algorithm
from function.f import Function
from grid.generator.grid_generator import GridGenerator
from grid.grid.grid import Grid


class _Stub:
    """Placeholder that satisfies the ``solver`` / ``basis_generator`` slots on
    Algorithm so that ExperimentExecutor can read ``.name`` and ``.abbr_name``
    for logging without OMP needing a real solver or basis generator."""
    def __init__(self, name: str, abbr_name: str):
        self.name = name
        self.abbr_name = abbr_name


def _hyp_cross(dim: int, R: int) -> np.ndarray:
    """
    Return all hyperbolic-cross multi-indices for dimension *dim* and
    bandwidth *R*.

    The hyperbolic cross of bandwidth R is the set
        HC(D, R) = { k in N_0^D : prod_d max(1, k_d) <= R }.

    Equivalent to ``hyp_cross(dim, R)`` defined locally in
    SparseRecovery/OMP_Fun3.py and OMP_Fun4.py.

    Returns an (M, dim) int32 array where M = |HC(D, R)|.
    """
    if dim == 1:
        return np.arange(R + 1, dtype=np.int32).reshape(-1, 1)
    out = []
    for k in range(R + 1):
        sub = _hyp_cross(dim - 1, R // max(1, k))
        block = np.empty((len(sub), dim), dtype=np.int32)
        block[:, 0] = k
        block[:, 1:] = sub
        out.append(block)
    return np.vstack(out)


def _tchebychev_matrix(points_normalized: np.ndarray, indices: np.ndarray,
                       norm_coeffs: np.ndarray) -> np.ndarray:
    """
    Build the N x M Tchebychev evaluation matrix.

        mat[i, m] = norm_coeffs[m] * prod_d T_{k[m,d]}(x[i,d])
                  = norm_coeffs[m] * prod_d cos(k[m,d] * arccos(x[i,d]))

    In SparseRecovery this product is computed lazily via PyKeOps
    (``Tchebychev_eval`` / ``aTchebychev_eval``); here it is materialised
    as a dense array.

    Parameters
    ----------
    points_normalized : (N, D) in [-1, 1]
    indices           : (M, D) non-negative integer multi-indices
    norm_coeffs       : (M,)  sqrt(2)^#{d : k_d >= 1}  (L2-normalisation weights)
    """
    eps = np.finfo(np.float64).eps
    samples_acos = np.arccos(np.clip(points_normalized, -1.0 + eps, 1.0 - eps))  # (N, D)
    angle = samples_acos[:, None, :] * indices[None, :, :].astype(np.float64)    # (N, M, D)
    mat = np.prod(np.cos(angle), axis=2)                                         # (N, M)
    mat *= norm_coeffs[None, :]
    return mat


def _omp_solve(A: np.ndarray, b: np.ndarray, num_iters: int, tol: float) -> np.ndarray:
    """
    OMP with incremental Cholesky updates (numpy/scipy).

    Direct numpy port of ``OMP()`` in SparseRecovery/algorithms/OMP.py.
    The algorithm is described in https://ieeexplore.ieee.org/document/6333943.
    ``torch.cholesky_solve`` maps to ``scipy.linalg.cho_solve``;
    ``torch.linalg.solve_triangular`` maps to ``scipy.linalg.solve_triangular``.

    Parameters
    ----------
    A         : (N, M)  full Tchebychev basis matrix
    b         : (N, 1)  function values at training points
    num_iters : maximum number of OMP iterations (= maximum sparsity)
    tol       : stop early when ||residual||_2 / sqrt(N) < tol

    Returns
    -------
    x : (M, 1)  sparse coefficient vector; non-zero only at selected indices.
    """
    N, M = A.shape
    eps = np.finfo(A.dtype).eps

    # Column norms for normalised correlation (≈ sqrt(N) for an orthonormal basis)
    col_norms = np.sqrt((A ** 2).sum(axis=0))
    col_norms = np.maximum(col_norms, eps)

    num_iters = min(num_iters, M, N)
    selected = np.empty(num_iters, dtype=int)
    L = np.zeros((num_iters, num_iters))    # lower-triangular Cholesky factor of A_S^T A_S
    rhs = np.zeros((num_iters, 1))
    res = b.copy()
    out = None
    actual_iters = 0

    for j in range(num_iters):
        # Normalised correlation with residual; mask already-selected columns
        corr = A.T @ res
        z2 = np.abs(corr.flatten()) / col_norms
        if j > 0:
            z2[selected[:j]] = -1.0

        idx = int(np.argmax(z2))
        selected[j] = idx
        col = A[:, idx]

        # Recursive Cholesky update of A_S^T A_S
        if j == 0:
            L[0, 0] = np.linalg.norm(col)
        else:
            c = A[:, selected[:j]].T @ col
            v = solve_triangular(L[:j, :j], c[:, None], lower=True).flatten()
            L[j, :j] = v
            L[j, j] = math.sqrt(max(col_norms[idx] ** 2 - v @ v, eps ** 2))

        rhs[j, 0] = col @ b.flatten()
        # Solve (L L^T) out = rhs  →  least-squares solution over selected columns
        out = cho_solve((L[:j + 1, :j + 1], True), rhs[:j + 1])
        res = b - A[:, selected[:j + 1]] @ out

        residual = np.linalg.norm(res) / math.sqrt(N)
        if j % 100 == 0:
            print(f"Iteration: {j + 1}  Residual: {residual:.6e}")
        if residual < tol:
            actual_iters = j + 1
            break
        actual_iters = j + 1

    x = np.zeros((M, 1))
    if out is not None:
        x[selected[:actual_iters]] = out[:actual_iters]
    return x


class OMP(Algorithm):
    """
    Orthogonal Matching Pursuit with Tchebychev basis on a hyperbolic cross
    index set.

    Wraps the Tchebychev path from SparseRecovery/algorithms/OMP.py in the
    ApxKit Algorithm interface so that it can be used interchangeably with
    LeastSquaresAlgorithm, WeightedLeastSquaresAlgorithm, and SmolyakAlgorithm
    inside ExperimentExecutor.

    Differences from the SparseRecovery interface
    ----------------------------------------------
    In SparseRecovery the caller constructs the index tensor and passes it
    directly to ``OMP.Tchebychev(samples, values, indices, ...)``.  Here the
    index set is built internally from ``hc_bandwidth`` (called ``M`` or ``J``
    in SparseRecovery) because the framework owns the ``fit()`` call and the
    caller never sees the raw indices.

    Unlike the other ApxKit algorithms, the basis size is controlled by
    ``hc_bandwidth`` and is independent of the ``scale`` parameter passed to
    ``fit()``.  ``scale`` only determines the number of training points via the
    grid generator.

    Parameters
    ----------
    grid_generator : GridGenerator
        Should be a ChebyshevGridGenerator to match the Tchebychev sampling
        measure.  Points are mapped from [lower, upper] to [-1, 1] internally.
    hc_bandwidth : int
        Hyperbolic-cross bandwidth R: the index set is
        HC(D, R) = { k in N_0^D : prod_d max(1, k_d) <= R }.
        Corresponds to the parameter ``M`` / ``J`` in SparseRecovery.
    num_iters : int
        Maximum OMP iterations, i.e. the maximum number of basis functions
        that can be selected.  Corresponds to ``num_iters`` in SparseRecovery.
    tol : float
        Early-stopping tolerance on the normalised residual ||r||_2 / sqrt(N).
        Corresponds to ``tol`` in SparseRecovery.
    """

    def __init__(
        self,
        grid_generator: GridGenerator,
        hc_bandwidth: int = 10,
        num_iters: int = 5000,
        tol: float = 1e-4,
        name: str = "OMP_Tchebychev",
        abbr_name: str = "OT",
    ):
        super().__init__(name, abbr_name,
                         _Stub("TCHEBYCHEV_HC", "TC"),
                         grid_generator,
                         _Stub("OMP", "OMP"))
        self.hc_bandwidth = hc_bandwidth
        self.num_iters = num_iters
        self.tol = tol

        self._indices: np.ndarray | None = None
        self._norm_coeffs: np.ndarray | None = None
        self._lower: float | None = None
        self._upper: float | None = None

    def fit(self, dim: int, scale: int, f: Function | list[Function],
            lower: float = 0.0, upper: float = 1.0) -> None:
        self._lower = lower
        self._upper = upper

        self.grid = self.grid_generator.get_grid(dim=dim, scale=scale,
                                                 lower_bound=lower, upper_bound=upper)
        points = np.array(self.grid)
        points_norm = 2.0 * (points - lower) / (upper - lower) - 1.0  # map to [-1, 1]

        y = self._calculate_y(f, self.grid)

        self._indices = _hyp_cross(dim, self.hc_bandwidth)
        self._norm_coeffs = (np.sqrt(2) ** np.clip(self._indices, 0, 1)
                             .sum(axis=1)).astype(np.float64)

        A = _tchebychev_matrix(points_norm, self._indices, self._norm_coeffs)

        M = self._indices.shape[0]
        n_funcs = y.shape[1]
        self.coeff = np.zeros((M, n_funcs))
        for col in range(n_funcs):
            x = _omp_solve(A, y[:, col:col + 1], self.num_iters, self.tol)
            self.coeff[:, col] = x.flatten()

    def evaluate(self, grid: Grid) -> np.ndarray:
        points = np.array(grid)
        points_norm = 2.0 * (points - self._lower) / (self._upper - self._lower) - 1.0
        A_test = _tchebychev_matrix(points_norm, self._indices, self._norm_coeffs)
        return A_test @ self.coeff
