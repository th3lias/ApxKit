import numba
import numpy as np
from scipy.spatial.distance import cdist


@numba.njit
def inverse_distance_weight(x: np.ndarray, center: np.ndarray, p: float = 2.0) -> float:
    """
        Basis function centered at `center`, evaluated at `x`.
        p: Power parameter (higher p -> more localized).
    """
    distance = np.linalg.norm(x - center)
    if distance < 1e-10:
        return 1.0  # Exact match
    else:
        return 1.0 / (distance ** p + 1e-10)  # Avoid division by zero


@numba.njit
def gaussian_rbf(x: np.ndarray, center: np.ndarray, epsilon: float) -> float:
    """
        Gaussian RBF centered at `center`, evaluated at `x`.
    """
    distance = np.linalg.norm(x - center)
    return np.exp(-(epsilon * distance) ** 2)


def compute_epsilon(grid_centers: np.ndarray, c: float = 1.0) -> float:
    """
        Computes epsilon for Gaussian RBF based on the distance to the nearest grid center.
    """
    pairwise_distances = cdist(grid_centers, grid_centers, metric="euclidean")
    np.fill_diagonal(pairwise_distances, np.inf)  # Ignore self-distances
    min_distance = np.min(pairwise_distances)
    return c / min_distance


@numba.njit(parallel=True)
def grid_basis_idw(points: np.ndarray, grid_centers: np.ndarray, p: float = 2.0) -> np.ndarray:
    """
        Builds Vandermonde matrix for scattered grid points using IDW basis.
    """
    n_samples = points.shape[0]
    n_basis = grid_centers.shape[0]
    matrix_A = np.zeros((n_samples, n_basis))
    for i in numba.prange(n_basis):
        center = grid_centers[i]
        for j in range(n_samples):
            matrix_A[j, i] = inverse_distance_weight(points[j], center, p)
    return matrix_A


@numba.njit
def grid_basis_gaussian(points: np.ndarray, grid_centers: np.ndarray, epsilon: float) -> np.ndarray:
    """
        Builds Vandermonde matrix for scattered grid points using Gaussian RBF basis.
    """
    n_samples = points.shape[0]
    n_basis = grid_centers.shape[0]
    matrix_A = np.zeros((n_samples, n_basis))
    for i in range(n_basis):
        center = grid_centers[i]
        for j in range(n_samples):
            matrix_A[j, i] = gaussian_rbf(points[j], center, epsilon)
    return matrix_A


@numba.njit
def evaluate_idw_basis(x: np.ndarray, grid_centers: np.ndarray, coefficients, p: float = 2.0) -> np.ndarray:
    """
        Evaluates the IDW basis at the given points.
    """
    result = np.zeros(x.shape[0])
    for j in range(x.shape[0]):
        result[j] = 0.0
        for i in range(grid_centers.shape[0]):
            result[j] += coefficients[i] * inverse_distance_weight(x[j], grid_centers[i], p)
    return result


@numba.njit
def evaluate_gaussian_rbf(x: np.ndarray, grid_centers: np.ndarray, coefficients: np.ndarray,
                          epsilon: float) -> np.ndarray:
    """
        Evaluates the Gaussian RBF basis at the given points.
    """
    result = np.zeros(x.shape[0])
    for j in range(x.shape[0]):
        result[j] = 0.0
        for i in range(grid_centers.shape[0]):
            result[j] += coefficients[i] * gaussian_rbf(x[j], grid_centers[i], epsilon)
    return result
