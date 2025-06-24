import numpy as np


def rescale(grid: np.ndarray, lower_bound: float, upper_bound: float) -> np.ndarray:
    """ Rescales a grid of points from the range [-1, 1] to the specified range [lower_bound, upper_bound]. """
    if lower_bound == -1. and upper_bound == 1.:
        return grid
    grid = (grid + 1) / 2
    return grid * (upper_bound - lower_bound) + lower_bound


def sample_chebyshev_univariate(num_points: int, lower_bound: float = 0.0, upper_bound: float = 1.0) -> np.ndarray:
    """Uses the inverse transform method. CDF is arcsin(x) and the inverse is sin(x)"""
    points = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=num_points)
    return rescale(grid=np.sin(points), lower_bound=lower_bound, upper_bound=upper_bound)
