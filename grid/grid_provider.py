"""
Provides sparse grids and random grids.
"""
import numpy as np

from grid.grid import Grid
from grid.grid_type import GridType

from utils.utils import calculate_num_points


class GridProvider:
    """
    Provides sparse grids, random grids and grids sampled from the chebyshev extrema.
    :param dimension: dimension of the function to be approximated
    :param seed: random seed to be used when option RANDOM is used
    """

    def __init__(self, dimension: int, seed: int = None, lower_bound: float = 0., upper_bound: float = 1.):
        self.dim = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        if isinstance(seed, int):
            self.seed = seed
            self.rng = np.random.default_rng(seed=seed)
        else:
            self.rng = np.random.default_rng(seed=None)

    def set_seed(self, seed: int):
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def generate(self, grid_type: GridType, scale: int = None, multiplier: float = 1.0,
                 remove_duplicates: bool = True) -> Grid:
        """
        Generate a grid of given type.
        :param scale:  Number of points (per dimension!) when generating the grid equidistantly or randomly.
        If GridType == RANDOM we aim to have the same number of points as in the regular
        grid and therefore sample the same num_points**dim number of points uniformly. If
        GridType == CHEBYSHEV we use the scale parameter to determine the fineness of the sparse grid.
        :param grid_type: GridType specification, e.g. Chebyshev, Random or Equidistant.
        :param multiplier: Only used in Random-Grid; Increases the number of samples by the given multiplier
        :param remove_duplicates: Remove duplicate values in the grid up to a small distance. Needed for Smolyak.
        :return: np.ndarray representing the grid
        """
        if not isinstance(grid_type, GridType):
            raise ValueError("grid type not supported: " + str(grid_type))
        if scale is None:
            raise ValueError("Please provide the fineness parameter of the grid")

        if grid_type == GridType.CHEBYSHEV:
            points = self._full_cheby_grid(level=scale, remove_duplicates=remove_duplicates)
            return Grid(self.dim, scale, points, grid_type)

        n_points = calculate_num_points(scale, self.dim)

        if grid_type == GridType.REGULAR:
            points = self._generate_equidistant_grid(num_points=n_points)
            return Grid(self.dim, scale, points, grid_type)
        if grid_type == GridType.RANDOM:
            n_points = int(n_points * multiplier)
            points = self._generate_random_grid(num_points=n_points)
            return Grid(self.dim, scale, points, grid_type)

    def _generate_random_grid(self, num_points: int) -> np.ndarray:
        return self.rng.uniform(low=self.lower_bound, high=self.upper_bound, size=(num_points, self.dim))

    def _generate_equidistant_grid(self, num_points: int) -> np.ndarray:
        num_points = np.full(shape=self.dim, fill_value=num_points)
        axes = [np.linspace(self.lower_bound, self.upper_bound, num_points[i]) for i in range(self.dim)]
        mesh = np.meshgrid(*axes, indexing='ij')
        return np.stack(mesh, axis=-1).reshape(-1, self.dim)

    def _full_cheby_grid(self, level: int, remove_duplicates: bool = True) -> np.ndarray:
        grids = [self._uni_grid(int(k)) for k in range(level + 1)]

        memo = {}
        valid_levels = self._valid_combinations(self.dim, level, memo)

        grid_points = []
        for levels in valid_levels:
            mesh = np.ix_(*[grids[levels[i]] for i in range(self.dim)])
            grid_points.append(np.stack(np.meshgrid(*mesh, indexing='ij')).reshape(self.dim, -1).T)
        grid = np.concatenate(grid_points, axis=0)
        if remove_duplicates:
            grid = self._remove_duplicates(grid)
        return self._rescale(grid)

    def _valid_combinations(self, d: int, level: int, memo: dict = None):
        if (d, level) in memo:
            return memo[(d, level)]
        if d == 1:
            result = [[k] for k in range(level + 1)]
        else:
            result = []
            for current_level in range(level + 1):
                for sub_combination in self._valid_combinations(int(d - 1), int(level - current_level), memo):
                    result.append([current_level] + sub_combination)
        memo[(d, level)] = result
        return result

    def _uni_grid(self, level: int) -> np.ndarray:
        return np.zeros(1) if level == 0 else self._cheby_nodes(2 ** level + 1)

    @staticmethod
    def _cheby_nodes(n: int) -> np.ndarray:
        arr = np.arange(1, n + 1)
        return (-1) * np.cos(np.pi * (arr - 1) / (n - 1))

    @staticmethod
    def _remove_duplicates(arr: np.ndarray, tol=1e-8):
        """
        Removes duplicate rows whenever they are closer than the tolerance using optimized NumPy operations
        to ensure memory and compute efficiency.
        :param arr: 2D NumPy array from which to remove nearly duplicate rows.
        :param tol: Tolerance for determining "near-duplicate" rows.
        :return: Array with near-duplicates removed.
        """
        if arr.size == 0:
            return arr
        arr = np.unique(arr, axis=0)  # Remove exact duplicates

        # Initialize an empty array for unique rows with a size that will dynamically grow.
        # We initialize with the first row of the input array to avoid handling an empty array.
        unique_rows = arr[:1]

        # Iterate over each row starting from the second row
        for row in arr[1:]:
            # Compute the Euclidean distance from the current row to all rows stored as unique
            diffs = np.linalg.norm(unique_rows - row, axis=1)
            # If no existing unique row is within the tolerance, append the new row to unique_rows
            if not np.any(diffs <= tol):
                unique_rows = np.vstack([unique_rows, row])

        return unique_rows

    def _rescale(self, grid: np.ndarray) -> np.ndarray:
        if self.lower_bound == -1. and self.upper_bound == 1.:
            return grid
        grid = (grid + 1) / 2
        return grid * (self.upper_bound - self.lower_bound) + self.lower_bound
