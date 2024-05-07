"""
Provides sparse grids and random grids.
"""
import numpy as np

from grid.grid_type import GridType


class GridProvider:
    """
    Provides sparse grids, random grids and grids sampled from the chebyshev extrema.
    :param dimension: dimension of the function to be approximated
    :param upper_bound: upper bound of the domain of each function in the tensor product to be approximated as array
    :param lower_bound: lower bound of the domain of each function in the tensor product as np.ndarray
    :param seed: random seed to be used when option RANDOM is used
    """

    def __init__(self, dimension: np.int8, upper_bound: np.float16, lower_bound: np.float16, seed: np.int8 = None):
        self.dim = dimension
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        self.mid_point = (self.upper_bound - self.lower_bound) / 2
        self.avg = (self.upper_bound + self.lower_bound) / 2

        if isinstance(seed, np.int8) or isinstance(seed, int):
            self.seed = seed
            self.rng = np.random.default_rng(seed=seed)
        else:
            self.rng = np.random.default_rng(seed=None)

    def set_seed(self, seed: np.int8):
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def generate(self, grid_type: GridType, scale: np.int32 = None, remove_duplicates: bool = True) -> np.ndarray:
        """
        Generate a grid of given type.
        :param scale:  Number of points (per dimension!) when generating the grid equidistantly or randomly.
        If GridType == RANDOM we aim to have the same number of points as in the regular
        grid and therefore sample the same num_points**dim number of points uniformly. If
        GridType == CHEBYSHEV we use the scale parameter to determine the fineness of the sparse grid.
        :param grid_type: GridType specification, e.g. Chebyshev, Random or Equidistant.
        :param remove_duplicates: Remove duplicate values in the grid up to a small distance. Needed for Smolyak.
        :return: np.ndarray representing the grid
        """
        if not isinstance(grid_type, GridType):
            raise ValueError("grid type not supported: " + str(grid_type))
        if grid_type == GridType.CHEBYSHEV:
            if scale is None:
                raise ValueError("Please provide the level of fineness of the chebyshev grid.")
            return self._full_cheby_grid(level=scale, remove_duplicates=remove_duplicates)
        else:
            if scale is None:
                raise ValueError("Please provide how many points to generate in each subspace.")
            if grid_type == GridType.REGULAR:
                return self._generate_equidistant_grid(num_points=scale)
            if grid_type == GridType.RANDOM:
                return self._generate_random_grid(num_points=scale)

    def _generate_random_grid(self, num_points: np.int32) -> np.ndarray:
        return self.rng.uniform(low=self.lower_bound, high=self.upper_bound, size=(num_points, self.dim))

    def _generate_equidistant_grid(self, num_points: np.int32) -> np.ndarray:
        num_points = np.full(shape=self.dim, fill_value=num_points)
        axes = [np.linspace(self.lower_bound, self.upper_bound, num_points[i]) for i in range(self.dim)]
        mesh = np.meshgrid(*axes, indexing='ij')
        return np.stack(mesh, axis=-1).reshape(-1, self.dim)

    def _full_cheby_grid(self, level: np.int32, remove_duplicates: bool = True) -> np.ndarray:
        grids = [self._uni_grid(np.int32(k)) for k in range(level + 1)]

        memo = {}
        valid_levels = self._valid_combinations(self.dim, level, memo)

        grid_points = []
        for levels in valid_levels:
            mesh = np.ix_(*[grids[levels[i]] for i in range(self.dim)])
            grid_points.append(np.stack(np.meshgrid(*mesh, indexing='ij')).reshape(self.dim, -1).T)
        concat_grid = np.concatenate(grid_points, axis=0)
        unique_grid = np.unique(concat_grid, axis=0)
        return self._remove_duplicates(unique_grid) if remove_duplicates else unique_grid

    def _valid_combinations(self, d: np.int32, level: np.int32, memo: dict = None):
        if (d, level) in memo:
            return memo[(d, level)]
        if d == 1:
            result = [[k] for k in range(level + 1)]
        else:
            result = []
            for current_level in range(level + 1):
                for sub_combination in self._valid_combinations(np.int32(d - 1), np.int32(level - current_level), memo):
                    result.append([current_level] + sub_combination)
        memo[(d, level)] = result
        return result

    def _uni_grid(self, level: np.int32) -> np.ndarray:
        return np.zeros(1) if level == 0 else self._cheby_nodes(2 ** level + 1)

    @staticmethod
    def _cheby_nodes(n: np.int8) -> np.ndarray:
        arr = np.arange(1, n + 1)
        return (-1) * np.cos(np.pi * (arr - 1) / (n - 1))

    @staticmethod
    def _remove_duplicates(arr: np.ndarray, tol: np.float32=np.float32(1e-8)):
        """
        Removes duplicate rows whenever they are closer than the tolerance
        :param arr:
        :param tol:
        :return:
        """
        if arr.size == 0:
            return arr
        diffs = np.sqrt(((arr[:, np.newaxis] - arr[np.newaxis, :]) ** 2).sum(axis=2))
        close = diffs <= tol
        not_dominated = ~np.any(np.triu(close, k=1), axis=0)
        unique_rows = arr[not_dominated]
        return unique_rows
