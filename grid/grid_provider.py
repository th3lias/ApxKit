"""
Provides sparse grids and random grids.
"""
import numpy as np
from deprecated import deprecated

from grid.grid import Grid
from grid.grid_type import GridType
from utils.utils import calculate_num_points

from typing import Union
from interpolate.partition import Partition
from interpolate.interpolator import Interpolator


class GridProvider:
    """
    Provides sparse grids, random grids and grids sampled from the chebyshev extrema.
    :param dimension: dimension of the function to be approximated
    :param seed: random seed to be used when option RANDOM is used
    """

    def __init__(self, dimension: int, seed: int = None, lower_bound: float = 0.0, upper_bound: float = 1.0):
        self.dim = dimension
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.multiplier = 1.0

        if isinstance(seed, int):
            self.seed = seed
            self.rng = np.random.default_rng(seed=seed)
        else:
            self.rng = np.random.default_rng(seed=None)

    def set_seed(self, seed: int):
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def generate(self, grid_type: GridType, scale: int = None, multiplier: float = 1.0) -> Grid:
        """
        Generate a grid of given type.
        :param scale: Determines the number of points to be used in positive correlation.
        :param grid_type: GridType specification, e.g. Chebyshev, Random or Equidistant.
        :param multiplier: Only used in Random-Grid; Increases the number of samples by the given multiplier
        :return: grid object containing the points
        """
        self.multiplier = multiplier

        if not isinstance(grid_type, GridType):
            raise ValueError("grid type not supported: " + str(grid_type))
        if scale is None:
            raise ValueError("Please provide the fineness parameter of the grid")

        if grid_type == GridType.CHEBYSHEV:
            if multiplier != 1.0:
                print("Be aware that the chosen multiplier for a Chebyshev Sparse Grid does not affect anything")
            points = self._full_cheby_grid(level=scale)
            return Grid(self.dim, scale, points, grid_type)

        n_points = calculate_num_points(scale, self.dim)  # TODO: Maybe not right
        n_points = int(n_points * multiplier)

        if grid_type == GridType.REGULAR:
            raise DeprecationWarning("The regular grid is deprecated and is most likely not working correctly.")
            # We leave this for a possible fix in the future
            # points = self._generate_equidistant_grid(num_points=n_points)
            # return Grid(self.dim, scale, points, grid_type)
        if grid_type == GridType.RANDOM_UNIFORM:
            points = self._generate_random_grid(num_points=n_points)
            return Grid(self.dim, scale, points, grid_type)
        if grid_type == GridType.RANDOM_CHEBYSHEV:
            points = self._generate_with_chebyshev_density(num_points=n_points)
            return Grid(self.dim, scale, points, grid_type)

    def increase_scale(self, current_grid: Union[Grid, np.ndarray], sample_new: bool = False,
                       scale: Union[None, int] = None,
                       grid_type: Union[GridType, None] = None, delta: int = 1) -> Grid:
        """
        Takes a given grid and increases the number of smaples. This has great performance benefits for Chebyshev grids
        :param current_grid: grid that should be enlarged
        :param sample_new: defines whether for non-chebyshev grids the new samples should be added or if it should be
        sampled completely new
        :param scale: current scale of the grid. If None, it is required to have current_grid as a Grid-type, where the
        scale parameter is then drawn
        :param grid_type: specifies which grid type we have
        :param delta: defines how much the scale should be increased
        :return: new grid with new number of samples
        """
        # TODO: Maybe change default behaviour for sample_new

        if isinstance(current_grid, Grid):
            scale = current_grid.scale
            dim = current_grid.dim
            n_points = current_grid.get_num_points()
            grid_type = current_grid.grid_type
            current_grid = current_grid.grid
        else:
            if current_grid is None:
                raise ValueError("Current grid is not allowed to be None")
            if grid_type is None:
                raise ValueError(
                    "grid_type is not allowed to be None, when passing a np.ndarray as current_grid parameter ")
            if scale is None:
                raise ValueError(
                    "scale is not allowed to be None, when passing a np.ndarray as current_grid parameter")
            dim = current_grid.shape[1]
            n_points = current_grid.shape[0]

        if grid_type == GridType.CHEBYSHEV:
            partitions = Partition(dim, dim + scale + delta).get_all_partitions()
            points = list()
            for parti in partitions:
                cart_array = list()
                for d in range(1, dim + 1):
                    j = parti[d - 1]
                    m_i = Interpolator._m_i(j)  # TODO: Maybe make a method in utils
                    c_nodes = self._cheby_nodes(m_i)
                    cart_array.append(c_nodes)
                points.append(self._cartesian_product(cart_array))

            additional_points = np.unique(np.vstack(points), axis=0)

            additional_points = self._rescale(additional_points, lower_bound=self.lower_bound,
                                              upper_bound=self.upper_bound)

            combined_grid = np.unique(np.vstack([current_grid, additional_points]), axis=0)

            return Grid(dim, scale + delta, combined_grid, grid_type, self.lower_bound, self.upper_bound)

        elif grid_type == GridType.RANDOM_UNIFORM:
            target_no_points = int(calculate_num_points(scale=scale + delta, dimension=dim) * self.multiplier)
            if sample_new:
                points = self._generate_random_grid(target_no_points)
                return Grid(dim, scale + delta, points, grid_type, self.lower_bound, self.upper_bound)
            else:
                while n_points < target_no_points:
                    no_new_points = target_no_points - n_points

                    new_points = np.random.uniform(low=self.lower_bound, high=self.upper_bound,
                                                   size=(no_new_points, dim))

                    current_grid = np.unique(np.vstack([current_grid, new_points]), axis=0)
                    n_points = current_grid.shape[0]
                return Grid(dim, scale + delta, current_grid, grid_type, self.lower_bound, self.upper_bound)

        elif grid_type == GridType.RANDOM_CHEBYSHEV:
            target_no_points = int(calculate_num_points(scale=scale + delta, dimension=dim) * self.multiplier)
            if sample_new:
                points = self._generate_with_chebyshev_density(target_no_points)
                return Grid(dim, scale + delta, points, grid_type, self.lower_bound, self.upper_bound)
            else:
                while n_points < target_no_points:
                    no_new_points = target_no_points - n_points

                    new_points = self._generate_with_chebyshev_density(no_new_points)

                    current_grid = np.unique(np.vstack([current_grid, new_points]), axis=0)
                    n_points = current_grid.shape[0]

                return Grid(dim, scale + delta, current_grid, grid_type, self.lower_bound, self.upper_bound)

        elif grid_type == GridType.REGULAR:
            raise ValueError(f"The Gridtype {GridType.REGULAR.name} is currently not supported")

        else:
            raise ValueError(f"Wrong argument. Expected GridTypes")

    def _generate_random_grid(self, num_points: int, precision: int = 8) -> np.ndarray:
        grid = self.rng.uniform(low=self.lower_bound, high=self.upper_bound, size=(num_points, self.dim))
        return np.round(grid, decimals=precision)

    def _generate_with_chebyshev_density(self, num_points: int) -> np.ndarray:
        samples = np.empty(shape=(num_points, self.dim))

        for i in range(self.dim):
            samples[:, i] = self._sample_chebyshev_univariate(num_points)

        return samples

    @deprecated
    def _generate_equidistant_grid(self, num_points: int) -> np.ndarray:
        num_points = np.full(shape=self.dim, fill_value=num_points)
        axes = [np.linspace(self.lower_bound, self.upper_bound, num_points[i]) for i in range(self.dim)]
        mesh = np.meshgrid(*axes, indexing='ij')
        return np.stack(mesh, axis=-1).reshape(-1, self.dim)

    def _full_cheby_grid(self, level: int) -> np.ndarray:
        grids = [self._uni_grid(int(k)) for k in range(level + 1)]

        memo = {}
        valid_levels = self._valid_combinations(self.dim, level, memo)

        grid_points = []
        for levels in valid_levels:
            mesh = np.ix_(*[grids[levels[i]] for i in range(self.dim)])
            grid_points.append(np.stack(np.meshgrid(*mesh, indexing='ij')).reshape(self.dim, -1).T)
        grid = np.unique(np.concatenate(grid_points, axis=0), axis=0)

        return self._rescale(grid, lower_bound=self.lower_bound, upper_bound=self.upper_bound)

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
    def _rescale(grid: np.ndarray, lower_bound: float, upper_bound: float) -> np.ndarray:

        # TODO: Maybe adapt in a way such that the density is correct for example chebyshev weight function
        if lower_bound == -1. and upper_bound == 1.:
            return grid
        grid = (grid + 1) / 2
        return grid * (upper_bound - lower_bound) + lower_bound

    @staticmethod
    def _cartesian_product(array_list: list, dtype: np.dtype = np.float64):
        la = len(array_list)
        arr = np.empty([len(a) for a in array_list] + [la], dtype=dtype)
        for i, a in enumerate(np.ix_(*array_list)):
            arr[..., i] = a
        return arr.reshape(-1, la)

    @staticmethod
    def _sample_chebyshev_univariate(num_points: int, lower_bound: float = 0.0, upper_bound: float = 1.0) -> np.ndarray:
        """Uses the inverse transform method. CDF is arcsin(x) and the inverse is sin(x)"""
        points = np.random.uniform(low=-np.pi / 2, high=np.pi / 2, size=num_points)
        return GridProvider._rescale(grid=np.sin(points), lower_bound=lower_bound, upper_bound=upper_bound)

    @staticmethod
    def _cheby_nodes(n: int, n_decimals: int = 13) -> np.ndarray:
        """Generates the zeros of the n-th univariate chebyshev polynomial"""
        if n == 1:
            return np.array([0.0])
        arr = np.arange(1, n + 1)
        nodes = np.around((-1) * np.cos(np.pi * (arr - 1) / (n - 1)), decimals=n_decimals)
        return nodes
