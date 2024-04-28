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
    :param q: fineness scale of the grid
    :param seed: random seed to be used when option RANDOM is used
    """
    def __init__(self,
                 dimension: np.int8,
                 upper_bound: np.float16,
                 lower_bound: np.float16,
                 seed: np.int8 = None):
        self.dim = dimension
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound

        self.mid_point = (self.upper_bound - self.lower_bound) / 2
        self.avg = (self.upper_bound + self.lower_bound) / 2

        if isinstance(seed, np.int8) or isinstance(seed, int):
            self.seed = seed
            self.rng = np.random.default_rng(seed=seed)

    def set_seed(self, seed: np.int8):
        self.seed = seed
        self.rng = np.random.default_rng(seed=seed)

    def generate(self, grid_type: GridType, num_points: np.int64 = None) -> np.ndarray:
        """
        Generate a grid of given type.
        :param num_points:  Number of points (per dimension!) when generating the grid equidistantly or randomly.
                            If GridType == EQUIDISTANT we sample uniformly per dimension, i.e. we have num_points**dim
                            points. If GridType == RANDOM we aim to have the same number of points as in the regular
                            grid and therefore sample the same num_points**dim number of points uniformly.
        :param grid_type: GridType specification, e.g. Chebyshev, Random or Equidistant.
        :return: np.ndarray representing the grid
        """
        if not isinstance(grid_type, GridType):
            raise ValueError(f"grid type not supported: {grid_type}")
        if grid_type == GridType.CHEBYSHEV:
            return self._generate_chebyshev_grid()
        else:
            if num_points is None:
                raise ValueError(f"Please provide how many points to generate subspace.")
            if grid_type == GridType.REGULAR:
                return self._generate_equidistant_grid(num_points=num_points)
            if grid_type == GridType.RANDOM:
                return self._generate_random_grid(num_points=num_points)

    def _generate_random_grid(self, num_points: np.int64) -> np.ndarray:
        return self.rng.uniform(low=self.lower_bound, high=self.upper_bound, size=(num_points ** self.dim, self.dim))

    def _generate_equidistant_grid(self, num_points: np.int64) -> np.ndarray:
        lower = [self.lower_bound] * self.dim
        upper = [self.upper_bound] * self.dim
        num_points = [num_points] * self.dim

        axes = [np.linspace(lower[i], upper[i], num_points[i]) for i in range(self.dim)]

        mesh = np.meshgrid(*axes, indexing='ij')
        return np.stack(mesh, axis=-1).reshape(-1, self.dim)

    def _generate_chebyshev_grid(self):
        m = self._generate_m()
        x = np.zeros(shape=(len(m), m[-1]))
        # TODO: Vectorise this.
        for i in range(1, len(m)):
            x[i, :m[i]] = self._generate_x(m[i])
        return x

    def _generate_m(self) -> np.ndarray:
        arr = np.arange(self.dim, dtype=np.int32)+1
        arr[1:] = 2**(arr[1:]-1)+1
        return arr

    @staticmethod
    def _generate_x(m_i: np.int32):
        arr = np.arange(m_i, dtype=np.float32)+1
        arr[0] = np.float32(0.0)
        # needs to be tested
        # the following line incorporates a rescaling to the interval [-1, 1]
        # arr[1:] = self.avg + self.mid_point * np.cos(np.pi * (2 * (m_i - arr[1:]) - 1) / (2 * m_i))
        arr[1:] = - np.cos(np.pi * (arr[1:]-1)/(m_i-1))
        return arr
