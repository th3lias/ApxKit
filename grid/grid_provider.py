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
                 upper_bound: np.ndarray,
                 lower_bound: np.ndarray,
                 q: np.int8,
                 seed: np.int8 = None):
        self.dim = dimension
        self.q = q
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.seed = seed
        if self.q < self.dim: raise ValueError(f"fineness must at least exceed dimension")

    def generate(self, grid_type: GridType) -> np.ndarray:
        """
        Generate a grid of given type.
        :param grid_type: GridType specification, e.g. Chebyshev, Random or Equidistant
        :return: np.ndarray representing the grid
        """
        if not isinstance(grid_type, GridType):
            raise ValueError(f"grid type not supported: {grid_type}")
        return self._generate_chebyshev_grid()

    def _generate_random_grid(self):
        pass

    def _generate_equidistant_grid(self):
        pass

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
        arr[1:] = - np.cos(np.pi * (arr[1:]-1)/(m_i-1))
        return arr
