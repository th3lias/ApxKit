import jax.numpy as jnp
import numpy as np

from grid.grid_type import GridType


class Grid:
    def __init__(self, dim: np.int8, scale: np.int32, grid: jnp.ndarray | np.ndarray, grid_type: GridType,
                 lower_bound: np.float16 = np.float16(-1.), upper_bound: np.float16 = np.float16(1.)):
        self.dim = dim
        self.scale = scale
        self.grid = grid
        self.grid_type = grid_type
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def get_num_points(self):
        return self.grid.shape[0]

    def jax(self):
        self.grid = jnp.array(self.grid)

    def numpy(self):
        self.grid = self.grid.asnumpy()

    def get_grid(self) -> jnp.ndarray | np.ndarray:
        return self.grid

    def __eq__(self, other):
        return (self.dim, self.scale, self.grid_type, self.lower_bound, self.upper_bound) == \
               (other.dim, other.scale, other.grid_type, other.lower_bound, other.upper_bound)

    def __hash__(self):
        return hash((self.dim, self.scale, self.grid_type, self.lower_bound, self.upper_bound))
