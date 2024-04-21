"""
Provides sparse grids and random grids.
"""
import numpy
import numpy as np

from grid.grid_type import GridType


class GridProvider:

    def __init__(self,
                 grid_type: GridType,
                 dimension: np.int8,
                 upper: np.ndarray,
                 lower: np.ndarray,
                 seed: np.int8 = None):
        if not isinstance(grid_type, GridType):
            raise TypeError("Invalid grid type")
        self.grid_type = grid_type

    def generate(self) -> numpy.ndarray:
        """
        Generates a grid according to type and returns it.
        :return:
        """
        if self.grid_type == GridType.REGULAR:
            return self._generate_regular()
        else:
            return self._generate_random()

    def _generate_regular(self):
        raise NotImplementedError

    def _generate_random(self):
        raise NotImplementedError
