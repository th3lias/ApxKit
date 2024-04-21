"""
Provides sparse grid and random grid.
"""
from grid_type import GridType


class GridProvider:
    def __init__(self, grid_type: GridType):
        if not isinstance(grid_type, GridType):
            raise TypeError("Invalid grid type")
        self.grid_type = grid_type

    def generate(self):
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
