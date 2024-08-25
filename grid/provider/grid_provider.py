#  Created 2024. (Elias Mindlberger)
from TasmanianSG import TasmanianSparseGrid

from grid.src.grid import Grid


class GridProvider:
    def __init__(self, input_dim: int, output_dim: int, lower_bound: float, upper_bound: float):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def generate(self, scale: int) -> TasmanianSparseGrid | Grid:
        raise NotImplementedError("This method must be implemented by a subclass.")

    def increase_scale(self, grid: Grid | TasmanianSparseGrid, delta: int) -> TasmanianSparseGrid | Grid:
        raise NotImplementedError("This method must be implemented by a subclass.")
