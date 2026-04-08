from grid.generator.grid_generator import GridGenerator
from grid.grid.random_grid import RandomGrid


class RandomGridGenerator(GridGenerator):
    """
    Base class for random grid generators.

    Provides caching and reshuffling so that the same (dim, scale) request
    returns identical points, while ``reshuffle()`` forces new draws.
    """

    def __init__(self, name: str, abbr_name: str, seed=42):
        super().__init__(name, abbr_name)
        self.seed = seed
        self._cache = dict()
        self._reshuffle_count = 0
        self._current_config = None

    def reshuffle(self) -> None:
        """Invalidate cached grids so subsequent calls draw fresh points."""
        self._reshuffle_count += 1
        self._cache.clear()
        self._current_config = None

    def first_time(self):
        return len(self._cache) == 0

    def _generate_seed(self, n_points: int, dim: int, scale, lower_bound: float = 0., upper_bound: float = 1.) -> int:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def get_grid(self, dim: int, scale: int, lower_bound: float = 0., upper_bound: float = 1.) -> RandomGrid:
        raise NotImplementedError("This method should be implemented in subclasses.")

    def _increase_scale(self, dim: int, scale: int, lower: float, upper: float, delta: int) -> RandomGrid:
        """Extend an existing cached grid by appending points for a higher scale."""
        raise NotImplementedError("This method should be implemented in subclasses.")

    @staticmethod
    def _is_subset(a, b):
        """Check whether every row in array *a* is also present in *b*."""
        b_rows_set = set(map(tuple, b))
        for row in a:
            if tuple(row) not in b_rows_set:
                return False
        return True
