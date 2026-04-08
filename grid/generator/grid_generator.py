from grid.grid.grid import Grid


class GridGenerator:
    """Abstract base for grid generators."""

    def __init__(self, name: str, abbr_name: str):
        self.name = name
        self.abbr_name = abbr_name

    def get_grid(self, **kwargs) -> Grid:
        raise NotImplementedError("This method should be implemented in subclasses.")
