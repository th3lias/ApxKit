from basis.basis import Basis
from grid.grid.grid import Grid


class BasisGenerator:
    """Abstract base for classes that build a basis matrix from a grid."""

    def __init__(self, name: str, abbr_name: str):
        self.name = name
        self.abbr_name = abbr_name

    def create_basis(self, grid: Grid, scale:int) -> Basis:
        raise NotImplementedError("Subclasses should implement this method")

    def _basis_rule(self):
        raise NotImplementedError("Subclasses should implement this method")
