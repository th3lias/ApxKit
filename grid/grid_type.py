"""
Grid types
"""
from enum import Enum


class GridType(Enum):
    CHEBYSHEV = 1,
    REGULAR = 2,
    RANDOM_UNIFORM = 3,
    RANDOM_CHEBYSHEV = 4
