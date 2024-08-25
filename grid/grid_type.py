"""
Grid types
"""
from enum import Enum


class GridType(Enum):
    RULE_BASED = 0,
    CHEBYSHEV = 1,
    REGULAR = 2,
    RANDOM_UNIFORM = 3,
    RANDOM_CHEBYSHEV = 4
