"""
Grid types
"""
from enum import Enum


class LeastSquaresMethod(Enum):
    EXACT = 1,
    ITERATIVE_LSMR = 2,
    SKLEARN = 3,
    PYTORCH = 4
