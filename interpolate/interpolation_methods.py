"""
Grid types
"""
from enum import Enum


class LeastSquaresMethod(Enum):
    EXACT = 1,
    NUMPY_LSTSQ = 2,


class SmolyakMethod(Enum):
    STANDARD = 1,
    LAGRANGE = 2
