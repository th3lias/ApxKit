"""
Grid types
"""
from enum import Enum


class LeastSquaresMethod(Enum):
    EXACT = 1,
    ITERATIVE_LSMR = 2,
    SKLEARN = 3,
    PYTORCH = 4,
    RLS = 5,
    ITERATIVE_RLS = 6

class SmolyakMethod(Enum):
    STANDARD = 1,
    LAGRANGE = 2
