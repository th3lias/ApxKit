"""
Grid types
"""
from enum import Enum


class LeastSquaresMethod(Enum):
    EXACT = 1,
    ITERATIVE_LSMR = 2,
    SKLEARN = 3,
    PYTORCH = 4,
    PYTORCH_NEURAL_NET = 5,
    JAX_NEURAL_NET = 6,
    RLS = 7,
    ITERATIVE_RLS = 8


class SmolyakMethod(Enum):
    STANDARD = 1,
    LAGRANGE = 2
