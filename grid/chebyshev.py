"""
This class provides the chebyshev polynomials and their extrema
"""
from typing import Callable

import numpy as np


class Chebyshev:

    def __init__(self):
        pass

    @staticmethod
    def t0(self):
        return 1

    @staticmethod
    def t1(self, x):
        return x

    @staticmethod
    def t(n, x):
        return np.cos(n*np.arccos(x))

    @staticmethod
    def recurrence(self, tnx: np.float16, tn1x: np.float16, x: np.float16) -> np.float16:
        return np.float16(2.*x*tnx-tn1x)

    @staticmethod
    def functional_recurrence(self, tn: Callable, tn1: Callable, x: np.float16) -> np.float16:
        return np.float16(2.*x*tn(x)*tn1(x))
