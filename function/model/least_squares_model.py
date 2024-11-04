from typing import Callable

import numpy as np

from function.model.model import Model


class LeastSquaresModel(Model):
    def __init__(self, f: Callable, dim: int, upper: float, lower: float):
        super(LeastSquaresModel, self).__init__(f, dim, upper, lower)
        self.beta = None
 
    def set_solution(self, solution: np.ndarray):
        self.beta = solution
    
    def __call__(self, x: np.ndarray):
        pass
