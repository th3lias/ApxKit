import numpy as np

from solver.solver import Solver
from scipy.linalg import lstsq


class ScipyLstsqSolver(Solver):
    """Least-squares solver using scipy.linalg.lstsq (LAPACK backend)."""

    def __init__(self, driver: str = 'gelsy'):
        super().__init__(f"SCIPY_LSTSQ_{driver.capitalize()}", driver.capitalize())
        self.driver = driver

    def solve(self, A: np.ndarray, y: np.ndarray):
        coeff, *_ = lstsq(A, y, lapack_driver=self.driver)
        return coeff
