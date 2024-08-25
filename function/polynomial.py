#  Created 2024. (Elias Mindlberger)
import numpy as np

from function.function import Function


class Polynomial(Function):
    """
    This class is intended to represent a polynomial function.
    """

    # Todo: implement abstract basis functionality, e.g. to use Chebyshev polynomials.
    def __init__(self, coefficients: np.ndarray):
        """
        Constructor for the Polynomial class.

        :param coefficients: The coefficients of the polynomial, as a 1d numpy array.
        """
        self.coefficients = coefficients
