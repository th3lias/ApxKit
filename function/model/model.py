from typing import Callable

from function.f import Function


class Model(Function):
    """
        This interface defines how a model, i.e. an approximation of a Function should behave.
    """

    def __init__(self, f: Callable, dim: int, upper: float, lower: float):
        """
            In this case, 'f: Callable' __is__ the function on which an approximation was carried out, if there
            was no approximation performed, this parameter should be None.
        """
        super().__init__(f, dim, upper, lower)
