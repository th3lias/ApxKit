import numpy as np
from typing import Callable


def ell_2_error_estimate(f:Callable, f_hat:Callable, d:int, no_samples:int) -> float:
    # generate random points
    points = np.random.uniform(low=0.0, high=1.0, size=(no_samples, d))

    # function evaluations
    y_hat = f_hat(points)
    y = f(points)

    # Error Estimate
    error = np.mean(np.square(np.abs(y-y_hat))).squeeze()

    return error

def max_abs_error(f:Callable, f_hat:Callable, d:int, no_samples:int) -> float:
    # generate random points
    points = np.random.uniform(low=0.0, high=1.0, size=(no_samples, d))

    # function evaluations
    y_hat = f_hat(points)
    y = f(points)

    # max absolute value difference
    error = np.max(np.abs(y_hat-y)).squeeze()

    return error



