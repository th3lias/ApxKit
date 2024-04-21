import numpy as np
from typing import Callable


def ell_2_error_estimate(f: Callable, f_hat: Callable, d: np.int, no_samples: int, lower_bound: np.float=np.float(0.0),
                         upper_bound: np.float=np.float(1.0)) -> np.float:
    """
    Calculates the ell_2 error estimate by sampling no_samples points in [lower_bound, upper_bound]^d and
    calculating the mean-squared absolute difference
    :param f: function that should be approximated
    :param f_hat: approximation of the function
    :param d: dimension of the function
    :param no_samples: number of samples
    :param lower_bound: lower bound of the rectangle to sample from
    :param upper_bound: upper bound of the rectangle to sample from
    :return: error estimate
    """
    points = np.random.uniform(low=lower_bound, high=upper_bound, size=(no_samples, d))

    y_hat = f_hat(points)
    y = f(points)

    error = np.mean(np.square(np.abs(y - y_hat))).squeeze()

    return error


def max_abs_error(f: Callable, f_hat: Callable, d: np.int, no_samples: int, lower_bound: np.float,
                  upper_bound: np.float) -> np.float:
    """
        Calculates the estimated max absolute value distance by sampling no_samples points in
        [lower_bound, upper_bound]^d and calculating the max absolute value difference at those points
        :param f: function that should be approximated
        :param f_hat: approximation of the function
        :param d: dimension of the function
        :param no_samples: number of samples
        :param lower_bound: lower bound of the rectangle to sample from
        :param upper_bound: upper bound of the rectangle to sample from
        :return: error estimate
        """

    points = np.random.uniform(low=lower_bound, high=upper_bound, size=(no_samples, d))
    y_hat = f_hat(points)
    y = f(points)

    error = np.max(np.abs(y_hat - y)).squeeze()

    return error
