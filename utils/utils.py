import numpy as np
import matplotlib.pyplot as plt
from typing import Callable
import pandas as pd


def ell_2_error_estimate(f: Callable, f_hat: Callable, d: np.int8, no_samples: np.int16,
                         lower_bound: np.float64 = np.float64(0.0),
                         upper_bound: np.float64 = np.float64(1.0)) -> np.float64:
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

    error = np.sqrt(np.mean(np.square(np.abs(y - y_hat)))).squeeze()

    return error


def max_abs_error(f: Callable, f_hat: Callable, d: np.int8, no_samples: np.int16,
                  lower_bound: np.float64 = np.float64(0.0),
                  upper_bound: np.float64 = np.float64(1.0)) -> np.float64:
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


def visualize_point_grid_2d(points: np.ndarray, alpha: np.float64) -> None:
    """
    Visualizes a 2D point grid in a scatter plot
    :param points: array that contains the points. Needs to be of shape (n, 2)
    :param alpha: specifies the opacity of the points
    :return:
    """
    if np.shape(points)[1] != 2:
        raise ValueError("points must be a 2-dimensional array")

    x = points[:, 0]
    y = points[:, 1]

    plt.figure(figsize=(10, 10))
    plt.scatter(x, y, color='black', alpha=alpha)
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')

    plt.grid(True)
    plt.show()


def visualize_point_grid_3d(points: np.ndarray, alpha: np.float64) -> None:
    """
        Visualizes a 3D point grid in a scatter plot
        :param points: array that contains the points. Needs to be of shape (n, 3)
        :param alpha: specifies the opacity of the points
        :return:
        """
    if np.shape(points)[1] != 3:
        raise ValueError("points must be a 3-dimensional array")

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, color='black', alpha=alpha, marker='o')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')

    plt.grid(True)
    plt.show()