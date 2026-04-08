import numpy as np
import matplotlib.pyplot as plt
from grid.grid.grid import Grid


def visualize_point_grid_1d(points: Grid | np.ndarray, alpha: float) -> None:
    """
    Visualizes a set of points in a histogram
    :param points: array that contains the points.
    :param alpha: specifies the opacity of the points
    :return: None
    """

    if isinstance(points, Grid):
        points = points.grid

    if len(points.shape) == 1:
        # 1D points
        plt.figure(figsize=(10, 6))
        plt.hist(points, bins=30, color='black', alpha=alpha)
        plt.xlabel('$x$')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.show()
    else:
        raise ValueError(f"Wrong dimension of the data. Expected dimension 1, got {points.ndim}")


def visualize_point_grid_2d(points: Grid | np.ndarray, alpha: float) -> None:
    """
    Visualizes a 2D point grid in a scatter plot
    :param points: array that contains the points. Needs to be of shape (n, 2)
    :param alpha: specifies the opacity of the points
    :return:
    """
    if isinstance(points, Grid):
        points = points.grid
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


def visualize_point_grid_3d(points: Grid | np.ndarray, alpha: float) -> None:
    """
        Visualizes a 3D point grid in a scatter plot
        :param points: array that contains the points. Needs to be of shape (n, 3)
        :param alpha: specifies the opacity of the points
        :return:
        """

    if isinstance(points, Grid):
        points = points.grid
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
