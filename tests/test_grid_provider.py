import unittest

import utils.test_utils as test_utils
from grid.grid_provider import GridProvider
from grid.grid_type import GridType
from utils import utils


class TestGridProvider(unittest.TestCase):

    def test_duplicate_removal(self):
        provider = GridProvider(4)
        grid = provider.generate(grid_type=GridType.CHEBYSHEV, scale=4, remove_duplicates=False)
        print(f"{grid.grid.shape}")

        print("Naive Python implementation.")
        new_grid = utils.test_function_time(test_utils._remove_almost_identical_rows, 10, grid.grid)
        print(f"{new_grid.shape}")
        print("Numpy implementation - n^2 memory")
        new_grid = utils.test_function_time(test_utils._remove_duplicates_squared_memory, 10, grid.grid)
        print(f"{new_grid.shape}")
        print("Numpy implementation - linear memory")
        new_grid = utils.test_function_time(test_utils._remove_duplicates_linear_memory_naive, 10, grid.grid)
        print(f"{new_grid.shape}")
        print("Numpy implementation - optimised linear memory")
        new_grid = utils.test_function_time(provider._remove_duplicates, 10, grid.grid)
        print(f"{new_grid.shape}")

    # noinspection PyTypeChecker
    def test_provider_type_error(self):
        provider = GridProvider(int(4))
        with self.assertRaises(ValueError):
            provider.generate(grid_type='1')


class VisualTests(unittest.TestCase):

    @staticmethod
    def test_equidistant_provider_2d():
        provider = GridProvider(int(2))
        grid = provider.generate(grid_type=GridType.REGULAR, scale=int(5))
        utils.visualize_point_grid_2d(grid, alpha=1.)

    @staticmethod
    def test_equidistant_provider_3d():
        provider = GridProvider(int(3))
        grid = provider.generate(grid_type=GridType.REGULAR, scale=int(3))
        utils.visualize_point_grid_3d(grid, alpha=1.)

    @staticmethod
    def test_random_provider_2d():
        provider = GridProvider(int(2), seed=int(42))
        grid = provider.generate(grid_type=GridType.RANDOM, scale=int(3))
        utils.visualize_point_grid_2d(grid, alpha=1.)

    @staticmethod
    def test_random_provider_3d():
        provider = GridProvider(int(3), seed=int(42))
        grid = provider.generate(grid_type=GridType.RANDOM, scale=int(5))
        utils.visualize_point_grid_3d(grid, alpha=1.)

    @staticmethod
    def test_chebyshev_2d():
        provider = GridProvider(int(2), seed=int(42))
        grid = provider.generate(grid_type=GridType.CHEBYSHEV, scale=int(5))
        utils.visualize_point_grid_2d(grid, alpha=1.)

    @staticmethod
    def test_chebyshev_2d_custom_range():
        provider = GridProvider(int(2), seed=int(42), lower_bound=1., upper_bound=4.)
        grid = provider.generate(grid_type=GridType.CHEBYSHEV, scale=int(5))
        utils.visualize_point_grid_2d(grid, alpha=1.)

    @staticmethod
    def test_chebyshev_3d():
        provider = GridProvider(int(3), seed=int(42))
        grid = provider.generate(grid_type=GridType.CHEBYSHEV, scale=int(5))
        utils.visualize_point_grid_3d(grid, alpha=1.)

    @staticmethod
    def test_chebyshev_3d_custom_range():
        provider = GridProvider(int(3), seed=int(42), lower_bound=1., upper_bound=4.)
        grid = provider.generate(grid_type=GridType.CHEBYSHEV, scale=int(5))
        utils.visualize_point_grid_3d(grid, alpha=1.)


if __name__ == '__main__':
    unittest.main()
