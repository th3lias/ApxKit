import unittest

import numpy as np

from grid.grid_provider import GridProvider
from grid.grid_type import GridType
from utils import utils


class TestGridProvider(unittest.TestCase):

    def test_duplicate_removal(self):
        provider = GridProvider(np.int8(4))
        grid = provider.generate(grid_type=GridType.CHEBYSHEV, scale=np.int8(4), remove_duplicates=False)
        print(f"{grid.shape}")

        print("Naive Python implementation.")
        new_grid = utils.test_function_time(utils._remove_almost_identical_rows, 10, grid)
        print(f"{new_grid.shape}")
        print("Numpy implementation - n^2 memory")
        new_grid = utils.test_function_time(utils._remove_duplicates_squared_memory, 10, grid)
        print(f"{new_grid.shape}")
        print("Numpy implementation - linear memory")
        new_grid = utils.test_function_time(utils._remove_duplicates_linear_memory_naive, 10, grid)
        print(f"{new_grid.shape}")
        print("Numpy implementation - optimised linear memory")
        new_grid = utils.test_function_time(provider._remove_duplicates, 10, grid)
        print(f"{new_grid.shape}")

    def test_provider_type_error(self):
        provider = GridProvider(np.int8(4))
        with self.assertRaises(ValueError):
            provider.generate(grid_type='1')


class VisualTests(unittest.TestCase):

    @staticmethod
    def test_equidistant_provider_2d():
        provider = GridProvider(np.int8(2))
        grid = provider.generate(grid_type=GridType.REGULAR, scale=np.int8(16))
        utils.visualize_point_grid_2d(grid, alpha=np.int8(1))

    @staticmethod
    def test_random_provider_3d():
        provider = GridProvider(np.int8(3), seed=np.int8(42))
        grid = provider.generate(grid_type=GridType.RANDOM, scale=np.int32(16))
        utils.visualize_point_grid_3d(grid, alpha=np.int8(1))

    @staticmethod
    def test_equidistant_provider_3d():
        provider = GridProvider(np.int8(3))
        grid = provider.generate(grid_type=GridType.REGULAR, scale=np.int8(16))
        utils.visualize_point_grid_3d(grid, alpha=np.int8(1))

    @staticmethod
    def test_random_provider_2d():
        provider = GridProvider(np.int8(2), seed=np.int8(42))
        grid = provider.generate(grid_type=GridType.RANDOM, scale=np.int32(16))
        utils.visualize_point_grid_2d(grid, alpha=np.int8(1))

    @staticmethod
    def test_chebyshev_2d():
        provider = GridProvider(np.int8(2), seed=np.int8(42))
        grid = provider.generate(grid_type=GridType.CHEBYSHEV, scale=np.int8(5))
        utils.visualize_point_grid_2d(grid, alpha=np.int8(1))

    @staticmethod
    def test_chebyshev_2d_custom_range():
        provider = GridProvider(np.int8(2), seed=np.int8(42),  lower_bound=np.float16(1.), upper_bound=np.float16(4.))
        grid = provider.generate(grid_type=GridType.CHEBYSHEV, scale=np.int8(5))
        utils.visualize_point_grid_2d(grid, alpha=np.int8(1))

    @staticmethod
    def test_chebyshev_3d():
        provider = GridProvider(np.int8(3), seed=np.int8(42))
        grid = provider.generate(grid_type=GridType.CHEBYSHEV, scale=np.int8(5))
        utils.visualize_point_grid_3d(grid, alpha=np.int8(1))

    @staticmethod
    def test_chebyshev_3d_custom_range():
        provider = GridProvider(np.int8(3), seed=np.int8(42), lower_bound=np.float16(1.), upper_bound=np.float16(4.))
        grid = provider.generate(grid_type=GridType.CHEBYSHEV, scale=np.int8(5))
        utils.visualize_point_grid_3d(grid, alpha=np.int8(1))


if __name__ == '__main__':
    unittest.main()
