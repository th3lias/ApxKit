import unittest
import numpy as np
from grid.grid_provider import GridProvider
from grid.grid_type import GridType
from utils import utils


class TestGridProvider(unittest.TestCase):

    def test_performance_of_duplicates(self):
        provider = GridProvider(np.int8(4), np.float16(1), np.float16(-1))
        grid = provider.generate(grid_type=GridType.CHEBYSHEV, scale=np.int8(4), remove_duplicates=False)
        print(f"{grid.shape}")

        @utils.timeit
        def _naive_implementation(arr):
            for i in range(10):
                grid_copy = utils._remove_almost_identical_rows(arr)
            print(f"{grid_copy.shape}")
            return None

        @utils.timeit
        def _numpy_implementation(arr):
            for i in range(10):
                grid_copy = provider._remove_duplicates(arr)
            print(f"{grid_copy.shape}")
            return None

        print("Naive implementation.")
        _naive_implementation(grid)
        print("Pure numpy implementation")
        _numpy_implementation(grid)

    def test_provider_type_error(self):
        provider = GridProvider(np.int8(4), np.float16(1), np.float16(0))
        with self.assertRaises(ValueError):
            provider.generate(grid_type='1')


class VisualTests(unittest.TestCase):

    @staticmethod
    def test_equidistant_provider_2d():
        provider = GridProvider(np.int8(2), np.float16(1), np.float16(0))
        grid = provider.generate(grid_type=GridType.REGULAR, scale=np.int8(16))
        utils.visualize_point_grid_2d(grid, alpha=np.int8(1))

    @staticmethod
    def test_random_provider_3d():
        provider = GridProvider(np.int8(3), np.float16(1), np.float16(0), seed=np.int8(42))
        grid = provider.generate(grid_type=GridType.RANDOM, scale=np.int32(16))
        utils.visualize_point_grid_3d(grid, alpha=np.int8(1))

    @staticmethod
    def test_equidistant_provider_3d():
        provider = GridProvider(np.int8(3), np.float16(1), np.float16(0))
        grid = provider.generate(grid_type=GridType.REGULAR, scale=np.int8(16))
        utils.visualize_point_grid_3d(grid, alpha=np.int8(1))

    @staticmethod
    def test_random_provider_2d():
        provider = GridProvider(np.int8(2), np.float16(1), np.float16(0), seed=np.int8(42))
        grid = provider.generate(grid_type=GridType.RANDOM, scale=np.int32(16))
        utils.visualize_point_grid_2d(grid, alpha=np.int8(1))

    @staticmethod
    def test_chebyshev_2d():
        provider = GridProvider(np.int8(2), np.float16(1), np.float16(0), seed=np.int8(42))
        grid = provider.generate(grid_type=GridType.CHEBYSHEV, scale=np.int8(5))
        utils.visualize_point_grid_2d(grid, alpha=np.int8(1))
        return True

    @staticmethod
    def test_chebyshev_3d():
        provider = GridProvider(np.int8(3), np.float16(1), np.float16(0), seed=np.int8(42))
        grid = provider.generate(grid_type=GridType.CHEBYSHEV, scale=np.int8(5))
        utils.visualize_point_grid_3d(grid, alpha=np.int8(1))


if __name__ == '__main__':
    unittest.main()
