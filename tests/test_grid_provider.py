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
    def test_random_uniform_provider_2d():
        provider = GridProvider(int(2), seed=int(42))
        grid = provider.generate(grid_type=GridType.RANDOM_UNIFORM, scale=int(3))
        utils.visualize_point_grid_2d(grid, alpha=1.)

    @staticmethod
    def test_random_uniform_provider_3d():
        provider = GridProvider(int(3), seed=int(42))
        grid = provider.generate(grid_type=GridType.RANDOM_UNIFORM, scale=int(5))
        utils.visualize_point_grid_3d(grid, alpha=1.)

    @staticmethod
    def test_random_chebyshev_density_provider_1d():
        points = GridProvider(dimension=1, seed=int(42))._sample_chebyshev_univariate(num_points=1000)
        utils.visualize_point_grid_1d(points, alpha=1.)

    @staticmethod
    def test_random_chebyshev_density_provider_2d():
        provider = GridProvider(int(2), seed=int(42))
        grid = provider.generate(grid_type=GridType.RANDOM_CHEBYSHEV, scale=int(8))
        utils.visualize_point_grid_2d(grid, alpha=1.)

    @staticmethod
    def test_random_chebyshev_density_provider_3d():
        provider = GridProvider(int(3), seed=int(42))
        grid = provider.generate(grid_type=GridType.RANDOM_CHEBYSHEV, scale=int(7))
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


class TestNumberGridPoints(unittest.TestCase):

    def test_2d_scale_6(self):
        dim = 2
        scale = 6
        provider = GridProvider(dim)
        grid = provider.generate(grid_type=GridType.CHEBYSHEV, scale=scale)
        n_points = utils.calculate_num_points(scale, dim)

        self.assertEqual(grid.get_num_points(), n_points)

    def test_4d_scale_1(self):
        dim = 4
        scale = 1
        provider = GridProvider(dim)
        grid = provider.generate(grid_type=GridType.CHEBYSHEV, scale=scale)
        n_points = utils.calculate_num_points(scale, dim)

        self.assertEqual(grid.get_num_points(), n_points)

    def test_12d_scale_4(self):
        dim = 12
        scale = 4
        provider = GridProvider(dim)
        grid = provider.generate(grid_type=GridType.CHEBYSHEV, scale=scale)
        n_points = utils.calculate_num_points(scale, dim)

        self.assertEqual(grid.get_num_points(), n_points)

    def test_3d_scale_5(self):
        dim = 3
        scale = 5
        provider = GridProvider(dim)
        grid = provider.generate(grid_type=GridType.CHEBYSHEV, scale=scale)
        n_points = utils.calculate_num_points(scale, dim)

        self.assertEqual(grid.get_num_points(), n_points)

    def test_8d_scale_4(self):
        dim = 8
        scale = 4
        provider = GridProvider(dim)
        grid = provider.generate(grid_type=GridType.CHEBYSHEV, scale=scale)
        n_points = utils.calculate_num_points(scale, dim)

        self.assertEqual(grid.get_num_points(), n_points)

    def test_7d_scale_1(self):
        dim = 7
        scale = 1
        provider = GridProvider(dim)
        grid = provider.generate(grid_type=GridType.CHEBYSHEV, scale=scale)
        n_points = utils.calculate_num_points(scale, dim)

        self.assertEqual(grid.get_num_points(), n_points)

if __name__ == '__main__':
    unittest.main()
