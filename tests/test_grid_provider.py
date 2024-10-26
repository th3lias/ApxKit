
# TODO [Jakob] Adapt the tests to also work with the new structure and not rely on the "interpolate" package anymore

import unittest

from grid.provider.random_grid_provider import RandomGridProvider
from grid.rule.random_grid_rule import RandomGridRule
from grid.provider.rule_grid_provider import RuleGridProvider
from utils.utils import visualize_point_grid_2d, visualize_point_grid_3d, visualize_point_grid_1d, calculate_num_points


class VisualTests(unittest.TestCase):

    @staticmethod
    def test_random_uniform_provider_2d():
        provider = RandomGridProvider(2)
        grid = provider.generate(scale=3)
        visualize_point_grid_2d(grid, alpha=1.)

    @staticmethod
    def test_random_uniform_provider_3d():
        provider = RandomGridProvider(3)
        grid = provider.generate(scale=int(5))
        visualize_point_grid_3d(grid, alpha=1.)

    @staticmethod
    def test_random_uniform_density_provider_1d():
        points = RandomGridProvider(1).generate(scale=5)
        visualize_point_grid_1d(points, alpha=1.)

    @staticmethod
    def test_random_chebyshev_density_provider_1d():
        points = RandomGridProvider(1, rule=RandomGridRule.CHEBYSHEV).generate(scale=5)
        visualize_point_grid_1d(points, alpha=1.)

    @staticmethod
    def test_random_chebyshev_density_provider_2d():
        provider = RandomGridProvider(2, rule=RandomGridRule.CHEBYSHEV)
        grid = provider.generate(scale=int(8))
        visualize_point_grid_2d(grid, alpha=1.)

    @staticmethod
    def test_random_chebyshev_density_provider_3d():
        provider = RandomGridProvider(3)
        grid = provider.generate(scale=int(7))
        visualize_point_grid_3d(grid, alpha=1.)

    @staticmethod
    def test_chebyshev_2d():
        provider = RandomGridProvider(2, rule=RandomGridRule.CHEBYSHEV)
        grid = provider.generate(scale=5)
        visualize_point_grid_2d(grid, alpha=1.)

    @staticmethod
    def test_chebyshev_2d_custom_range():
        provider = RandomGridProvider(2, seed=42, lower_bound=1., upper_bound=4., rule=RandomGridRule.CHEBYSHEV)
        grid = provider.generate(scale=5)
        visualize_point_grid_2d(grid, alpha=1.)

    @staticmethod
    def test_chebyshev_3d():
        provider = RandomGridProvider(3, seed=42)
        grid = provider.generate(scale=int(5))
        visualize_point_grid_3d(grid, alpha=1.)

    @staticmethod
    def test_chebyshev_3d_custom_range():
        provider = RandomGridProvider(3, lower_bound=1., upper_bound=4.)
        grid = provider.generate(scale=int(5))
        visualize_point_grid_3d(grid, alpha=1.)


class TestNumberGridPoints(unittest.TestCase):

    def test_2d_scale_6(self):
        dim = 2
        scale = 6
        provider = RuleGridProvider(dim)
        grid = provider.generate(scale=scale)
        n_points = calculate_num_points(scale, dim)
        self.assertEqual(grid.get_num_points(), n_points)

    def test_4d_scale_1(self):
        dim = 4
        scale = 1
        provider = RuleGridProvider(dim)
        grid = provider.generate(scale=scale)
        n_points = calculate_num_points(scale, dim)
        self.assertEqual(grid.get_num_points(), n_points)

    def test_12d_scale_4(self):
        dim = 12
        scale = 4
        provider = RuleGridProvider(dim)
        grid = provider.generate(scale=scale)
        n_points = calculate_num_points(scale, dim)
        self.assertEqual(grid.get_num_points(), n_points)

    def test_3d_scale_5(self):
        dim = 3
        scale = 5
        provider = RuleGridProvider(dim)
        grid = provider.generate(scale=scale)
        n_points = calculate_num_points(scale, dim)
        self.assertEqual(grid.get_num_points(), n_points)

    def test_8d_scale_4(self):
        dim = 8
        scale = 4
        provider = RuleGridProvider(dim)
        grid = provider.generate(scale=scale)
        n_points = calculate_num_points(scale, dim)
        self.assertEqual(grid.get_num_points(), n_points)

    def test_7d_scale_1(self):
        dim = 7
        scale = 1
        provider = RuleGridProvider(dim)
        grid = provider.generate(scale=scale)
        n_points = calculate_num_points(scale, dim)
        self.assertEqual(grid.get_num_points(), n_points)


if __name__ == '__main__':
    unittest.main()
