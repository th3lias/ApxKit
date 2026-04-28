import unittest

import numpy as np

from grid.generator.chebyshev_grid_generator import ChebyshevGridGenerator
from grid.generator.rule_grid_generator import RuleGridGenerator
from grid.generator.selection_strategy import SelectionStrategy
from grid.generator.uniform_grid_generator import UniformGridGenerator
from grid.rule.rule_grid_rule import RuleGridRule
from grid.rule.sparse_grid_type import SparseGridType
from plot.visualize_grid import visualize_point_grid_2d, visualize_point_grid_3d
from utils.utils import calculate_num_points


class UniformGridGeneratorTest(unittest.TestCase):

    def test_reproducibility_multiple_generators(self):
        gen1 = UniformGridGenerator(seed=42)
        gen2 = UniformGridGenerator(seed=42)

        grid1 = gen1.get_grid(2, 3)
        grid2 = gen2.get_grid(2, 3)

        self.assertTrue(np.allclose(grid1, grid2), "Grids generated with the same seed should be equal.")

    def test_reproducibility_same_generator(self):
        gen = UniformGridGenerator(seed=42)

        grid1 = gen.get_grid(2, 3)
        grid2 = gen.get_grid(2, 3)

        self.assertTrue(np.allclose(grid1, grid2), "Grids generated with the same seed should be equal.")

    def test_reshuffle(self):
        gen = UniformGridGenerator(seed=42)

        grid1 = gen.get_grid(10, 2)
        gen.reshuffle()
        grid2 = gen.get_grid(10, 2)

        self.assertFalse(np.allclose(grid1, grid2), "Grids generated after reshuffle should not be equal.")

    def test_upper_lower_bounds1(self):
        gen = UniformGridGenerator(seed=42)

        lower = 0.0
        upper = 1.0

        grid = gen.get_grid(10, 2, lower_bound=lower, upper_bound=upper)

        self.assertTrue(np.all((grid.grid >= lower) & (grid.grid <= upper)),
                        "All points should be within the specified bounds.")

    def test_upper_lower_bounds2(self):
        gen = UniformGridGenerator(seed=42)

        lower = -1.0
        upper = 1.0

        grid = gen.get_grid(10, 2, lower_bound=lower, upper_bound=upper)

        self.assertTrue(np.all((grid.grid >= lower) & (grid.grid <= upper)),
                        "All points should be within the specified bounds.")

    def test_upper_lower_bounds3(self):
        gen = UniformGridGenerator(seed=42)

        lower = -3.0
        upper = -1.0

        grid = gen.get_grid(10, 2, lower_bound=lower, upper_bound=upper)

        self.assertTrue(np.all((grid.grid >= lower) & (grid.grid <= upper)),
                        "All points should be within the specified bounds.")

    def test_upper_lower_bounds4(self):
        gen = UniformGridGenerator(seed=42)

        lower = 2.0
        upper = 4.0

        grid = gen.get_grid(10, 2, lower_bound=lower, upper_bound=upper)

        self.assertTrue(np.all((grid.grid >= lower) & (grid.grid <= upper)),
                        "All points should be within the specified bounds.")

    def test_seeded_grid(self):
        gen1 = UniformGridGenerator(seed=42)
        gen2 = UniformGridGenerator(seed=420)

        grid1 = gen1.get_grid(10, 2)
        grid2 = gen2.get_grid(10, 2)

        self.assertFalse(np.allclose(grid1, grid2), "Grids generated with different seeds should not be equal.")

    def test_add_points_correct_number(self):
        gen = UniformGridGenerator(seed=42)

        gen.get_grid(3, 4)
        new_grid = gen._increase_scale(3, 4, delta=5)

        self.assertEqual(new_grid.get_num_points(), calculate_num_points(dim=3, scale=9),
                         "Total number of points after adding should be correct.")

    def test_add_points_correct_values(self):
        gen = UniformGridGenerator(seed=42)

        initial_grid = gen.get_grid(2, 5)
        new_grid = gen._increase_scale(2, 5, delta=2)

        self.assertTrue(gen._is_subset(initial_grid.grid, new_grid.grid),
                        "New grid should contain all points from the initial grid plus new points.")

    def test_add_points_reproducibility(self):
        gen = UniformGridGenerator(seed=42)
        dim = 2
        scale = 3
        delta = 5
        gen.get_grid(dim, scale)
        new_grid = gen._increase_scale(dim, scale, 5)

        second_grid = gen.get_grid(dim, scale + delta)

        self.assertTrue(np.allclose(new_grid, second_grid),
                        "The grid after adding points should match the grid generated with the new total number of points.")

    def test_add_points_no_initial_grid(self):
        gen = UniformGridGenerator(seed=42)

        with self.assertRaises(ValueError):
            gen._increase_scale(1, 1, 5)

    def test_add_points_number_already_exists(self):
        gen = UniformGridGenerator(seed=42)

        grid1 = gen.get_grid(2, 7)
        grid2 = gen.get_grid(2, 3)

        grid3 = gen._increase_scale(2, 3, 4)

        grid4 = gen.get_grid(2, 7)

        self.assertFalse(np.allclose(grid1, grid3),
                         "Adding points should not return the same grid if the number of points already exists.")
        self.assertTrue(np.allclose(grid3, grid4),
                        "The grid after adding points should match the grid generated with the new total number of points.")


class ChebyshevGridGeneratorTest(unittest.TestCase):

    def test_reproducibility_multiple_generators(self):
        gen1 = ChebyshevGridGenerator(seed=42)
        gen2 = ChebyshevGridGenerator(seed=42)

        grid1 = gen1.get_grid(2, 3)
        grid2 = gen2.get_grid(2, 3)

        self.assertTrue(np.allclose(grid1, grid2), "Grids generated with the same seed should be equal.")

    def test_reproducibility_same_generator(self):
        gen = ChebyshevGridGenerator(seed=42)

        grid1 = gen.get_grid(2, 3)
        grid2 = gen.get_grid(2, 3)

        self.assertTrue(np.allclose(grid1, grid2), "Grids generated with the same seed should be equal.")

    def test_reshuffle(self):
        gen = ChebyshevGridGenerator(seed=42)

        grid1 = gen.get_grid(10, 2)
        gen.reshuffle()
        grid2 = gen.get_grid(10, 2)

        self.assertFalse(np.allclose(grid1, grid2), "Grids generated after reshuffle should not be equal.")

    def test_upper_lower_bounds1(self):
        gen = ChebyshevGridGenerator(seed=42)

        lower = 0.0
        upper = 1.0

        grid = gen.get_grid(10, 2, lower_bound=lower, upper_bound=upper)

        self.assertTrue(np.all((grid.grid >= lower) & (grid.grid <= upper)),
                        "All points should be within the specified bounds.")

    def test_upper_lower_bounds2(self):
        gen = ChebyshevGridGenerator(seed=42)

        lower = -1.0
        upper = 1.0

        grid = gen.get_grid(10, 2, lower_bound=lower, upper_bound=upper)

        self.assertTrue(np.all((grid.grid >= lower) & (grid.grid <= upper)),
                        "All points should be within the specified bounds.")

    def test_upper_lower_bounds3(self):
        gen = ChebyshevGridGenerator(seed=42)

        lower = -3.0
        upper = -1.0

        grid = gen.get_grid(10, 2, lower_bound=lower, upper_bound=upper)

        self.assertTrue(np.all((grid.grid >= lower) & (grid.grid <= upper)),
                        "All points should be within the specified bounds.")

    def test_upper_lower_bounds4(self):
        gen = ChebyshevGridGenerator(seed=42)

        lower = 2.0
        upper = 4.0

        grid = gen.get_grid(10, 2, lower_bound=lower, upper_bound=upper)

        self.assertTrue(np.all((grid.grid >= lower) & (grid.grid <= upper)),
                        "All points should be within the specified bounds.")

    def test_seeded_grid(self):
        gen1 = ChebyshevGridGenerator(seed=42)
        gen2 = ChebyshevGridGenerator(seed=420)

        grid1 = gen1.get_grid(10, 2)
        grid2 = gen2.get_grid(10, 2)

        self.assertFalse(np.allclose(grid1, grid2), "Grids generated with different seeds should not be equal.")

    def test_add_points_correct_number(self):
        gen = ChebyshevGridGenerator(seed=42)

        gen.get_grid(3, 4)
        new_grid = gen._increase_scale(3, 4, delta=5)

        self.assertEqual(new_grid.get_num_points(), calculate_num_points(dim=3, scale=9),
                         "Total number of points after adding should be correct.")

    def test_add_points_correct_values(self):
        gen = ChebyshevGridGenerator(seed=42)

        initial_grid = gen.get_grid(2, 5)
        new_grid = gen._increase_scale(2, 5, delta=2)

        self.assertTrue(gen._is_subset(initial_grid.grid, new_grid.grid),
                        "New grid should contain all points from the initial grid plus new points.")

    def test_add_points_reproducibility(self):
        gen = ChebyshevGridGenerator(seed=42)
        dim = 2
        scale = 3
        delta = 5
        gen.get_grid(dim, scale)
        new_grid = gen._increase_scale(dim, scale, 5)

        second_grid = gen.get_grid(dim, scale + delta)

        self.assertTrue(np.allclose(new_grid, second_grid),
                        "The grid after adding points should match the grid generated with the new total number of points.")

    def test_add_points_no_initial_grid(self):
        gen = ChebyshevGridGenerator(seed=42)

        with self.assertRaises(ValueError):
            gen._increase_scale(1, 1, 5)

    def test_add_points_number_already_exists(self):
        gen = ChebyshevGridGenerator(seed=42)

        grid1 = gen.get_grid(2, 7)
        grid2 = gen.get_grid(2, 3)

        grid3 = gen._increase_scale(2, 3, 4)

        grid4 = gen.get_grid(2, 7)

        self.assertFalse(np.allclose(grid1, grid3),
                         "Adding points should not return the same grid if the number of points already exists.")
        self.assertTrue(np.allclose(grid3, grid4),
                        "The grid after adding points should match the grid generated with the new total number of points.")


@unittest.skip("Visual tests are not run by default, uncomment to run them.")
class VisualTests(unittest.TestCase):

    @staticmethod
    def test_visual_random_uniform_generator_2d():
        generator = UniformGridGenerator(42)
        grid = generator.get_grid(dim=2, scale=6, lower_bound=0.0, upper_bound=1.0)
        visualize_point_grid_2d(grid, alpha=1.)

    @staticmethod
    def test_visual_random_uniform_generator_3d():
        generator = UniformGridGenerator(42)
        grid = generator.get_grid(dim=3, scale=6, lower_bound=0.0, upper_bound=1.0)
        visualize_point_grid_3d(grid, alpha=1.)

    @staticmethod
    def test_visual_random_chebyshev_generator_2d():
        generator = ChebyshevGridGenerator(42)
        grid = generator.get_grid(dim=2, scale=6, lower_bound=0.0, upper_bound=1.0)
        visualize_point_grid_2d(grid, alpha=1.)

    @staticmethod
    def test_visual_random_chebyshev_generator_3d():
        generator = ChebyshevGridGenerator(42)
        grid = generator.get_grid(dim=3, scale=6, lower_bound=0.0, upper_bound=1.0)
        visualize_point_grid_3d(grid, alpha=1.)

    @staticmethod
    def test_visual_rule_grid_generator_2d():
        generator = RuleGridGenerator()
        grid = generator.get_grid(input_dim=2, scale=4, lower=0.0, upper=1.0, strategy=SelectionStrategy.LEVEL,
                                  rule=RuleGridRule.CLENSHAW_CURTIS, sparse_grid_type=SparseGridType.STANDARD_GLOBAL)
        grid = grid.get_needed_points()
        visualize_point_grid_2d(grid, alpha=1.)

    @staticmethod
    def test_visual_rule_grid_generator_3d():
        generator = RuleGridGenerator()
        grid = generator.get_grid(input_dim=3, scale=4, lower=0.0, upper=1.0, strategy=SelectionStrategy.LEVEL,
                                  rule=RuleGridRule.CLENSHAW_CURTIS, sparse_grid_type=SparseGridType.STANDARD_GLOBAL)
        grid = grid.get_needed_points()
        visualize_point_grid_3d(grid, alpha=1.)

    @staticmethod
    def test_visual_rule_grid_generator_lower_upper_2d():
        generator = RuleGridGenerator()
        grid = generator.get_grid(input_dim=2, scale=4, lower=-3.0, upper=1.0, strategy=SelectionStrategy.LEVEL,
                                  rule=RuleGridRule.CLENSHAW_CURTIS, sparse_grid_type=SparseGridType.STANDARD_GLOBAL)
        grid = grid.get_needed_points()
        visualize_point_grid_2d(grid, alpha=1.)

    @staticmethod
    def test_visual_rule_grid_generator_lower_upper_3d():
        generator = RuleGridGenerator()
        grid = generator.get_grid(input_dim=3, scale=4, lower=5.0, upper=7.0, strategy=SelectionStrategy.LEVEL,
                                  rule=RuleGridRule.CLENSHAW_CURTIS, sparse_grid_type=SparseGridType.STANDARD_GLOBAL)
        grid = grid.get_needed_points()
        visualize_point_grid_3d(grid, alpha=1.)


class NumberOfPointsTests(unittest.TestCase):

    def test_2d_scale_6(self):
        dim = 2
        scale = 6
        generator = RuleGridGenerator()
        grid = generator.get_grid(input_dim=dim, scale=scale, lower=0.0, upper=1.0, strategy=SelectionStrategy.LEVEL,
                                  rule=RuleGridRule.CLENSHAW_CURTIS, sparse_grid_type=SparseGridType.STANDARD_GLOBAL)
        n_points = calculate_num_points(dim, scale)
        self.assertEqual(grid.get_num_points(), n_points)

    def test_4d_scale_1(self):
        dim = 4
        scale = 1
        generator = RuleGridGenerator()
        grid = generator.get_grid(input_dim=dim, scale=scale, lower=0.0, upper=1.0, strategy=SelectionStrategy.LEVEL,
                                  rule=RuleGridRule.CLENSHAW_CURTIS, sparse_grid_type=SparseGridType.STANDARD_GLOBAL)
        n_points = calculate_num_points(dim, scale)
        self.assertEqual(grid.get_num_points(), n_points)

    def test_12d_scale_4(self):
        dim = 12
        scale = 4
        n_points = calculate_num_points(dim, scale)
        generator = RuleGridGenerator()
        grid = generator.get_grid(input_dim=dim, scale=scale, lower=0.0, upper=1.0, strategy=SelectionStrategy.LEVEL,
                                  rule=RuleGridRule.CLENSHAW_CURTIS, sparse_grid_type=SparseGridType.STANDARD_GLOBAL)
        self.assertEqual(grid.get_num_points(), n_points)

    def test_3d_scale_5(self):
        dim = 3
        scale = 5
        generator = RuleGridGenerator()
        grid = generator.get_grid(input_dim=dim, scale=scale, lower=0.0, upper=1.0, strategy=SelectionStrategy.LEVEL,
                                  rule=RuleGridRule.CLENSHAW_CURTIS, sparse_grid_type=SparseGridType.STANDARD_GLOBAL)
        n_points = calculate_num_points(dim, scale)
        self.assertEqual(grid.get_num_points(), n_points)

    def test_8d_scale_4(self):
        dim = 8
        scale = 4
        generator = RuleGridGenerator()
        grid = generator.get_grid(input_dim=dim, scale=scale, lower=0.0, upper=1.0, strategy=SelectionStrategy.LEVEL,
                                  rule=RuleGridRule.CLENSHAW_CURTIS, sparse_grid_type=SparseGridType.STANDARD_GLOBAL)
        n_points = calculate_num_points(dim, scale)
        self.assertEqual(grid.get_num_points(), n_points)

    def test_7d_scale_1(self):
        dim = 7
        scale = 1
        generator = RuleGridGenerator()
        grid = generator.get_grid(input_dim=dim, scale=scale, lower=0.0, upper=1.0, strategy=SelectionStrategy.LEVEL,
                                  rule=RuleGridRule.CLENSHAW_CURTIS, sparse_grid_type=SparseGridType.STANDARD_GLOBAL)
        n_points = calculate_num_points(dim, scale)
        self.assertEqual(grid.get_num_points(), n_points)


if __name__ == '__main__':
    unittest.main()
