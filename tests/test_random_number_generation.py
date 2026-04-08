import unittest

import numpy as np

from grid.generator.uniform_number_generator import UniformNumberGenerator


class UniformNumberGeneratorTest(unittest.TestCase):
    def test_reproducibility(self):
        gen1 = UniformNumberGenerator(seed=42)
        gen2 = UniformNumberGenerator(seed=42)

        array_1 = gen1.get_random_numbers(10, 5, lower_bound=0.0, upper_bound=1.0)
        array_2 = gen2.get_random_numbers(10, 5, lower_bound=0.0, upper_bound=1.0)

        self.assertTrue(np.allclose(array_1, array_2), "Arrays should be equal for the same seed.")

    def test_reproducibility_reshuffle(self):
        gen1 = UniformNumberGenerator(seed=42)

        array_1 = gen1.get_random_numbers(10, 5, lower_bound=0.0, upper_bound=1.0)
        gen1.reshuffle()
        
        array_2 = gen1.get_random_numbers(10, 5, lower_bound=0.0, upper_bound=1.0)

        self.assertFalse(np.allclose(array_1, array_2), "Arrays should not be equal after reshuffling.")

    def test_reproducibility_different_seeds(self):
        gen1 = UniformNumberGenerator(seed=42)
        gen2 = UniformNumberGenerator(seed=43)

        array_1 = gen1.get_random_numbers(10, 5, lower_bound=0.0, upper_bound=1.0)
        array_2 = gen2.get_random_numbers(10, 5, lower_bound=0.0, upper_bound=1.0)

        self.assertFalse(np.allclose(array_1, array_2), "Arrays should not be equal for different seeds.")


    def test_grid_bounds(self):
        gen1 = UniformNumberGenerator(seed=42)
        array = gen1.get_random_numbers(10, 5, lower_bound=0.0, upper_bound=1.0)
        self.assertTrue(np.all(array >= 0.0) and np.all(array <= 1.0),
                        "All values should be within the specified bounds [0.0, 1.0].")


    def test_grid_bounds_negative(self):
        gen1 = UniformNumberGenerator(seed=42)
        array = gen1.get_random_numbers(10, 5, lower_bound=-1.0, upper_bound=1.0)
        self.assertTrue(np.all(array >= -1.0) and np.all(array <= 1.0),
                        "All values should be within the specified bounds [-1.0, 1.0].")

    def test_grid_shape(self):
        gen1 = UniformNumberGenerator(seed=42)
        array = gen1.get_random_numbers(10, 5, lower_bound=0.0, upper_bound=1.0)
        self.assertEqual(array.shape, (10, 5), "The shape of the generated array should be (10, 5).")


if __name__ == '__main__':
    unittest.main()
