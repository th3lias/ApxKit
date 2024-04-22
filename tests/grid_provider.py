import unittest
import numpy as np
from grid import grid_provider
from grid.grid_type import GridType


class TestGridProvider(unittest.TestCase):
    def test_q_d_error(self):
        with self.assertRaises(ValueError):
            provider = grid_provider.GridProvider(np.int8(4), np.zeros(1), np.zeros(1), np.int8(1))

    def test_provider_type_error(self):
        provider = grid_provider.GridProvider(np.int8(4), np.zeros(1), np.zeros(1), np.int8(4))
        with self.assertRaises(ValueError):
            grid = provider.generate(grid_type=1)

    def test_provider_create_chebyshev_x_i(self):
        provider = grid_provider.GridProvider(np.int8(4), np.zeros(1), np.zeros(1), np.int8(4))
        provider.generate(grid_type=GridType.CHEBYSHEV)
        m = provider._generate_m()
        self.assertIsInstance(m, np.ndarray)
        np.testing.assert_array_equal(m, np.array([1, 3, 5, 9], dtype=np.int8))
        # TODO: Expand this test case.


if __name__ == '__main__':
    unittest.main()
