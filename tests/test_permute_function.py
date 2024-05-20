import math
import unittest
from interpolate.smolyak import SmolyakInterpolator
import numpy as np


class PermutationTests(unittest.TestCase):

    def test_array_size10(self):
        size = 10
        array = np.arange(size)

        gen = SmolyakInterpolator._permute(array)

        self.assertTrue(np.all(next(gen) == array))

        next(gen)
        next(gen)
        next(gen)

        self.assertTrue(np.all(next(gen) == np.array([0,1,2,3,4,5,6,9,7,8])))

    def test_length_size10(self):
        size = 10
        array = np.arange(size)

        gen = SmolyakInterpolator._permute(array)

        s = sum(1 for _ in gen)

        self.assertEqual(s, math.factorial(size))

    def test_array_size3(self):
        size = 3
        array = np.arange(size)

        gen = SmolyakInterpolator._permute(array)

        self.assertTrue(np.all(next(gen) == array))

        next(gen)
        next(gen)
        next(gen)

        self.assertTrue(np.all(next(gen) == np.array([2,0,1])))

    def test_length_size3(self):
        size = 3
        array = np.arange(size)

        gen = SmolyakInterpolator._permute(array)

        s = sum(1 for _ in gen)

        self.assertEqual(s, math.factorial(size))


if __name__ == '__main__':
    unittest.main()
