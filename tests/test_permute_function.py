import math
import unittest
from interpolate.smolyak import SmolyakInterpolator
import numpy as np
from deprecated import deprecated


class PermutationTests(unittest.TestCase):

    @deprecated
    def test_array_size10(self):
        size = 10
        array = np.arange(size)

        gen = SmolyakInterpolator._permute(array)

        self.assertTrue(np.all(next(gen) == array))

        next(gen)
        next(gen)
        next(gen)

        self.assertTrue(np.all(next(gen) == np.array([0, 1, 2, 3, 4, 5, 6, 9, 7, 8])))

    @deprecated
    def test_length_size10(self):
        size = 10
        array = np.arange(size)

        gen = SmolyakInterpolator._permute(array)

        s = sum(1 for _ in gen)

        self.assertEqual(s, math.factorial(size))

    @deprecated
    def test_array_size3(self):
        size = 3
        array = np.arange(size)

        gen = SmolyakInterpolator._permute(array)

        self.assertTrue(np.all(next(gen) == array))

        next(gen)
        next(gen)
        next(gen)

        self.assertTrue(np.all(next(gen) == np.array([2, 0, 1])))

    @deprecated
    def test_length_size3(self):
        size = 3
        array = np.arange(size)

        gen = SmolyakInterpolator._permute(array)

        s = sum(1 for _ in gen)

        self.assertEqual(s, math.factorial(size))

    @deprecated
    def test_array_drop_duplicate(self):
        array = np.array([1, 3, 4, 2, 2, 2])

        gen = SmolyakInterpolator._permute(array, drop_duplicates=True)

        self.assertTrue(np.all(next(gen) == np.sort(array)))

        for i in range(110):
            _ = next(gen)

        self.assertTrue(np.all(next(gen) == np.array([4, 2, 2, 3, 1, 2])))

    @deprecated
    def test_length_drop_duplicate(self):
        array = np.array([1, 3, 4, 2, 2, 2])

        size = array.shape[0]

        gen = SmolyakInterpolator._permute(array, drop_duplicates=True)

        s = sum(1 for _ in gen)

        self.assertEqual(s, math.factorial(size) / math.factorial(3))

    @deprecated
    def test_array_keep_duplicate(self):
        array = np.array([1, 3, 4, 2, 2, 2])

        gen = SmolyakInterpolator._permute(array, drop_duplicates=False)

        self.assertTrue(np.all(next(gen) == np.sort(array)))

        for i in range(110):
            _ = next(gen)

        self.assertTrue(np.all(next(gen) == np.array([1, 4, 2, 2, 3, 2])))

    @deprecated
    def test_length_keep_duplicate(self):
        array = np.array([1, 3, 4, 2, 2, 2])

        size = array.shape[0]

        gen = SmolyakInterpolator._permute(array, drop_duplicates=False)

        s = sum(1 for _ in gen)

        self.assertEqual(s, math.factorial(size))


if __name__ == '__main__':
    unittest.main()
