import unittest
from utils.utils import calculate_num_points


class TestNumberChebyshevPoints(unittest.TestCase):
    """
    Tests the calculate_num_points method by checking various values.
    Values are based on table of page 5 from https://people.math.sc.edu/Burkardt/presentations/sgmga_counting.pdf
    """

    def test_dim1(self):
        goal = [3, 5, 9, 17, 33, 65, 129, 257, 513, 1025]
        results = []

        for scale in range(1, 11):
            points = calculate_num_points(scale=scale, dimension=1)
            results.append(points)

        combined_results = zip(goal, results)

        for i, (y, y_hat) in enumerate(combined_results):
            self.assertEqual(y, y_hat, msg=f"For scale {i} the number of points should be {y} but is {y_hat}")

    def test_dim2(self):
        goal = [5, 13, 29, 65, 145, 321, 705, 1537, 3329, 7169]
        results = []

        for scale in range(1, 11):
            points = calculate_num_points(scale=scale, dimension=2)
            results.append(points)

        combined_results = zip(goal, results)

        for i, (y, y_hat) in enumerate(combined_results):
            self.assertEqual(y, y_hat, msg=f"For scale {i} the number of points should be {y} but is {y_hat}")

    def test_dim5(self):
        goal = [11, 61, 241, 801, 2433, 6993, 19313, 51713, 135073, 345665]
        results = []

        for scale in range(1, 11):
            points = calculate_num_points(scale=scale, dimension=5)
            results.append(points)

        combined_results = zip(goal, results)

        for i, (y, y_hat) in enumerate(combined_results):
            self.assertEqual(y, y_hat, msg=f"For scale {i} the number of points should be {y} but is {y_hat}")

    def test_dim10(self):
        goal = [21, 221, 1581, 8801, 41265, 171425, 652065, 2320385, 7836545, 25370753]
        results = []

        for scale in range(1, 11):
            points = calculate_num_points(scale=scale, dimension=10)
            results.append(points)

        combined_results = zip(goal, results)

        for i, (y, y_hat) in enumerate(combined_results):
            self.assertEqual(y, y_hat, msg=f"For scale {i} the number of points should be {y} but is {y_hat}")


if __name__ == '__main__':
    unittest.main()
