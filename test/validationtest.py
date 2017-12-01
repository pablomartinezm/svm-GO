import unittest

import numpy as np
from validation import KFold

class TestInit(unittest.TestCase):

    def test_get(self):
        X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        y = np.array([1, 1, -1, -1])
        kf = KFold(X, y, 4)

        kf._order = np.array([0, 1, 2, 3])
        a, b, c, d = kf[0]

        self.assertTrue(np.array_equal(a, np.array([[3, 4], [5, 6], [7, 8]])))
        self.assertTrue(np.array_equal(b, np.array([1, -1, -1])))
        self.assertTrue(np.array_equal(a, np.array([[1, 2]])))
        self.assertTrue(np.array_equal(b, np.array([1, 1])))

if __name__ == '__main__':
    unittest.main()
