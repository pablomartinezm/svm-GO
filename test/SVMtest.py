import unittest

import numpy as np
from classes import SVM
from classes import SVMGo

class TestInit(unittest.TestCase):

    def test_add_sv(self):
        clf = SVM()
        clf.X = np.array([[1, 1], [2, 2], [3, 3]])
        clf.y = np.array([1, 1, 1])

        self.assertTrue(clf.add_sv(0))
        self.assertFalse(clf.add_sv(0))
        self.assertEqual(clf.alpha, np.array([1]))
        self.assertEqual(clf._supp_idx, np.array([0]))

    def test_regularization(self):
        clf = SVM()
        clf.alpha = np.array([0.5])
        clf.epsilon = 1
        clf.regularize(1, 1)
        self.assertTrue(clf.alpha == np.array([0]))

    def test_remove_sv(self):
        clf = SVMGo()
        clf._alpha = np.array([1, 1, 0.5, -0.5, 1, -0.1])
        clf._supp_idx = np.array([5, 6, 1, 3, 12, 8])
        clf.remove = 0.5
        self.assertTrue(all(clf._alpha))


if __name__ == '__main__':
    unittest.main()
