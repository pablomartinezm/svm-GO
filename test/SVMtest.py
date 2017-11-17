import unittest

import numpy as np

from classes.svm_new import SVM


class TestInit(unittest.TestCase):

    def test_add_sv(self):
        clf = SVM()
        clf.X = np.array([[1, 1], [2, 2], [3, 3]])
        clf.y = np.array([1, 1, 1])

        self.assertTrue(clf.add_sv(0))
        self.assertFalse(clf.add_sv(0))
        self.assertEqual(clf.alpha, np.array([1]))
        self.assertEqual(clf._supp_idx, np.array([0]))

    def test_random_vectors(self):
        pass



if __name__ == '__main__':
    unittest.main()
