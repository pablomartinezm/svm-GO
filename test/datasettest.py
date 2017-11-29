import unittest

import numpy as np

from classes.svm_new import SVM


class TestInit(unittest.TestCase):

    def test_bank(self):
        from datasets.datasets import get_bank
        X, y = get_bank()
        self.assertTrue(type(X) == np.array)

    def test_regularization(self):
        clf = SVM()
        clf.alpha = np.array([0.5])
        clf.epsilon = 1
        clf.regularize(1, 1)
        self.assertTrue(clf.alpha == np.array([0]))




if __name__ == '__main__':
    unittest.main()
