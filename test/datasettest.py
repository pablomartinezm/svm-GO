import unittest

import numpy as np

from .. import SVM
from .. import SVMGo


class TestInit(unittest.TestCase):

    def test_regularization(self):
        clf = SVM()
        clf.alpha = np.array([0.5])
        clf.epsilon = 1
        clf.regularize(1, 1)
        self.assertTrue(clf.alpha == np.array([0]))



if __name__ == '__main__':
    unittest.main()
