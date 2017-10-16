import unittest
from ..svm import SVM
import numpy as np



class TestInit(unittest.TestCase):
    def test_input_creation(self):
        clf = SVM()
        self.assertEqual(clf.__class__, SVM)

    def test_set_gamma(self):
        clf = SVM()
        clf.set_gamma(0.5)
        self.assertEqual(clf.gamma, 0.5)

    def test_add_index(self):
        clf = SVM()
        clf.set_gamma(0.5)
        self.assertTrue(np.array_equal(clf.__add_idx__(np.asarray([[1,2],[1,2]])), np.asarray([[0,1,2],[1,1,2]])))

    def test_supp_list(self):
        clf = SVM()
        clf.fit(np.asarray([[1, 2, 3], [1, 2, 3]]), [0, 1])
        print(clf.supp.shape[0])
        self.assertEqual(clf.supp.shape[0], 1)

    def test_pick_random(self):
        a = np.asarray([[0, 2, 3], [1, 2, 3]])
        print("AAA")
        print(SVM.__random_vector__(a))


if __name__ == '__main__':
    unittest.main()
