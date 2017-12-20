from sklearn.datasets import make_blobs, make_moons

import matplotlib.pyplot as plt
import numpy as np
from classes.svm_old import SVM as SVMO
import datasets

from classes.svm import SVM
from classes.svmgo import SVMGo
from util.draw import draw_kernel_svm
X = np.array([
    [1,1],
    [2,2],
    [3,3],
    [-1,-1],
    [-2,-2],
    [-3,-3]
])

y = np.array([
    1,
    1,
    1,
    -1,
    -1,
    -1])

X, y = make_moons(n_samples=1000, noise=0.2)
X, y = make_blobs(centers=2, n_samples=1000, n_features=2)
X, y = get_adult()


y[y == 0] = -1
if False:
    clf = SVMO()
    clf.niter = 10000
    clf.C = 1
    clf.dropout = 0
    clf._gamma = 0.001
    clf.gamma_opt = False
    clf.regularize = True
    clf.epsilon = 0.001
    clf.fit(X, y)
    draw_kernel_svm(X, y, clf.support_vectors_, clf.alpha, clf._gamma)

else:
    clf = SVMGo()
    clf.C = 0.1
    clf.dropout = 0.9
    clf._gamma = 0.001
    #clf.regularize = True
    clf.epsilon = 0.001
    clf.fit(X, y, epochs=20)
    #draw_kernel_svm(X, y, X[clf._supp_idx], clf._alpha, clf._gamma)

print(clf.score(X,y))
plt.figure()
plt.plot(clf.alpha_progress)

plt.figure()
plt.plot(clf.gamma_progress)

plt.figure()
plt.hist(clf._alpha)
plt.show()

