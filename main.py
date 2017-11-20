from sklearn.datasets import make_blobs, make_moons
import matplotlib.pyplot as plt
import numpy as np
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

y = np.array([1,1,1,-1,-1,-1])

X, y = make_moons(n_samples=10, noise=0.2)
# X, y = make_blobs(centers=2, n_samples=1000, n_features=2)
y[y==0]=-1
clf = SVMGo()
clf.niter = 100
clf._gamma = 0.01
clf.epsilon = 0.001
clf.fit(X, y)
draw_kernel_svm(X, y, X[clf._supp_idx], clf._alpha, clf._gamma)
plt.figure()
plt.plot(clf.alpha_progress)

plt.show()
plt.figure()
plt.hist(clf.alpha)
plt.show()

