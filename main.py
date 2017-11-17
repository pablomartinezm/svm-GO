from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np
from classes.svm_new import SVM
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

X, y = make_blobs(centers=2, n_samples=1000, n_features=2)
y[y==0]=-1
clf = SVM()
clf.niter = 10000
clf.gamma = 0
clf.gamma_opt = True
clf.fit(X,y)
draw_kernel_svm(X, y, X[clf._supp_idx], clf.alpha, clf.gamma)
plt.figure()
plt.plot(clf.alpha_progress)

plt.show()
plt.figure()
plt.hist(clf.alpha)
plt.show()

