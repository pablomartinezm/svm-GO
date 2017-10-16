import numpy as np
import sklearn as sk
from sklearn import svm as sksvm
import sklearn.metrics as metrics
from sklearn.datasets import make_blobs, make_moons, make_circles, load_iris
import matplotlib.pyplot as plt
from svm import SVM
from draw import draw_kernel_svm



X, y = make_blobs(centers=2, n_samples=1000, n_features=2)

clf = SVM()
clf.niter = 10000
clf.gamma = 0
clf.gamma_opt = True
clf.fit(X,y)
draw_kernel_svm(X,y,clf.support_vectors_, clf.alpha, clf.gamma)
clf.__show__()