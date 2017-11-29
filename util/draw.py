import numpy as np
import matplotlib.pyplot as plt
import sklearn as sk


def draw_kernel_svm(X, y, supp, alpha, g, many_dim=False):
    # print (w)
    # create a mesh to plot in
    y[y==0]=-1
    h = 0.2

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    print(x_min, x_max)
    print(y_min, y_max)

    Z = np.dot(sk.metrics.pairwise.rbf_kernel(np.c_[xx.ravel(), yy.ravel()], Y=supp, gamma=g), np.squeeze(alpha))

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    print(Z[Z<0].shape)
    plt.contour(xx, yy, Z, [-1, 0, 1], colors=['blue', 'black', 'red'], alpha=0.8)

    # Plot also the training points
    #plt.scatter(supp[:, 0], supp[:, 1], c=alpha, s=120, alpha=0.7, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)

    plt.xlim((x_min, x_max))
    plt.ylim((y_min, y_max))
    print("done")
    print(g)
