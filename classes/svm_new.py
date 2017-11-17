import sklearn as sk
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import matplotlib.pyplot as plt


class SVM:
    def __init__(self):
        self._gamma = 'auto'
        self.alpha = np.array([])
        self.kernel = 'rbf'
        self.random_state = None
        self.niter = 1000
        self.epsilon = 0.001
        self.support_vectors_ = None
        self._supp_idx = np.array([], dtype='uint8')
        self.dropout = 0
        self.C = 1
        self.batch_size = 1
        self.X = None
        self.y = None

        #Metrics
        self.gamma_progress = []
        self.alpha_progress = []
        self.support_progress = []

    def fit(self, X, y):
        # Preset the system
        self.assertions(X, y)
        self.set_gamma()
        self.set_random_state()

        # Add an initial Support Vector
        idx, _ = self.get_random_vectors(self.batch_size)
        self.add_sv(idx)

        # Select a random element from the dataset equal to the batch size
        for i in range(self.niter):
            # Get the random batch of vectors
            xt_idx, xt = self.get_random_vectors(self.batch_size)
            kernel = rbf_kernel(xt, Y=self.X[self._supp_idx], gamma=self._gamma)
            decision = np.dot(kernel, self.alpha)
            self.log_state()

            if self.y[xt_idx] * decision <= 1:
                self.optimize(xt_idx, xt, kernel)
                self.add_sv(xt_idx)

            else:
                self.regularize(xt_idx, xt)

    def log_state(self, alpha=None):
        if alpha is not None:
            self.alpha_progress.append(np.mean(np.abs(alpha)))
        else:
            self.gamma_progress.append(self._gamma)
            self.support_progress.append(len(self._supp_idx))

    def optimize(self, xt_idx, xt, kernel):
        # TODO: Check if we must just update the current alpha or all alphas.
        reg_term = self.epsilon * np.array(self.alpha)
        opt_term = self.C * self.y[xt_idx] * kernel
        d_alpha = np.squeeze(reg_term - opt_term)
        self.log_state(alpha=d_alpha)
        self.alpha -= d_alpha

    def regularize(self, xt_idx, xt):
        reg_term = self.epsilon * self.alpha
        self.alpha -= reg_term

    def add_sv(self, idx, base_alpha=True):
        # Check if the vector is already in the
        if idx not in self._supp_idx:
            self._supp_idx = np.append(self._supp_idx, idx)
            if base_alpha:
                self.alpha = np.append(self.alpha, self.y[idx])
            else:
                self.alpha = np.append(self.alpha, np.zeros(idx.shape))
            return True
        return False

    def get_random_vectors(self, n, y=False):
        """
        Returns `n` random vectors from X dataset. If y set to True, it also
        returns the corresponding y's.
        :param n:
        :param y:
        :return:
        """
        xt_idx = np.random.randint(0, self.X.shape[0], self.batch_size)
        xt = self.X[xt_idx]
        if y:
            return xt_idx, xt, self.y[xt_idx]
        return xt_idx, xt

    def set_random_state(self):
        if self.random_state is not None:
            np.random.seed(self.random_state)

    def set_gamma(self):
        if self._gamma == 'auto':
            self._gamma = 1./self.X.shape[0]

    def assertions(self, X, y):
        """
        Asserts all values are correct
        :param X: Samples matrix
        :param y: Labels vector
        :return:
        """
        assert X.shape[0] == y.shape[0]
        assert self.batch_size >= 1
        assert self.niter >= 1
        assert self.epsilon > 0
        assert self.dropout >= 0
        assert self.C > 0
        assert len(np.unique(y)) == 2
        assert 1 in y
        assert -1 in y

        self.X = X if self.X is None else X.append(X)
        self.y = y if self.y is None else y.append(X)


