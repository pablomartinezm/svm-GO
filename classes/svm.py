import sklearn as sk
from sklearn.metrics.pairwise import rbf_kernel
import numpy as np
import matplotlib.pyplot as plt


class SVM:
    def __init__(self):
        # State variables
        self._supp_idx = np.array([], dtype='uint8')
        self.random_state = None

        # Train and test
        self.X = None
        self.y = None

        # Hyperparameters
        self.C = 1
        self.niter = 1000
        self.epsilon = 0.001
        self.batch_size = 1
        self._gamma = 'auto'
        self._alpha = np.array([])

        # Metrics
        self.gamma_progress = []
        self.alpha_progress = []
        self.support_progress = []

    @property
    def sv(self):
        """
        Return current Support Vectors
        :return:
        """
        return self.X[self._supp_idx]

    @property
    def sv_y(self):
        """
        Get the support vector's labels
        :return:
        """
        return self.y[self._supp_idx]

    @property
    def alpha(self):
        return self._alpha

    @alpha.setter
    def alpha(self, alpha):
        self._alpha = alpha

    def random_sample(self):
        pass

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
            # Set random state
            self.random_sample()

            # Get the random batch of vectors
            xt_idx, xt = self.get_random_vectors(self.batch_size)
            kernel = rbf_kernel(xt, Y=self.sv, gamma=self._gamma)
            decision = np.dot(kernel, self.alpha)
            self.log_state()

            if self.y[xt_idx] * decision <= 1:
                self.optimize(xt_idx, xt, kernel)
                self.add_sv(xt_idx)

            else:
                self.regularize(xt_idx, xt)

            print(self.alpha)

    def log_state(self, alpha=None):
        """
        Logs the current state
        :param alpha:
        :return:
        """
        if alpha is not None:
            self.alpha_progress.append(np.sum(np.abs(alpha)))
        else:
            self.gamma_progress.append(self._gamma)
            self.support_progress.append(len(self._supp_idx))

    def optimize(self, xt_idx, xt, kernel):
        """
        Optimizes alpha according to it's derivative
        :param xt_idx:
        :param xt:
        :param kernel:
        :return:
        """
        # TODO: Check if we must just update the current alpha or all alphas.
        reg_term = self.epsilon * self.alpha
        opt_term = self.C * self.y[xt_idx] * kernel
        d_alpha = np.squeeze(reg_term - opt_term)
        self.log_state(alpha=d_alpha)
        self.alpha -= d_alpha

    def regularize(self, xt_idx, xt):
        """
        Norm2 regularization to maximize the bounds
        :param xt_idx:
        :param xt:
        :return:
        """
        reg_term = self.epsilon * self.alpha
        self.log_state(alpha=reg_term)
        self.alpha -= reg_term

    def add_sv(self, idx, base_alpha=True):
        """
        Adds a support vector to the system. Internally checks if that Support Vector is
        already inserted into the list
        :param idx:
        :param base_alpha:
        :return:
        """
        # Check if the vector is already in the
        if idx not in self._supp_idx:
            self._supp_idx = np.append(self._supp_idx, idx)
            if base_alpha:
                self._alpha = np.append(self._alpha, self.y[idx])
            else:
                self._alpha = np.append(self._alpha, np.zeros(idx.shape))
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
        assert self.C > 0
        assert len(np.unique(y)) == 2
        assert 1 in y
        assert -1 in y

        self.X = X if self.X is None else X.append(X)
        self.y = y if self.y is None else y.append(X)


