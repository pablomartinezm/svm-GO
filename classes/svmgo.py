from . import SVM
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


class SVMGo(SVM):

    def __init__(self):
        SVM.__init__(self)
        self.dropout = 0.1
        self._random_sample = np.array([], dtype='uint8')

    @property
    def sv(self):
        """
        Return current Support Vectors
        :return:
        """
        if self.dropout is None:
            return self.X[self._supp_idx]
        return self.X[self._supp_idx[self._random_sample]]

    @property
    def sv_y(self):
        """
        Get the support vector's labels
        :return:
        """
        if self.dropout is None:
            return self.y[self._supp_idx]
        return self.y[self._supp_idx[self._random_sample]]

    @property
    def alpha(self):
        if self.dropout is None:
            return self._alpha
        return self._alpha[self._random_sample]

    @alpha.setter
    def alpha(self, alpha):
        if self.dropout is None:
            self._alpha = alpha
        else:
            self._alpha[self._random_sample] = alpha

    def optimize(self, xt_idx, xt, kernel):
        super(SVMGo, self).optimize(xt_idx, xt, kernel)

        # Optimize gamma
        norm = np.linalg.norm(xt-self.sv, axis=1)
        d_gamma = float(np.dot(kernel * norm, self.y[xt_idx] * self.alpha))
        ngamma = self._gamma - self.epsilon * d_gamma
        self._gamma = max(ngamma, 0)

    def regularize(self, xt_idx, xt):
        pass

    def assertions(self, X, y):
        super().assertions(X, y)
        assert self.dropout is None or self.dropout >= 0

    def random_sample(self):
        if self.dropout is not None:
            rnd_idx = np.random.rand(self._supp_idx.shape[0]) > self.dropout
            self._random_sample = rnd_idx if np.sum(rnd_idx) != 0 else np.logical_not(rnd_idx)
