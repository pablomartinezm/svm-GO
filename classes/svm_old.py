import sklearn as sk
import sklearn.metrics as metrics
import numpy as np
import matplotlib.pyplot as plt


class SVM:
    def __init__(self):
        self.ensemble = None
        self.gamma_opt = True
        self._gamma = 'auto'
        self.alpha = None
        self.kernel = 'rbf'
        self.random_state = None
        self.niter = 1000
        self.epsilon = 0.001
        self.gd_method = 'ssgd'
        self.support_vectors_ = None
        self._supp_idx = []
        self.dropout = 0
        self.regularize = False
        self.C = 1



        #Metrics
        self.gamma_progress = []
        self.alpha_progress = []
        self.loss_progress = []
        self.support_progress = []

    def fit(self, X, y):
        if self.random_state is not None:
            np.random.seed(self.random_state)

        if self._gamma == 'auto':
            self._gamma = 1. / X.shape[1]

        if self.gd_method == 'ssgd':
            self.__ssgd__(X, y)

    """
    Add a row in the position 0 with the index of that row. This 'trick' is used to track the elements included in the
    support vectors list.
    """
    @staticmethod
    def __add_idx__(X):
        rng = np.atleast_2d(np.arange(X.shape[0])).transpose()  # Range to add in the X's to track
        return np.hstack((rng, X))

    @staticmethod
    def __random_vector__(X):
        idx = np.random.choice(X.shape[0], replace=False, )
        return idx

    @staticmethod
    def __random_vector__(X):
        idx = np.random.choice(X.shape[0], replace=False, )
        return idx

    def __dropout__(self):
        rnd_idx = np.random.rand(self.support_vectors_.shape[0]) > self.dropout
        return rnd_idx if np.sum(rnd_idx) != 0 else np.logical_not(rnd_idx)

    def __gamma_opt__(self, Xi, yi, rndi):
        norm = np.linalg.norm(Xi - self.support_vectors_[rndi, :], axis=1)
        K = self.__create_kernel__(Xi, self.support_vectors_[rndi, :])
        # derivative = self.C * np.dot(norm * np.exp(-self._gamma * norm), yi*self.alpha[rndi, :].ravel()) + 1
        # derivative = self.C * np.dot(yi*self.alpha[rndi,:]*np.exp(-self._gamma * norm), norm) + 1
        d_gamma = float(self.C * np.dot(K * norm, yi*self.alpha[rndi, :]))
        ngamma = self._gamma - self.epsilon * d_gamma
        self._gamma = max(ngamma, 0)

    def __gamma_reg__(self):
        self._gamma += max(self._gamma * self.epsilon, 0)

    def __ssgd__(self, X, y):
        # Add a first vector as SV
        self._supp_idx = []
        assert np.unique(y).shape == (2,)
        # Convert 0's to -1's
        y[y == 0] = -1

        # Get a random support vector and add it to SVM
        i = SVM.__random_vector__(X)
        self.support_vectors_ = np.array(X[i], ndmin=2)
        self.alpha = np.array(y[i], ndmin=2, dtype='float64')

        for it_number in range(self.niter):
            i = SVM.__random_vector__(X)
            rndi = self.__dropout__()

            K = self.__create_kernel__(X[i], self.support_vectors_[rndi, :])
            prod = np.dot(K, self.alpha[rndi, :])

            self.gamma_progress.append(self._gamma)
            self.support_progress.append(len(self._supp_idx))
            self.loss_progress.append(float(np.abs(y[i] * prod)))
            if y[i] * prod < 1:
                # Support Vector found! Optimize alpha
                regularize = self.alpha[rndi, :] if self.regularize else 0
                d_alpha = regularize - self.C * y[i] * K.T
                self.alpha_progress.append(np.mean(np.abs(d_alpha)))
                # TODO: Update must affect just to those alphas that are kept
                self.alpha[rndi, :] -= self.epsilon * d_alpha

                if self.gamma_opt:
                    self.__gamma_opt__(X[i], y[i], rndi)

                if i not in self._supp_idx:
                    self._supp_idx.append(i)
                    self.support_vectors_ = np.vstack((self.support_vectors_, X[i]))
                    self.alpha = np.vstack((self.alpha, [y[i]]))

            elif self.regularize:
                self.alpha -= self.alpha * self.epsilon
                '''if self.gamma_opt:
                    self.__gamma_reg__()
                '''


    def predict(self, X):
        pass

    def set_gamma(self, gamma):
        self._gamma = gamma

    def score(self, xtest, ytest):
        ytest[ytest == 0] = -1
        Z = np.ravel(
            np.sign(np.dot(self.__create_kernel__(xtest, self.support_vectors_), self.alpha)).T)
        return np.sum(Z == ytest) / xtest.shape[0]

    def score_ensemble(self, xtest, ytest, n_classifiers=5):
        '''
        This function requires y labels to be -1 and 1
        :param xtest:
        :param ytest:
        :param n_classifiers:
        :return:
        '''
        ytest[ytest == 0] = -1

        uq = np.unique(ytest)
        assert uq.shape[0] == 2
        assert 1 in uq
        assert -1 in uq
        del uq

        rndidx = self.__dropout__()
        Z = np.sign(np.dot(self.__create_kernel__(xtest, self.support_vectors_[rndidx, :]), self.alpha[rndidx]))

        for i in range(n_classifiers-1):
            rndidx = self.__dropout__()
            Z = np.hstack((Z, np.sign(np.dot(self.__create_kernel__(xtest, self.support_vectors_[rndidx, :]), self.alpha[rndidx]))))

        Z = np.ravel(np.sign(np.sum(Z, axis=1)))
        return np.sum(Z == ytest) / xtest.shape[0]

    def __create_kernel__(self, m1, m2):
        if self.kernel == 'rbf':
            return metrics.pairwise.rbf_kernel(np.atleast_2d(m1), Y=np.atleast_2d(m2), gamma=self._gamma)

    def __show__(self):
        print("Number of SV:", self.support_vectors_.shape)
        print("Gamma:", self._gamma)
        print("Gamma progress:")
        plt.figure()
        plt.plot(self.gamma_progress)
        plt.show()
        print("Alpha progress:")
        plt.figure()
        plt.plot(self.alpha_progress)
        plt.show()
        print("Loss progress:")
        plt.figure()
        plt.plot(self.loss_progress)
        plt.show()
        print("Alpha histogram")
        plt.figure()
        plt.hist(self.alpha)

