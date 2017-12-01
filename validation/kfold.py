import numpy as np


class KFold:
    def __init__(self, X, y, k, seed=1):
        if seed is None:
            np.random.seed(None)
        self._order = np.arange(X.shape[0])
        np.random.shuffle(self._order)
        self.X = X
        self.y = y
        self.k = k
        self.ksize = int(X.shape[0]/k)
        self.seed = seed

    def __getitem__(self, instance):
        if instance >= self.k:
            raise ValueError('KFold only holds %i folds' % self.k)

        idx = self.ksize*instance
        mask = np.zeros(self.X.shape[0], dtype=bool)
        mask[idx:idx+self.ksize] = True
        mask = mask.T

        x_train = self.X[self._order[mask]]
        y_train = self.y[self._order[mask]]
        x_test = self.X[self._order[~mask]]
        y_test = self.y[self._order[~mask]]

        return x_train, x_test, y_train, y_test
