from sklearn.model_selection import StratifiedKFold


class KFoldCrossValidation:
    def __init__(self, n_folds):
        self.n_folds = n_folds
        self.monitor = {}
        self.results = []

    def validate(self, Klass, X, y, monitor=None, attr={}):
        if monitor is not None:
            for key in monitor:
                self.monitor[key] = []
        skf = StratifiedKFold(n_splits=self.n_folds)
        res = []
        for train, test in skf.split(X, y):
            X_train, X_test = X[train], X[test]
            y_train, y_test = y[train], y[test]
            clf = Klass()
            for key, at in attr.items():
                setattr(clf, key, at)

            clf.fit(X_train, y_train)
            res.append(clf.score(X_test, y_test))
            if monitor is not None:
                for key in monitor:
                    self.monitor[key].append(getattr(clf, key))
        self.results = res
        return self.results
