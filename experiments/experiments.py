import time
import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import sklearn as sk
from sklearn.svm import SVC
import numpy as np
from .. import SVMGo
from ..datasets.getdata import datasets, small_datasets


class Experiment:
    def __init__(self, name, directory='./results/'):
        self._name = name
        self._dir_name = directory + '/' + name + str(int(time.time()))
        self._figures = []
        self._models = []
        self._df = pd.DataFrame()
        if not os.path.exists(self._dir_name):
            os.makedirs(self._dir_name)

    def save_csv(self):
        self._df.to_csv(self._dir_name+"/results.csv")

    def save_model(self, model, name):
        model_path = self._dir_name + "/model/"
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        with open(model_path+name, 'wb') as f:
            pickle.dump(model, f)

    def save_figure(self, name):
        img_path = self._dir_name + "/img/"
        img_file = img_path+name+".png"
        if not os.path.exists(img_path):
            os.makedirs(img_path)
        self._figures.append(img_file)
        plt.savefig(img_file, format="png")
        return img_path

    def kfold_split(self, X, y, k=5):
        [sk.model_selection.train_test_split(X, y, test_size=1. / k) for _ in range(k)]
        return [sk.model_selection.train_test_split(X, y, test_size=1./k) for _ in range(k)]


class ExperimentSample(Experiment):
    def start(self):
        ds = small_datasets()

        self._df = pd.DataFrame(columns=['classifier', 'dataset', 'dropout', 'gamma', 'accuracy'])

        for i in range(1, 10, 2):
            for d in ds[1:]:
                print('Dropout: %0.1f, Dataset: %s' % (i, d))
                X, y = d()
                y[y == 0] = -1
                do = i/10.
                clf = SVMGo()
                clf.dropout = do
                clf._gamma = 0.001
                # clf.regularize = True
                clf.epsilon = 0.001
                clf.fit(X, y, epochs=10, verbose=False)
                print(clf)

                plt.figure()
                plt.plot(clf.accuracy_progress)
                self.save_figure(d.__name__ + str(do)+"accuracy")
                plt.figure()
                plt.plot(clf.gamma_progress)
                self.save_figure(d.__name__ + str(do) + "gamma")
                plt.figure()
                plt.plot(clf.support_progress)
                self.save_figure(d.__name__ + str(do) + "support")
                self.save_model(clf, "svmgo"+d.__name__ +str(do))

                self._df = self._df.append({'classifier': 'svmgo',
                                            'dataset': d.__name__.split('_')[1],
                                            'dropout': clf.dropout,
                                            'gamma': clf._gamma,
                                            'accuracy': clf.score(X, y)},
                                           ignore_index=True)
                self.save_csv()


class ExpAccuracyBatchOnline(Experiment):
    def start(self):
        kfold = 5
        nsplits = 5
        ds = datasets()
        self._df = pd.DataFrame(columns=['classifier', 'dataset', 'dropout', 'gamma', 'C', 'accuracy'])

        for d in ds:
            X, y = d()
            y[y == 0] = -1
            data = self.kfold_split(X, y, k=kfold)
            """
            svm-GO
            """
            for i in range(nsplits):
                score = []
                gamma = []
                do = (1+i)*(1 / float(nsplits))
                for X_train, X_test, y_train, y_test in data:
                    print('Dropout: %0.1f, Dataset: %s' % (i, d))
                    clf = SVMGo()
                    clf.dropout = do
                    clf._gamma = 0.001
                    # clf.regularize = True
                    clf.epsilon = 0.001
                    clf.fit(X_train, y_train, epochs=20, verbose=False)
                    print(clf)
                    score.append(clf.score(X_test, y_test))
                    gamma.append(clf._gamma)

                self._df = self._df.append({'classifier': 'svmgo',
                                            'dataset': d.__name__.split('_')[1],
                                            'dropout': do,
                                            'gamma': np.mean(gamma),
                                            'C': 0,
                                            'accuracy': np.mean(np.array(score))},
                                           ignore_index=True)
                self.save_csv()

            """
            SVC
            """
            print('---SVC')
            param_grid = [{'C': [1, 10, 100, 1000], 'gamma': [0.1, 1, 0.01, 0.001, 0.0001], 'kernel': ['rbf']}]
            svc = SVC()
            clf = sk.model_selection.GridSearchCV(svc, param_grid, n_jobs=4, cv=kfold)
            clf.fit(X, y)
            self._df = self._df.append({'classifier': 'SVC',
                                        'dataset': d.__name__.split('_')[1],
                                        'dropout': "",
                                        'gamma': clf.best_params_['gamma'],
                                        'C': clf.best_params_['C'],
                                        'accuracy': clf.best_score_},
                                       ignore_index=True)
            self.save_csv()
