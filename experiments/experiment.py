import time
import os
import pickle
import matplotlib.pyplot as plt
from ..classes.svmgo import SVMGo
from ..datasets.getdata import datasets


class Experiment:
    def __init__(self, name, directory='./results/'):
        self._name = name
        self._dir_name = directory + '/' + name + str(int(time.time()))
        self._figures = []
        self._models = []
        if not os.path.exists(self._dir_name):
            os.makedirs(self._dir_name)

    def save_csv(self):
        pass

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


class ExperimentSample(Experiment):
    def start(self):
        ds = datasets()
        X, y = ds[2]()
        y[y == 0] = -1

        for i in range(1,10):
            do = i/10.
            clf = SVMGo()
            clf.dropout = do
            clf._gamma = 0.001
            # clf.regularize = True
            clf.epsilon = 0.001
            clf.fit(X, y, epochs=10)
            print(clf)

            plt.figure()
            plt.plot(clf.accuracy_progress)
            self.save_figure(str(do)+"accuracy")
            plt.figure()
            plt.plot(clf.gamma_progress)
            self.save_figure(str(do) + "gamma")
            plt.figure()
            plt.plot(clf.support_progress)
            self.save_figure(str(do) + "support")
            self.save_model(clf, "svmgo"+str(do))
