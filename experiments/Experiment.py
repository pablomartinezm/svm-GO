import time
import os
import matplotlib.pyplot as plt

class Experiment:
    def __init__(self, name, directory='./results/'):
        self._name = name
        self._dir_name = directory + '/' + name + str(time.time())
        self._figures = []
        self._models = []
        if not os.path.exists(directory):
            os.makedirs(directory)

        if not os.path.exists(directory+self._dir_name):
            os.makedirs(directory+self._dir_name)


    def save(self):
        pass


    def add_model(self, model, name):
        self._models.append((model, name))

    def add_figure(self, fig, name):
        if type(fig) != plt.Figure:
            raise TypeError('Figure must be of type matplotlib.pyplot.Figure.')
        self._figures.append((fig, name))

    def _write_model(self):
        pass

    def _write_images(self):
        img_path = self._dir_name + "/img"
        if not os.path.exists(img_path + self._dir_name):
            os.makedirs(img_path + self._dir_name)

        for fig, name in self._figures:
            if type(fig) != plt.Figure:
                raise TypeError('Type error in figures stored.')



