import pandas as pd
import numpy as np
from sklearn.datasets import load_svmlight_file
from sklearn.preprocessing import MinMaxScaler

DIR_PREFIX = "datasets/data/"


def set_dir(dir):
    global DIR_PREFIX
    DIR_PREFIX = dir

def get_bank():
    xx = pd.read_csv(DIR_PREFIX+'/bank-full.csv',
                     engine='python',
                     sep=',',
                     )
    xx['y'] = (xx.y == 'yes') * 1
    xx = pd.get_dummies(xx)
    return _scale(xx.loc[:, xx.columns != 'y'].as_matrix()), xx['y'].as_matrix()


def get_blood():
    xx = pd.read_csv(DIR_PREFIX+"blood.csv")
    xx = pd.get_dummies(xx)
    return _scale(xx.loc[:, xx.columns != 'y'].as_matrix()), xx['y'].as_matrix()


def get_arrhythmia():
    ds = pd.read_csv(DIR_PREFIX + 'arrhythmia.csv', header=None)
    ds = ds.replace('?', np.nan)
    ds[ds.columns[-1]][ds[ds.columns[-1]] != 1] = -1
    ds = ds.dropna(axis=1, how='any')
    return _scale(ds.loc[:, ds.columns != 279].as_matrix()), ds[279].as_matrix()


def get_magic():
    xx = pd.read_csv("./svmgo/datasets/data/magic.csv")
    xx['g'] = xx['g'].replace('g', -1)
    xx['g'] = xx['g'].replace('h', 1)
    return _scale(xx.loc[:, xx.columns != 'g'].as_matrix()), xx['g'].as_matrix()


def get_adult():
    data = load_svmlight_file(DIR_PREFIX+"adult.libsvm")
    return _scale(data[0].todense()), data[1]


def get_malicious():
    data = load_svmlight_file(DIR_PREFIX+"adult.libsvm")
    return _scale(data[0].todense()), data[1]


def get_ringnorm():
    xx = pd.read_csv(DIR_PREFIX+'/ringnorm.csv',
                     engine='python',
                     sep=' ',
                     )

    xx = np.asarray(xx)
    xx = _scale(xx)
    return xx[:,:-1], xx[:,-1]

def get_arcene():
    xx = pd.read_csv('svmgo/datasets/data/arcene.csv', sep=' ', header=None)
    yy = np.genfromtxt('svmgo/datasets/data/arcene_y.csv').astype('int8')
    return _scale(xx.as_matrix().astype('float64')), yy

def _scale(df):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    return scaler.fit_transform(df)
