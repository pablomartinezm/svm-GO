import pandas as pd
import numpy as np


def get_bank(data_folder='.'):
    xx = pd.read_csv(data_folder+'/datasets/data/bank-full.csv',
                     engine='python',
                     sep=',',
                     )
    xx['y'] = (xx.y == 'yes') * 1
    xx = pd.get_dummies(xx)
    return xx.loc[:, xx.columns != 'y'].as_matrix(), xx['y'].as_matrix()

