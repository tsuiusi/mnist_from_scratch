import mlx
import mlx.nn as nn
import mlx.core as mx
import pandas as pd
import numpy as np
import os

PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mnist/')

train = np.array(pd.read_csv(PATH + 'train.csv'))
print(train)
print(type(train))
test = np.array(pd.read_csv(PATH + 'test.csv'))
print(test.shape)
