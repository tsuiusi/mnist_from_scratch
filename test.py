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
print(mx.array(train))
train = mx.transpose(mx.array(train))

train_images = mx.transpose(train[1:] / 255.)
train_labels = train[0]

print(train_images[0])
print(train_images.shape)
