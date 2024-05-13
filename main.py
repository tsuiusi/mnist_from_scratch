import numpy as np
import pandas as pd
import os
import sys
import random
import math

from mnist import MLP, Linear

# --- Getting Data -------------------------------------------------------------------------------------------------------
# Train dataset
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mnist/')

train = pd.read_csv(PATH + 'train.csv')
train = np.array(train)

m, n = train.shape

np.random.shuffle(train)

train = train[100:m].T
train_images = train[1:n] / 255.
train_labels = train[0]

val = train[:1000].T
val_images = val[1:n] / 255.
val_labels = val[0]

# Test dataset
test = pd.read_csv(PATH + 'test.csv')
test = np.array(test)

linear = Linear(784, 10, bias=True)
model = MLP(3, 784, 10, 10)

print(model(train_images).T[0])
