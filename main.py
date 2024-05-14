import numpy as np
import pandas as pd
import os
import sys
import random
import math

from mnist import MLP

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

# --- Hyperparameters ----------------------------------------------------------------------------------------------------
no_epochs = 200
input_dims = 784
hidden_dims = 32
output_dims = 10
no_layers = 2
learning_rate = 1e-4
seed = 69

# --- Model and etc ---------------------------------------------------------------------------------------------------- 
model = MLP(no_layers, input_dims, hidden_dims, output_dims)

"""
[0.2, 0.2, 0.6]
[0, 0, 1]
"""
