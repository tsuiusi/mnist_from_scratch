import numpy as np
import pandas as pd
import os
import sys
import random
import math
import time

from model import MLP, Optimizer

# --- Getting Data -------------------------------------------------------------------------------------------------------
# Train dataset
PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mnist/')

train = np.transpose(np.array(np.array(pd.read_csv(PATH + 'train.csv'))))
n, m = train.shape
train_images = np.transpose(train[1:n] / 255.) # get image, normalize
train_labels = train[0] # get labels

test = np.transpose(np.array(np.array(pd.read_csv(PATH + 'test.csv'))))
i, h = test.shape
test_images = np.transpose(test[1:i] / 255.)
test_labels = test[0]

# --- Hyperparameters ----------------------------------------------------------------------------------------------------
no_epochs = 200
input_dims = 784
hidden_dims = 32
output_dims = 10
no_layers = 2
batch_size = 256
learning_rate = 1e-4
seed = 69

# --- Model and etc ---------------------------------------------------------------------------------------------------- 
model = MLP(no_layers, input_dims, hidden_dims, output_dims)
optimizer = Optimizer(learning_rate)

def log(epoch, tic, loss):
    toc = time.perf_counter()
    print(f'Epoch: {epoch} | Time: {(toc - tic):.3f} | Loss: {loss:.3f}')
    return toc

def batch_iterate(batch_size, X, y):
    perm = np.array(np.random.permutation(y.size))
    for i in range(0, y.size, batch_size):
        ids = perm[i: i + batch_size]
        yield X[ids], y[ids]

def eval_fn(model, X, y):
	return np.mean(np.argmax(model(X), axis=1) == y)

tic = time.perf_counter()
for i in range(no_epochs):
    for X, y in batch_iterate(batch_size, train_images, train_labels):
        loss = optimizer(model, X.T, y)

    tic = log(i, tic, loss)
