import mlx
import mlx.nn as nn
import mlx.core as mx
import  mlx.optimizers as optim

import numpy as np
import pandas as pd
import argparse

import math
import time
import os

class MLP(nn.Module):
    def __init__(self, input_dims, hidden_dims, output_dims, no_layers):
        super().__init__()
        layer_sizes = [input_dims] + [hidden_dims] * no_layers + [output_dims]
        self.layers = [nn.Linear(in_dims, out_dims) for in_dims, out_dims in zip(layer_sizes[:-1], layer_sizes[1:])]
        print(layer_sizes)
        print(self.layers)

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = mx.maximum(layer(x), 0.0)

        return self.layers[-1](x)

def loss_fn(model, X, y):
    return mx.mean(nn.losses.cross_entropy(model(X), y))

def eval_fn(model, X, y):
    return mx.mean(mx.argmax(model(X), axis=1) == y)

def batch_iterate(batch_size, X, y):
    perm = mx.array(np.random.permutation(y.size))
    for i in range(0, y.size, batch_size):
        ids = perm[i: i + batch_size]
        yield X[ids], y[ids]

def predict(model, image):
    image = mx.array(image)
    image = image.reshape(1, -1)
    predictions = model(image)
    predicted_class = mx.argmax(predictions, axis=1)
    return predicted_class.item()

def main():
    seed = 0
    input_dims = 784
    hidden_dims = 32
    output_dims = 10
    no_layers = 2
    batch_size = 256
    no_epochs = 10
    learning_rate = 1e-1

    np.random.seed(seed)

    # Getting data
    PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mnist/')

    train = mx.transpose(mx.array(np.array(pd.read_csv(PATH + 'train.csv'))))
    n, m = train.shape
    train_images = mx.transpose(train[1:n] / 255.) # (600000, 784)
    train_labels = train[0] # get labels

    test = mx.transpose(mx.array(np.array(pd.read_csv(PATH + 'test.csv'))))
    i, h = test.shape
    test_images = mx.transpose(test[1:i] / 255.) # (10000, 784)
    test_labels = test[0]

    # Load model
    model = MLP(input_dims, hidden_dims, output_dims, no_layers)
    mx.eval(model.parameters())

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    optimizer = optim.SGD(learning_rate=learning_rate)

    for e in range(no_epochs):
        tic = time.perf_counter()
        for X, y in batch_iterate(batch_size, train_images, train_labels):
            loss, grads = loss_and_grad_fn(model, X, y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

        accuracy = eval_fn(model, test_images, test_labels)
        toc = time.perf_counter()

        print(f'Epoch {e} | Accuracy: {accuracy.item():.3f} | Time: {(toc - tic):.3f}s')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser('Train a simple MLP on MNIST')
    parser.add_argument('--gpu', action='store_true', help='Use the Metal backend')
    args = parser.parse_args()
    if not args.gpu:
        mx.set_default_device(mx.cpu)
    main()
