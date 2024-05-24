import numpy as np
import math
import random
import pandas as pd

"""
784 -> 256 -> 32 -> 10

linear relu linear relu linear relu linear softmax

forward, backwards, update gradient based on LR, rinse and repeat

MLP class, optimizer class, gradient update classa, batch_iterate function

input data is csv format

WX + b
(784, 1) -> (10, 10) -> (10, 10) -> (10, 1)
1. (10, 784)(784, 1) + (10, 1)
2. (10, 10)(10, 1) + (10, 1)
3. (10 ,10)(10, 1) + (10, 1)
4. (10, 10)(10, 1) + (10, 1)

softmax + one hot encoding

how do i get the gradients? backprop from the losses

how do i calculate gradients


implement:
    * weight update
    * backpropagation
    * loss function
    * gradient descent


second level:
    * adam optimizer
"""

class Linear():
    def __init__(self, input_dims, output_dims, bias=False):
        self.weight = np.random.rand(output_dims, input_dims)   
        if bias:
            self.bias = np.random.rand(output_dims, 1)

        self.shape = self.weights.shape

    def __call__(self, x):
        print(x.shape)
        print(self.shape)
        return np.matmul(self.weight, x) + self.bias

class MLP():
    def __init__(self, no_layers : int, input_dims : int, hidden_dims : int, output_dims : int):
        layer_sizes = [input_dims] + [hidden_dims] * no_layers + [output_dims]
        self.layers = [Linear(in_dim, out_dim, bias=True) for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.weights = [layer.weight for layer in self.layers]
        self.biases = [layer.bias for layer in self.layers] 

    def __call__(self, x):
        for layer in self.layers[:-1]:
           x = np.maximum(0.0, layer(x))

        return self.layers[-1](x)

class SGD():
    def __init__(self, learning_rate, data):
        self.learning_rate = learning_rate
        self.m, _ = data.shape

    def __call__(self, model, X, y):
        # Forward, Backward, Update gradients
        deriv_w = []
        deriv_b = []
        difference = np.argmax(model(X), axis=1) - y 
        mae = mx.mean(difference)

        deriv_w.append(difference)
        deriv_w.append(1/ self.m * deriv_w[-1].dot(layer.weight.T))
        deriv_b.append(1 / self.m * np.sum(deriv_w[-1]))

        for layer in model.layers[::-2]:
            deriv_w.append(1/ self.m * deriv_w[-1].dot(layer.weight.T))
            deriv_b.append(1 / self.m * np.sum(deriv_w[-1]))


    def back_relu(self, x):
        return x > 0 
 
    def MSE(model, X, y):
        return np.mean(np.square(np.argmax(model(X), axis=1) - y)) 

    def backprop(layer, prev_deriv):
        return 1 / self.m * prev_deriv.dot(layer.T)

    def update_gradients(self, model, grads):
        # remember to reverse the order of gradients when returning them from backprop since it's a stack-type event
        for layer, layer_grads in zip(model.layers, grads):
            assert layer.weights.shape == layer_grads.shape
            layer.weights += layer_grads

def one_hot(x):
    zeros_x = np.zeros(x.size, x.max() + 1)
    zeros_x[np.arange(x.size), x] = 1

    return zeros_x.T

def get_predictions(model, X):
    return np.argmax(model(X), 0)

"""
To do:
    1. implement backpropagation
    2. implement sgd
    3. 
"""
