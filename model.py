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
        self.weight = np.random.rand(output_dims, input_dims) - 0.5 
        if bias:
            self.bias = np.random.rand(output_dims, 1)

        self.shape = self.weight.shape

    def __call__(self, x):
        self.input = x
        return np.matmul(self.weight, x) + self.bias

    def relu_deriv(self, x):
        return (x > 0).astype(float)

class MLP():
    def __init__(self, no_layers : int, input_dims : int, hidden_dims : int, output_dims : int):
        layer_sizes = [input_dims] + [hidden_dims] * no_layers + [output_dims]
        self.layers = [Linear(in_dim, out_dim, bias=True) for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.weights = [layer.weight for layer in self.layers]
        self.biases = [layer.bias for layer in self.layers] 
        self.layer_outputs = []

    def __call__(self, x):
        self.layer_outputs = []
        for layer in self.layers[:-1]:
           x = np.maximum(0.0, layer(x))
           self.layer_outputs.append(x)
        
        return self.normalize(self.layers[-1](x).T)

    def normalize(self, x):
        return x / np.sqrt(np.sum(np.square(x), axis=1, keepdims=True))

class Optimizer():
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def update_gradients(self, model, deriv_w, deriv_b):
        # remember to reverse the order of gradients when returning them from backprop since it's a stack-type event
        for layer, weight_grads, bias_grads in zip(model.layers[::-1], deriv_w, deriv_b):
            assert layer.weight.shape == weight_grads.shape
            layer.weight -= self.learning_rate * weight_grads
            layer.bias -= self.learning_rate * bias_grads.reshape(layer.bias.shape)

    def one_hot(self, y):
        ohy = np.zeros((y.size, y.max() + 1))
        ohy[np.arange(y.size), y] = 1
        return ohy

    def __call__(self, model, X, y):
        # Forward, Backward, Update gradients
        m, n = X.shape
        deriv_w = []
        deriv_b = []

        pred = model(X)
        delta = (pred - self.one_hot(y)).T
        loss = np.mean(delta ** 2)

        for layer, layer_output in zip(model.layers[::-1], model.layer_outputs[1::-1]):
            deriv_w.append(1/m * delta.dot(layer.input.T))
            deriv_b.append(1/m * np.sum(delta, axis=1, keepdims=True))

            if layer != model.layers[0]:
                if hasattr(layer, 'relu_deriv'):
                    delta = layer.weight.T.dot(delta) * layer.relu_deriv(layer_output)
                else:
                    delta = delta.dot(layer.weight)

        self.update_gradients(model, deriv_w, deriv_b)
        return loss

            
