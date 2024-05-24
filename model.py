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

    def __call__(self, x):
        for layer in self.layers[:-1]:
           x = np.maximum(0.0, layer(x))

        return self.layers[-1](x)

class Step():
    def __init__(self, learning_rate, data):
        self.learning_rate = learning_rate
        self.m, _ = data.shape

    def __call__(self, model, X, y):
        # Forward, Backward, Update gradients
        deriv_w = []
        deriv_b = []

        pred = model(X)
        delta = np.argmax(pred, axis=1) - y 
        loss = np.mean(delta ** 2)

        for layer in model.layers[::-1]:
            deriv_w.append(1/ self.m * delta.T.dot(layer.input))
            deriv_b.append(1 / self.m * np.sum(delta, axis=0)) # not sure about the axis on this one 

            if layer != model.layers[0]:
                if hasattr(layer, 'relu_deriv'):
                    delta = delta.dot(layer.weight.T) * layer.relu_deriv(layer.input)
                else:
                    delta = delta.dot(layer.weight.T)

        update_gradients(model, deriv_w, deriv_b)

    def update_gradients(self, model, deriv_w, deriv_b):
        # remember to reverse the order of gradients when returning them from backprop since it's a stack-type event
        for layer, weight_grads, bias_grads in zip(model.layers, deriv_w, deriv_b):
            assert layer.weight.shape == weight_grads.shape
            layer.weight += self.learning_rate * weight_grads
            layer.bias += self.learning_rate * bias_grads
            
