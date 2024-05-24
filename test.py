import mlx
import mlx.nn as nn
import mlx.core as mx
import pandas as pd
import numpy as np
import os

from model import Linear

linear = Linear(784, 10)
print(linear.weights)

linear.weights = linear.weights + 1
print(linear.weights)
