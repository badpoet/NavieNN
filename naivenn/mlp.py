__author__ = 'badpoet'

import numpy as np

from functions import sigmoid, d_sigmoid, tanh, d_tanh
from layer import FCLayer


class MultiLayerPerceptron(object):

    def __init__(self, layer):
        self.input_layer = layer
        while layer.next_layer is not None:
            layer = layer.next_layer
        self.output_layer = layer

    def set_weights(self, *args, **kwargs):
        self.input_layer.set_weights(*args, **kwargs)

    def compute(self, signals):
        return self.input_layer.compute(signals)

    def bp(self, error):
        return self.output_layer.bp(error)

if __name__ == "__main__":
    mlp = MultiLayerPerceptron(
        FCLayer(8, 3, sigmoid, d_sigmoid, 0.1).connect(
        FCLayer(3, 8, sigmoid, d_sigmoid)
    ))
    mlp.set_weights()
    for i in range(100000):
        t = np.zeros(8)
        t[0] = 1
        np.random.shuffle(t)
        result = mlp.compute(t)
        mlp.bp(result - t)

    for i in range(8):
        t = np.zeros(8)
        t[i] = 1
        for c in mlp.compute(t): print c,
        print
