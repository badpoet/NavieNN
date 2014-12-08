__author__ = 'badpoet'

import numpy as np

from functions import sigmoid, d_sigmoid, tanh, d_tanh

class Layer(object):

    def __init__(self, in_size, out_size, h, d_h, learning_rate = 0.1):
        self.in_size = in_size
        self.out_size = out_size
        self.h = h
        self.d_h = d_h
        self.next_layer = None
        self.prev_layer = None
        self.learning_rate = learning_rate

    def connect(self, out_size, h, d_h):
        it = self
        while it.next_layer is not None: it = it.next_layer
        it.next_layer = Layer(it.out_size, out_size, h, d_h, self.learning_rate)
        it.next_layer.prev_layer = it
        return self

    def set_weights(self, weight_range = (-0.5, 0.5), bias_range = (-0.01, 0.01)):
        self.w = np.random.uniform(weight_range[0],
                                   weight_range[1],
                                   (self.out_size, self.in_size))
        self.b = np.random.uniform(bias_range[0], bias_range[1], self.out_size)
        if self.next_layer is not None:
            self.next_layer.set_weights(weight_range, bias_range)

    def compute(self, signals):
        self.s = signals
        self.a = np.dot(self.w, self.s) + self.b
        self.z = np.array(map(self.h, self.a))
        if self.next_layer is not None:
            return self.next_layer.compute(self.z)
        else:
            return self.z

    def bp(self, error):
        self.err = np.array(map(self.d_h, self.a)) * error
        back_err = np.dot(self.err, self.w)
        self.w -= self.err[np.newaxis].T.dot(self.s[np.newaxis]) * self.learning_rate
        self.b -= self.err * self.learning_rate
        if self.prev_layer is not None:
            return self.prev_layer.bp(back_err)
        else:
            return back_err

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
    mlp = MultiLayerPerceptron(Layer(8, 3, tanh, d_tanh, 0.1).connect(
        8, sigmoid, d_sigmoid
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
