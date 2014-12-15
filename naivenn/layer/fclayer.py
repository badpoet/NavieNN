__author__ = 'badpoet'

import numpy as np
from layer import Layer
from naivenn.lib.theanolib import SigmoidAgent
class FCLayer(Layer):

    def __init__(self, batch_size, in_size, out_size, sigma, activate):
        self.in_size = in_size
        self.out_size = out_size
        self.batch_size = batch_size
        self.in_shape = (batch_size, in_size)
        self.w_shape = (out_size, in_size)
        self.out_shape = (batch_size, out_size)
        self.activate = activate
        self.sa = SigmoidAgent()
        self.w = np.random.normal(0, sigma, self.w_shape)
        self.b = np.zeros(self.out_size)

    def fp(self, signals):
        assert signals.shape == self.in_shape
        self.s = signals
        self.a = np.zeros(self.out_shape)
        for i in range(self.batch_size):
            # print self.a[i].shape
            # print self.w.shape
            # print self.s[i].shape
            # print self.b.shape
            self.a[i] = np.dot(self.w, self.s[i]) + self.b
        if self.activate == "tanh":
            self.z = np.tanh(self.a)
        elif self.activate == "sigmoid":
            self.z = self.sa.sigmoid2(self.a)
        else:
            self.z = self.a
        assert self.z.shape == self.out_shape
        return self.z

    def bp(self, error, learning_rate):
        assert error.shape == self.out_shape
        if self.activate == "tanh":
            self.err = -(np.tanh(self.a) ** 2) + 1
        elif self.activate == "sigmoid":
            self.err = self.z * (1 - self.z)
        else:
            self.err = np.ones(error.shape)
        self.err *= error
        back_err = np.zeros(self.in_shape)
        for i in range(self.batch_size):
            back_err[i] = np.dot(self.err[i], self.w)
        for i in range(self.batch_size):
            self.w -= self.err[i][np.newaxis].T.dot(self.s[i][np.newaxis]) * learning_rate
            self.b -= self.err[i] * learning_rate
        assert back_err.shape == self.in_shape
        return back_err

