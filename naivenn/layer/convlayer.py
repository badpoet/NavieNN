__author__ = 'badpoet'

import numpy as np

import scipy as sp
import theano

from naivenn.lib.theanolib import SigmoidAgent

from layer import Layer

def rot180(d4tensor):
    d4tensor_new = d4tensor.copy()
    for i in range(len(d4tensor)):
        for j in range(len(d4tensor[i])):
            d4tensor_new[i][j] = np.rot90(d4tensor[i][j], 2)
    return d4tensor_new

class ConvLayer(Layer):

    def __init__(self, image_size, filter_size, batch_size, m, n, sigma, activate = "relu", mom = 0.9, decay = 0.0005):
        self.image_shape = (batch_size, m, image_size[0], image_size[1])
        self.filter_shape = (n, m, filter_size[0], filter_size[1])
        self.output_shape = (
            batch_size,
            n,
            image_size[0] - filter_size[0] + 1,
            image_size[1] - filter_size[1] + 1
        )
        self.ker = np.random.normal(0, sigma, size = self.filter_shape)
        self.last_delta_ker = np.zeros(self.filter_shape)
        self.b = np.zeros(n)
        self.last_delta_b = np.zeros(n)
        self.activate = activate
        self.mom = mom
        self.decay = decay
        self.m = m  # number of input features
        self.n = n  # number of output features
        self.sa = SigmoidAgent()
        self.ca = ConvAgent(self.image_shape, self.filter_shape, "valid")
        self.dca = ConvAgent(
            (m, batch_size, image_size[0], image_size[1]),
            (n, batch_size, image_size[0] - filter_size[0] + 1, image_size[1] - filter_size[1] + 1),
            "valid"
        )
        self.dca2 = ConvAgent(
            (batch_size, n, image_size[0] - filter_size[0] + 1, image_size[1] - filter_size[1] + 1),
            (m, n, filter_size[0], filter_size[1]),
            "full"
        )

    def conv(self, signal, filters):
        return self.ca.convolution(signal, filters)

    def d_ker_conv(self, image, act):
        return self.dca.convolution(image, act)

    def d_act_conv(self, image, ker):
        return self.dca2.convolution(image, ker)

    def fp(self, batch):
        assert batch.shape == self.image_shape
        self.batch_size = batch.shape[0]
        assert(batch.shape[1] == self.m)  # dimension of input features matches
        self.image = batch  # store the input
        self.mid = self.conv(batch, self.ker)
        for i in range(self.batch_size):
            for j in range(self.n):
                self.mid[i][j] += self.b[j]
        if self.activate == "relu":
            self.output = self.mid * (self.mid > 0)
        elif self.activate == "sigmoid":
            self.output = self.sa.sigmoid(self.mid)
        elif self.activate == "tanh":
            self.output = np.tanh(self.mid)
        else:
            self.output = self.mid
        assert self.output.shape == self.output_shape

        return self.output

    def bp(self, delta, learning_rate):
        assert delta.shape == self.output_shape
        if self.activate == "relu":
            delta *= (self.mid > 0)  # delta's dimension : (batch_size, n, o_h, o_w)
        elif self.activate == "sigmoid":
            delta *= self.output * (1 - self.output)
        elif self.activate == "tanh":
            delta *= 1 - self.output ** 2
        rot_ker = rot180(self.ker.swapaxes(0, 1))
        self.delta = self.d_act_conv(delta, rot_ker)
        img = self.image.swapaxes(0, 1)
        act = rot180(delta.swapaxes(0, 1))
        ker_delta = rot180(self.d_ker_conv(img, act)).swapaxes(0, 1)
        self.ker -= ker_delta * learning_rate - self.mom * self.last_delta_ker
        self.last_delta_ker = ker_delta
        delta_b = np.zeros(self.n)
        for batch_delta in delta:
            for i in range(self.n):
                delta_b[i] -= batch_delta[i].sum() * learning_rate
        self.b += delta_b + self.mom * self.last_delta_b
        self.last_delta_b = delta_b
        assert self.delta.shape == self.image_shape
        return self.delta


class ConvAgent(object):

    def __init__(self, image_shape, filter_shape, conv_type):
        # print "CONVAGENT"
        # print image_shape
        # print filter_shape
        assert filter_shape[1] == image_shape[1]
        self.signal = theano.tensor.matrix().reshape(image_shape)
        self.filters = theano.tensor.matrix().reshape(filter_shape)
        self.conv_func = theano.tensor.nnet.conv2d(
            input = self.signal,
            filters = self.filters,
            filter_shape = filter_shape,
            image_shape = image_shape,
            border_mode = conv_type
        )
        self.convolution = theano.function(
            [self.signal, self.filters],
            self.conv_func
        )

def __test_convolution():
    ca = ConvAgent((1, 1, 3, 3), (1, 1, 5, 5), "valid")
    signal = np.array([[[
        [0.6, 0.7, 0.8, 0.9, 1],
        [0.5, 0.6, 0.7, 0.8, 0.9],
        [0.4, 0.5, 0.6, 0.7, 0.8],
        [0.3, 0.4, 0.5, 0.6, 0.7],
        [0.2, 0.3, 0.4, 0.5, 0.6]
    ]]])
    filter = np.array([[[
        [0, 0, 0],
        [0, 0.5, 0],
        [1, 0, 0]
    ]]])
    cost = np.array([[[
        [0, 0, 0],
        [0, 1, 0],
        [0, 0, 0]
    ]]])
    print ca.convolution(signal, filter)

def __test_convlayer():
    batch = np.array([
        [[[1.0, 1, -1, 1, 1],
          [1, 1, -1, -1, 1],
          [1, -1, -1, 1, 1],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]],
         [[0, 0, 0, 0, 1],
          [0.0, 0, 0, 1, 1],
          [0, 0, 1, 1, 1],
          [0, 1, -1, -1, 1],
          [1, 1, -1, 1, 1]]]
    ])
    for i in range(3):
        for j in range(2):
            for k in range(3):
                for l in range(3):
                    cl = ConvLayer((5, 5), (3, 3), 1, 2, 3, 0.5, "relu")
                    res0 = cl.fp(batch)
                    dd = 0.0001
                    cl.ker[i][j][k][l] += dd
                    res = cl.fp(batch)
                    v = ((res**2).sum() - (res0**2).sum()) / (2 * dd)
                    cl.ker[i][j][k][l] -= dd
                    prev = cl.ker[i][j][k][l]
                    cl.bp(res, 1)
                    print abs(cl.ker[i][j][k][l] - prev + v)
    for i in range(3):
        cl = ConvLayer((5, 5), (3, 3), 1, 2, 3, 0.5, "relu")
        res0 = cl.fp(batch)
        dd = 0.0001
        cl.b[i] += dd
        res = cl.fp(batch)
        v = ((res**2).sum() - (res0**2).sum()) / (2 * dd)
        cl.b[i] -= dd
        prev = cl.b[i]
        cl.bp(res, 1)
        print abs(cl.b[i] - prev + v)

if __name__ == "__main__":
    __test_convlayer()
