__author__ = 'badpoet'

import numpy as np
from theano.tensor.signal.downsample import max_pool_2d
import theano
from layer import Layer

from naivenn.lib.theanolib import PoolAgent

class PoolLayer(Layer):

    def __init__(self, pool_size, batch_size, n_feature, h, w):
        image_shape = (batch_size, n_feature, h, w)
        self.pool_size = pool_size
        self.image_shape = image_shape
        self.output_shape = (
            batch_size,
            n_feature,
            h / pool_size[0] + (h % pool_size[0] != 0),
            w / pool_size[1] + (w % pool_size[1] != 0)
        )
        self.pa = PoolAgent(pool_size, image_shape)

    def fp(self, image):
        assert image.shape == self.image_shape
        self.image = image
        self.output = self.pa.pool(image)
        assert self.output.shape == self.output_shape
        return self.output

    def bp(self, delta, learning_rate = 0):
        assert delta.shape == self.output_shape
        self.delta = delta
        self.act = np.zeros(self.image_shape)
        for i in range(len(delta)):
            for j in range(len(delta[i])):
                for k in range(len(delta[i][j])):
                    for l in range(len(delta[i][j][k])):
                        for t_i in range(k * self.pool_size[0], min((k + 1) * self.pool_size[0], self.image_shape[2])):
                            for t_j in range(l * self.pool_size[1], min((l + 1) * self.pool_size[1], self.image_shape[3])):
                                if self.image[i][j][t_i][t_j] == self.output[i][j][k][l]:
                                    self.act[i][j][t_i][t_j] = delta[i][j][k][l]
        assert self.act.shape == self.image_shape
        return self.act


def __test_pool_layer():
    pl = PoolLayer((2, 2), 1, 2, 3, 3)
    batches = np.array([[
        [[1, 0, -0.3],
         [0.2, 0.5, 0.7],
         [0.8, 0.6, -0.6]],
        [[0.5, -0.4, 0.1],
         [0.1, 0.9, 0.0],
         [-0.3, -0.4, 0.5]]
    ]])
    print pl.fp(batches)
    print pl.bp(np.ones((1, 2, 2, 2)))

if __name__ == "__main__":
    __test_pool_layer()