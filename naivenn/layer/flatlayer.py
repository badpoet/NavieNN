__author__ = 'badpoet'

import numpy as np
from layer import Layer

class FlatLayer(Layer):

    def __init__(self, batch_size, m, h, w, n):
        self.out_shape = (batch_size, n)
        self.batch_size = batch_size
        self.image_shape = (batch_size, m, h, w)

    def fp(self, image):
        return image.reshape(self.out_shape)

    def bp(self, delta, learning_rate):
        return delta.reshape(self.image_shape)

