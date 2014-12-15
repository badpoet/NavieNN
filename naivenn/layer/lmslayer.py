__author__ = 'badpoet'

import numpy as np
import theano
import scipy as sp

from outputlayer import OutputLayer
from naivenn.lib.theanolib import SigmoidAgent

class LeastMeanSquareLayer(OutputLayer):

    def __init__(self, batch_size, out_size):
        self.batch_size = batch_size
        self.out_size = out_size
        self.sa = SigmoidAgent()

    def fp(self, image):
        self.image = image
        return image

    def bp(self, target, learning_rate):
        return self.image - target

    def loss(self, target):
        return ((self.image - target) ** 2).sum() * 0.5

    def accuracy(self, label):
        vid = np.argmax(self.image, axis = 1)
        print "labels  = ", label
        print "results = ", vid
        return (label == vid).sum() * 100.0 / len(label)