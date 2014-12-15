__author__ = 'badpoet'

from outputlayer import OutputLayer
import numpy as np
import theano

from naivenn.lib.theanolib import LogAgent, SoftmaxAgent

class SoftmaxLayer(OutputLayer):

    def __init__(self, batch_size, out_size):
        self.batch_size = batch_size
        self.out_size = out_size
        self.la = LogAgent()
        self.sa = SoftmaxAgent()

    def fp(self, image):
        self.output = self.sa.softmax(image)
        return self.output

    def bp(self, target, learning_rate = 0):
        assert self.output.shape == target.shape
        return self.output - target

    def loss(self, target):
        return (-target * self.la.log(self.output)).sum()

    def accuracy(self, label):
        vid = np.argmax(self.output, axis = 1)
        print "labels  = ", label
        print "results = ", vid
        return (label == vid).sum() * 100.0 / len(label)



