__author__ = 'badpoet'

from layer import *

class NeuralNetwork(object):

    def __init__(self):
        self.layers = []

    def fp(self, batch):
        for layer in self.layers:
            batch = layer.fp(batch)
        return batch

    def loss(self, target):
        return self.layers[-1].loss(target)

    def accuracy(self, label):
        return self.layers[-1].accuracy(label)

    def bp(self, target, learning_rate):
        for layer in reversed(self.layers):
            target = layer.bp(target, learning_rate)
        return target
