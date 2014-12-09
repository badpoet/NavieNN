__author__ = 'badpoet'

import struct
import numpy as np
import random
from naivenn.mlp import MultiLayerPerceptron, Layer
from naivenn.functions import d_sigmoid, sigmoid
import cPickle

batches = cPickle.load(open("digit/digit.dat", "rb"))

print "CALCULATING"

mlp = MultiLayerPerceptron(Layer(
    9216, 200, sigmoid, d_sigmoid
).connect(
    10, sigmoid, d_sigmoid
))
mlp.set_weights()


ITER = 100
for i in range(ITER):
    if i % 5 == 0: print i * 100.0 / ITER, "%"
    k = random.randint(0, len(batches) - 1)
    input_data = np.array([int(c) for c in batches[k]["in"]])
    answer = batches[k]["out"]
    target = np.zeros(10)
    target[answer] = 1
    result = mlp.compute(input_data)
    mlp.bp(result - target)
print "100.0%"

print "EVALUATING"

n = len(batches)
correct_num = 0
for t in batches:
    max_c = 0
    max_c_score = -100000
    input_data = np.array([int(c) for c in t["in"]])
    result = mlp.compute(input_data)
    for j in range(10):
        if result[j] > max_c_score:
            max_c_score = result[j]
            max_c = j
    if max_c == t["out"]:
        correct_num += 1

print correct_num * 100.0 / n
