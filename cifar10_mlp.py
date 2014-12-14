from __future__ import division
from naivenn.mlp import MultiLayerPerceptron
from naivenn.layer import FCLayer
from naivenn.functions import sigmoid, d_sigmoid, tanh, d_tanh

import cPickle
import numpy as np
import random

__author__ = 'badpoet'

def load_from_file(fn):
    dic = cPickle.load(open(fn, "rb"))
    return [{
                "in": d / 255.0,
                "out": r
            } for d, r in zip(dic["data"], dic["labels"])]

batches = []
for i in range(1, 6):
    fn = "data/data_batch_" + str(i)
    batches.extend(load_from_file(fn))

tests = load_from_file("data/test_batch")

mlp = MultiLayerPerceptron(
    FCLayer(3072, 100, sigmoid, d_sigmoid).connect(
    FCLayer(100, 10, sigmoid, d_sigmoid))
)
mlp.set_weights()

print "CALCULATING"

ITER = 100000
for i in range(ITER):
    if i % 500 == 0: print i * 100 / ITER, "%"
    k = random.randint(0, len(batches) - 1)
    input = batches[k]["in"]
    answer = batches[k]["out"]
    target = np.zeros(10)
    target[answer] = 1
    result = mlp.compute(input)
    mlp.bp(result - target)
print "100.0%"

print "EVALUATING"

n = len(tests)
correct_num = 0
for t in tests:
    max_c = 0
    max_c_score = -100000
    result = mlp.compute(t["in"])
    for j in range(10):
        if result[j] > max_c_score:
            max_c_score = result[j]
            max_c = j
    if max_c == t["out"]:
        correct_num += 1

print correct_num / n