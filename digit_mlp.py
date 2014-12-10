__author__ = 'badpoet'

import struct
import numpy as np
import random
from naivenn.mlp import MultiLayerPerceptron, Layer
from naivenn.functions import d_sigmoid, sigmoid
import cPickle

ITER = input("Iteration = ")
FOLD = input("C-V fold = ")

batches = cPickle.load(open("digit/digit.dat", "rb"))
each_len = len(batches) / FOLD
folds = [batches[m : m + each_len] for m in range(FOLD)]
print "CALCULATING"

def train_model(batches, tests):
    mlp = MultiLayerPerceptron(Layer(
        9216, 200, sigmoid, d_sigmoid
    ).connect(
        10, sigmoid, d_sigmoid
    ))
    mlp.set_weights()

    for i in range(ITER):
        if i % 12340 == 0: print i * 100.0 / ITER, "%"
        k = random.randint(0, len(batches) - 1)
        input_data = np.array([int(c) for c in batches[k]["in"]])
        answer = batches[k]["out"]
        target = np.zeros(10)
        target[answer] = 1
        result = mlp.compute(input_data)
        mlp.bp(result - target)
    print "100.0%"

    print "EVALUATING"

    n = len(tests)
    correct_num = 0
    for t in tests:
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
    return correct_num * 100.0 / n

ans = 0
for i in range(FOLD):
    print "MODEL: " , i + 1
    ans += train_model(
        sum(folds[ : i], []) + sum(folds[i + 1 : ], []),
        folds[i]
    )
print "OVERALL: (FOLD ", FOLD, ") = ", ans / FOLD
