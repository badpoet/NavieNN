__author__ = 'badpoet'

import numpy as np
from naivenn.nnet import NeuralNetwork
from naivenn.layer import *
import cPickle

training_set = cPickle.load(open("digit/digit_train.dat", "rb"))
validation_set = cPickle.load(open("digit/digit_test.dat", "rb"))
print "CALCULATING"

test_set = validation_set

nn = NeuralNetwork()
batch_size = 20
learning_rate = 0.01
nn.layers.append(FCLayer(batch_size, 9216, 800, 0.01, "sigmoid"))
nn.layers.append(FCLayer(batch_size, 800, 10, 0.01, "sigmoid"))
nn.layers.append(LeastMeanSquareLayer(batch_size, 10))

def mkinput(x):
    return np.array([int(y) for y in x])

def mktarget(x):
    # v = -np.ones(10)
    v = np.zeros(10)
    v[x] = 1
    return v

def train(learning_rate):
    try:
        epoch = 0
        while (True):
            epoch += 1
            last_acc = 0
            loss = 0
            accuracy = 0
            for batch_id in range(len(training_set) / batch_size):
                data = training_set[batch_size * batch_id : batch_size * (batch_id + 1)]
                label = np.array([each["out"] for each in data])
                batch = np.array([mkinput(each["in"]) for each in data])
                target = np.array([mktarget(each["out"]) for each in data])
                nn.fp(batch)
                loss += nn.loss(target)
                accuracy += nn.accuracy(label)
                nn.bp(target, learning_rate)
                if batch_id % verbose_n == verbose_n - 1:
                    print "EPOCH %d BATCHID %d TRAINING" % (epoch, (batch_id + 1))
                    print "........AVG LOSS %f" % (loss / (batch_id + 1), )
                    print "........AVG ACC  %f" % (accuracy / (batch_id + 1), )
                    avg_train_loss.append(loss / (batch_id + 1))
                    avg_train_acc.append(accuracy / (batch_id + 1))
            loss = 0
            accuracy = 0
            for batch_id in range(len(validation_set) / batch_size):
                data = validation_set[batch_size * batch_id : batch_size * (1 + batch_id)]
                label = np.array([each["out"] for each in data])
                batch = np.array([mkinput(each["in"]) for each in data])
                target = np.array([mktarget(each["out"]) for each in data])
                nn.fp(batch)
                loss += nn.loss(target)
                accuracy += nn.accuracy(label)
                if batch_id % 100 == 0:
                    print "VALIDATING... %f" % (100.0 * batch_id / (len(validation_set) / batch_size), )
            print "EPOCH %d VALIDATION" % (epoch, )
            print "........AVG LOSS %f" % (loss / (len(validation_set) / batch_size), )
            print "........AVG ACC  %f" % (accuracy / (len(validation_set) / batch_size), )
            valid_loss.append(loss / (len(validation_set) / batch_size))
            valid_acc.append(accuracy / (len(validation_set) / batch_size))
            if epoch > 1 and (accuracy - last_acc) / accuracy < 0.05:
                learning_rate *= 0.1
                last_acc = accuracy
    except KeyboardInterrupt, e:
        print "STOP"

verbose_n = 5
avg_train_loss = []
avg_train_acc = []
valid_loss = []
valid_acc = []

print "training set: ", len(training_set)
print "validation set: ", len(validation_set)
print "test set: ", len(test_set)
print ">>>>>>>> ENGINE START >>>>>>>"
train(learning_rate)

loss = 0
accuracy = 0
for batch_id in range(len(test_set) / batch_size):
    data = test_set[batch_size * batch_id : batch_size * (1 + batch_id)]
    label = np.array([each["out"] for each in data])
    batch = np.array([mkinput(each["in"]) for each in data])
    target = np.array([mktarget(each["out"]) for each in data])
    nn.fp(batch)
    loss += nn.loss(target)
    accuracy += nn.accuracy(label)
    if batch_id % 100 == 0:
        print "TESTING... %f" % (100.0 * batch_id / (len(test_set) / batch_size), )
print "TEST"
print "....AVG LOSS %f" % (loss / (len(test_set) / batch_size), )
print "....AVG ACC  %f" % (accuracy / (len(validation_set) / batch_size), )
test_loss = loss / (len(validation_set) / batch_size)
test_acc = accuracy / (len(validation_set) / batch_size)

import random
key = random.randint(10000, 99999)
f = open("digit_stat" + str(key) + ".dat", "wb")
cPickle.dump({
    "avg_train_loss": avg_train_loss,
    "avg_train_acc": avg_train_acc,
    "valid_loss": valid_loss,
    "valid_acc": valid_acc,
    "test_loss": test_loss,
    "test_acc": test_acc
}, f)
f.close()
f = open("digit_model" + str(key) + ".dat", "wb")
cPickle.dump(nn, f)
f.close()
print "KEY: " + str(key)

