__author__ = 'badpoet'

import cPickle
import numpy as np
from naivenn.layer import *
from naivenn.nnet import NeuralNetwork

def load_from_file(fn):
    dic = cPickle.load(open(fn, "rb"))
    return [{
                "in": d,
                "out": r
            } for d, r in zip(dic["data"], dic["labels"])]

training_set = load_from_file("cifar10/data_batch_1") +\
               load_from_file("cifar10/data_batch_2") +\
               load_from_file("cifar10/data_batch_3") +\
               load_from_file("cifar10/data_batch_4")
validation_set = load_from_file("cifar10/data_batch_5")
test_set = load_from_file("cifar10/test_batch")

nn = NeuralNetwork()
batch_size = 20
learning_rate = 0.001
nn.layers.append(ConvLayer((32, 32), (5, 5), batch_size, 3, 32, 0.1, "sigmoid"))
nn.layers.append(PoolLayer((2, 2), batch_size, 32, 28, 28))
nn.layers.append(ConvLayer((14, 14), (3, 3), batch_size, 32, 32, 0.1, "sigmoid"))
nn.layers.append(PoolLayer((2, 2), batch_size, 32, 12, 12))
nn.layers.append(ConvLayer((6, 6), (3, 3), batch_size, 32, 64, 0.1, "sigmoid"))
nn.layers.append(FlatLayer(batch_size, 64, 4, 4, 1024))
nn.layers.append(FCLayer(batch_size, 1024, 10, 0.01, "tanh"))
nn.layers.append(SoftmaxLayer(batch_size, 10))

def mktarget(x):
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
                batch = np.array([each["in"] for each in data])
                batch = batch.reshape((batch_size, 3, 32, 32)) - data_mean
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
                batch = np.array([each["in"] for each in data])
                batch = batch.reshape((batch_size, 3, 32, 32)) - data_mean
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

verbose_n = 1
avg_train_loss = []
avg_train_acc = []
valid_loss = []
valid_acc = []
data_mean = np.zeros(3072)
for each in training_set:
    data_mean += each["in"]
data_mean = data_mean.sum() * 1.0 / (len(training_set) * len(data_mean))
data_mean = np.tile(data_mean, batch_size * 3 * 32 * 32)
data_mean = data_mean.reshape((batch_size, 3, 32, 32))

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
    batch = np.array([each["in"] for each in data])
    batch = batch.reshape((batch_size, 3, 32, 32)) - data_mean
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
f = open("stat" + str(key) + ".dat", "wb")
cPickle.dump({
    "avg_train_loss": avg_train_loss,
    "avg_train_acc": avg_train_acc,
    "valid_loss": valid_loss,
    "valid_acc": valid_acc,
    "test_loss": test_loss,
    "test_acc": test_acc
}, f)
f.close()
f = open("model" + str(key) + ".dat", "wb")
cPickle.dump(nn, f)
f.close()
print "KEY: " + str(key)
