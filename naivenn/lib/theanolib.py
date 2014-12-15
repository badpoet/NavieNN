__author__ = 'badpoet'

import theano
from theano.tensor.signal.downsample import max_pool_2d

class SigmoidAgent(object):

    def __init__(self):
        _x = theano.tensor.tensor4()
        _x2 = theano.tensor.matrix()
        _func = theano.tensor.nnet.sigmoid(_x)
        _func2 = theano.tensor.nnet.sigmoid(_x2)
        self.sigmoid = theano.function([_x], _func)
        self.sigmoid2 = theano.function([_x2], _func2)


class PoolAgent(object):

    def __init__(self, pool_size, image_shape):
        self.pool_size = pool_size
        self.image = theano.tensor.matrix().reshape(image_shape)
        self.pool_func = max_pool_2d(self.image, pool_size)
        self.pool = theano.function(
            [self.image],
            self.pool_func
        )


class SoftmaxAgent(object):

    def __init__(self):
        _x = theano.tensor.matrix()
        _func = theano.tensor.nnet.softmax(_x)
        self.softmax = theano.function([_x], _func)

class LogAgent(object):

    def __init__(self):
        _x = theano.tensor.matrix()
        _func = theano.tensor.log(_x)
        self.log = theano.function([_x], _func)

