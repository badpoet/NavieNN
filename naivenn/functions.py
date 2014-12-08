__author__ = 'badpoet'

import math
identity = lambda x: x
d_identity = lambda x: 1
sigmoid = lambda x: 1 / (1 + math.exp(-x))
d_sigmoid = lambda x: sigmoid(x) * (1 - sigmoid(x))
tanh = lambda x: math.tanh(x)
d_tanh = lambda x: 1 - tanh(x) ** 2
