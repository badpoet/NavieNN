# ANN notes

## MLP on digit

### 1 

input -> full connected layer, sigmod, 200 -> full connected layer, sigmod, 10 

learning rate = 0.1

iteration = 100000

83.85%

## CNN on CIFAR-10

### term 6

```
learning_rate : 0.001
batch_size : 10

Conv 1 : {
    image size : (32, 32),
    filter size : (5, 5),
    features in : 3,
    features out: 32,
    init weight : 0.01,
    activation : "relu"
}
Pool 1 : {
    pooling size : (2, 2),
    features : 32,
    image in size : (28, 28),
    type : "max"
}
Conv 2 : {
    image size : (14, 14),
    filter size : (5, 5),
    features in : 32,
    features out : 64,
    init weight : 0.01,
    activation : "relu"
}
Pool 2 : {
    pooling size : (3, 3),
    features : 64,
    image in size : (12, 12),
    type : "max"
}
Flat : {
    image in : (64, 4, 4)
    signal out : 1024
}
FC : {
    signal in : 1024,
    signal out : 10,
    init weight : 0.01,
    activation : "tanh"
}
Softmax : {
}
```

55%

### term 7
```
batch_size = 20
learning_rate = 0.01  # *= 0.3 (E6, E9)
nn.layers.append(ConvLayer((32, 32), (5, 5), batch_size, 3, 32, 0.01, "sigmoid"))
nn.layers.append(PoolLayer((2, 2), batch_size, 32, 28, 28))
nn.layers.append(ConvLayer((14, 14), (3, 3), batch_size, 32, 64, 0.01, "sigmoid"))
nn.layers.append(PoolLayer((3, 3), batch_size, 64, 12, 12))
nn.layers.append(FlatLayer(batch_size, 64, 4, 4, 1024))
nn.layers.append(FCLayer(batch_size, 1024, 10, 0.01, "tanh"))
nn.layers.append(SoftmaxLayer(batch_size, 10))
```

### term 5
```
nn = NeuralNetwork()
batch_size = 20
learning_rate = 0.001  # *= 0.3 (E6, E9)
nn.layers.append(ConvLayer((32, 32), (5, 5), batch_size, 3, 32, 0.01, "relu"))
nn.layers.append(PoolLayer((2, 2), batch_size, 32, 28, 28))
nn.layers.append(ConvLayer((14, 14), (3, 3), batch_size, 32, 64, 0.01, "relu"))
nn.layers.append(PoolLayer((2, 2), batch_size, 64, 12, 12))
nn.layers.append(ConvLayer((6, 6), (3, 3), batch_size, 64, 64, 0.01, "relu"))
nn.layers.append(FlatLayer(batch_size, 64, 4, 4, 1024))
nn.layers.append(FCLayer(batch_size, 1024, 10, 0.01, "tanh"))
nn.layers.append(SoftmaxLayer(batch_size, 10))
```
key=86381, 48%

### term 4
```
batch_size = 20
learning_rate = 0.001
nn.layers.append(ConvLayer((32, 32), (5, 5), batch_size, 3, 32, 0.01, "relu"))
nn.layers.append(PoolLayer((2, 2), batch_size, 32, 28, 28))
nn.layers.append(ConvLayer((14, 14), (3, 3), batch_size, 32, 32, 0.01, "relu"))
nn.layers.append(PoolLayer((2, 2), batch_size, 32, 12, 12))
nn.layers.append(ConvLayer((6, 6), (3, 3), batch_size, 32, 64, 0.01, "relu"))
nn.layers.append(FlatLayer(batch_size, 64, 4, 4, 1024))
nn.layers.append(FCLayer(batch_size, 1024, 10, 0.01, "tanh"))
nn.layers.append(SoftmaxLayer(batch_size, 10))
```
data mean

### term 5 again
```
batch_size = 20
learning_rate = 0.01
nn.layers.append(ConvLayer((32, 32), (5, 5), batch_size, 3, 32, 0.1, "sigmoid"))
nn.layers.append(PoolLayer((2, 2), batch_size, 32, 28, 28))
nn.layers.append(ConvLayer((14, 14), (3, 3), batch_size, 32, 32, 0.1, "sigmoid"))
nn.layers.append(PoolLayer((2, 2), batch_size, 32, 12, 12))
nn.layers.append(ConvLayer((6, 6), (3, 3), batch_size, 32, 64, 0.1, "sigmoid"))
nn.layers.append(FlatLayer(batch_size, 64, 4, 4, 1024))
nn.layers.append(FCLayer(batch_size, 1024, 10, 0.01, "tanh"))
nn.layers.append(SoftmaxLayer(batch_size, 10))
```
data mean

### term 6 again
```
batch_size = 20
learning_rate = 0.001
nn.layers.append(ConvLayer((32, 32), (5, 5), batch_size, 3, 32, 0.01, "tanh"))
nn.layers.append(PoolLayer((2, 2), batch_size, 32, 28, 28))
nn.layers.append(ConvLayer((14, 14), (3, 3), batch_size, 32, 32, 0.01, "tanh"))
nn.layers.append(PoolLayer((2, 2), batch_size, 32, 12, 12))
nn.layers.append(ConvLayer((6, 6), (3, 3), batch_size, 32, 64, 0.01, "tanh"))
nn.layers.append(FlatLayer(batch_size, 64, 4, 4, 1024))
nn.layers.append(FCLayer(batch_size, 1024, 10, 0.01, "tanh"))
nn.layers.append(SoftmaxLayer(batch_size, 10))
```
data mean