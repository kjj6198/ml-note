import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def relu(x):
    return np.maximum(x, 0)


def relu_grad(x):
    grad = np.zeros(x)
    grad[x >= 0] = 1
    return grad


def cross_entropy_loss(y, t):
    if y.dim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return (-1 * np.sum(t * np.log(y))) / batch_size


def softmax_cross_entropy(X, t):
    y = softmax(X)
    return cross_entropy_loss(y, t)
