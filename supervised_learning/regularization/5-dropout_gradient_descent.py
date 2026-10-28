#!/usr/bin/env python3
""" updates the weights of a network using dropout gradient descent """
import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """ updates the weights of a network using dropout gradient descent """
    m = Y.shape[1]
    dZ = cache['A{}'.format(L)] - Y
    for i in range(L, 0, -1):
        W = weights['W{}'.format(i)]
        A_prev = cache['A{}'.format(i - 1)]
        dW = np.matmul(dZ, A_prev.T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        if i > 1:
            dA = np.matmul(W.T, dZ) * cache['D{}'.format(i - 1)] / keep_prob
            dZ = dA * (1 - A_prev ** 2)
        weights['W{}'.format(i)] -= alpha * dW
        weights['b{}'.format(i)] -= alpha * db
