#!/usr/bin/env python3
""" updates weights and biases using L2 regularized gradient descent """
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ updates weights and biases using L2 regularized gradient descent """
    m = Y.shape[1]
    dZ = cache['A{}'.format(L)] - Y
    for i in range(L, 0, -1):
        W = weights['W{}'.format(i)]
        A_prev = cache['A{}'.format(i - 1)]
        dW = np.matmul(dZ, A_prev.T) / m + lambtha / m * W
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dZ = np.matmul(W.T, dZ) * (1 - A_prev ** 2)
        weights['W{}'.format(i)] -= alpha * dW
        weights['b{}'.format(i)] -= alpha * db
