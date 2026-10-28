#!/usr/bin/env python3
""" conducts forward propagation using Dropout """
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """ conducts forward propagation using Dropout """
    cache = {'A0': X}
    for i in range(1, L + 1):
        Z = np.matmul(weights['W{}'.format(i)], cache['A{}'.format(i - 1)]) \
            + weights['b{}'.format(i)]
        if i == L:
            T = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            cache['A{}'.format(i)] = T / np.sum(T, axis=0, keepdims=True)
        else:
            D = np.random.binomial(1, keep_prob, size=Z.shape)
            cache['D{}'.format(i)] = D
            cache['A{}'.format(i)] = np.tanh(Z) * D / keep_prob
    return cache
