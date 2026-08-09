#!/usr/bin/env python3
""" performs a t-SNE transformation """
import numpy as np
pca = __import__('1-pca').pca
P_affinities = __import__('4-P_affinities').P_affinities
grads = __import__('6-grads').grads
cost = __import__('7-cost').cost


def tsne(X, ndims=2, idims=50, perplexity=30.0, iterations=1000, lr=500):
    """ performs a t-SNE transformation.

    Args:
        X: numpy.ndarray of shape (n, d), the dataset.
        ndims: the new dimensional representation of X.
        idims: the intermediate dimensional representation of X after PCA.
        perplexity: the perplexity.
        iterations: the number of iterations.
        lr: the learning rate.

    Returns:
        Y: numpy.ndarray of shape (n, ndims), the optimized low dimensional
            transformation of X.
    """
    n = X.shape[0]
    X = pca(X, idims)
    P = P_affinities(X, perplexity=perplexity) * 4
    Y = np.random.randn(n, ndims)
    dY_prev = np.zeros((n, ndims))

    for i in range(1, iterations + 1):
        dY, Q = grads(Y, P)
        momentum = 0.5 if i <= 20 else 0.8
        dY_prev = momentum * dY_prev - lr * dY
        Y = Y + dY_prev
        Y = Y - np.mean(Y, axis=0)

        if i % 100 == 0:
            print('Cost at iteration {}: {}'.format(i, cost(P, Q)))
        if i == 100:
            P = P / 4

    return Y
