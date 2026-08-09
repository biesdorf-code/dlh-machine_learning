#!/usr/bin/env python3
""" calculates the gradients of Y """
import numpy as np
Q_affinities = __import__('5-Q_affinities').Q_affinities


def grads(Y, P):
    """ calculates the gradients of Y.

    Args:
        Y: numpy.ndarray of shape (n, ndim), the low dimensional
            transformation of X.
        P: numpy.ndarray of shape (n, n), the P affinities of X.

    Returns:
        (dY, Q):
            dY: numpy.ndarray of shape (n, ndim), the gradients of Y.
            Q: numpy.ndarray of shape (n, n), the Q affinities of Y.
    """
    n, ndim = Y.shape
    Q, num = Q_affinities(Y)
    dY = np.zeros((n, ndim))
    PQ = (P - Q) * num

    for i in range(n):
        dY[i] = np.matmul(PQ[i], Y[i] - Y)

    return dY, Q
