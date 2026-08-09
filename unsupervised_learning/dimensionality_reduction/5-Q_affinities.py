#!/usr/bin/env python3
""" calculates the Q affinities """
import numpy as np


def Q_affinities(Y):
    """ calculates the Q affinities.

    Args:
        Y: numpy.ndarray of shape (n, ndim), the low dimensional
            transformation of X.

    Returns:
        (Q, num):
            Q: numpy.ndarray of shape (n, n), the Q affinities.
            num: numpy.ndarray of shape (n, n), the numerator of the Q
                affinities.
    """
    sum_sq = np.sum(np.square(Y), axis=1)
    D = sum_sq + (sum_sq[:, None] - 2 * np.matmul(Y, Y.T))
    num = 1 / (1 + D)
    np.fill_diagonal(num, 0)
    Q = num / np.sum(num)
    return Q, num
