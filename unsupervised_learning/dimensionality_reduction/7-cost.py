#!/usr/bin/env python3
""" calculates the cost of the t-SNE transformation """
import numpy as np


def cost(P, Q):
    """ calculates the cost of the t-SNE transformation.

    Args:
        P: numpy.ndarray of shape (n, n), the P affinities.
        Q: numpy.ndarray of shape (n, n), the Q affinities.

    Returns:
        C: the cost of the transformation.
    """
    P = np.maximum(P, 1e-12)
    Q = np.maximum(Q, 1e-12)
    return np.sum(P * np.log(P / Q))
