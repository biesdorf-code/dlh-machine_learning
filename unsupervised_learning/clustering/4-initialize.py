#!/usr/bin/env python3
"""module that initializes variables for a Gaussian Mixture Model"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """Initialize variables for a Gaussian Mixture Model.

    Returns: pi, m, S, or None, None, None on failure
        pi: numpy.ndarray of shape (k,)
        m: numpy.ndarray of shape (k, d)
        S: numpy.ndarray of shape (k, d, d)
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None

    d = X.shape[1]

    pi = np.full((k,), 1 / k)
    m, _ = kmeans(X, k)
    if m is None:
        return None, None, None
    S = np.tile(np.identity(d), (k, 1, 1))

    return pi, m, S
