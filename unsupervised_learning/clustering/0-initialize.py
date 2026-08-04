#!/usr/bin/env python3
"""module that initializes cluster centroids for K-means"""
import numpy as np


def initialize(X, k):
    """Initialize cluster centroids for K-means.

    Returns: numpy.ndarray of shape (k, d)
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(k, int) or k <= 0:
        return None

    low = X.min(axis=0)
    high = X.max(axis=0)

    return np.random.uniform(low, high, size=(k, X.shape[1]))
