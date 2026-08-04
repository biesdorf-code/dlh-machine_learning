#!/usr/bin/env python3
"""module implements k-means"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """Perform K-means on a dataset."""

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    initialize = __import__('0-initialize').initialize
    n, d = X.shape

    low = X.min(axis=0)
    high = X.max(axis=0)

    C = initialize(X, k)
    if C is None:
        return None, None

    for i in range(iterations):
        C_prev = C.copy()

        diff = X[:, np.newaxis] - C
        dist = np.linalg.norm(diff, axis=2)
        clss = np.argmin(dist, axis=1)

        for j in range(k):
            mask = X[clss == j]
            if mask.size == 0:
                C[j] = np.random.uniform(low, high)
            else:
                C[j] = mask.mean(axis=0)

        if np.array_equal(C, C_prev):
            break

    diff = X[:, np.newaxis] - C
    dist = np.linalg.norm(diff, axis=2)
    clss = np.argmin(dist, axis=1)

    return C, clss
