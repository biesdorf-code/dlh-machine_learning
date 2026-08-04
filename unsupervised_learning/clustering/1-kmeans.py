#!/usr/bin/env python3
import numpy as np


def kmeans(X, k, iterations=1000):
    """Perform K-means on a dataset.

    Returns: C, clss, or None, None on failure
        C: numpy.ndarray of shape (k, d)
        clss: numpy.ndarray of shape (n,)
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k <= 0:
        return None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None

    initialize = __import__('0-initialize').initialize
    n, d = X.shape  # ints

    low = X.min(axis=0)  # numpy.ndarray of shape (d,)
    high = X.max(axis=0)  # numpy.ndarray of shape (d,)

    C = initialize(X, k)  # numpy.ndarray of shape (k, d)
    if C is None:
        return None, None

    for i in range(iterations):
        C_prev = C.copy()  # numpy.ndarray of shape (k, d)

        # (n, 1, d) - (k, d) broadcasts to (n, k, d)
        diff = X[:, np.newaxis] - C  # numpy.ndarray of shape (n, k, d)
        dist = np.linalg.norm(diff, axis=2)  # numpy.ndarray of shape (n, k)
        clss = np.argmin(dist, axis=1)  # numpy.ndarray of shape (n,)

        for j in range(k):
            mask = X[clss == j]  # numpy.ndarray of shape (points_in_j, d)
            if mask.size == 0:
                C[j] = np.random.uniform(low, high)  # ndarray (d,)
            else:
                C[j] = mask.mean(axis=0)  # numpy.ndarray of shape (d,)

        if np.array_equal(C, C_prev):
            break

    diff = X[:, np.newaxis] - C  # numpy.ndarray of shape (n, k, d)
    dist = np.linalg.norm(diff, axis=2)  # numpy.ndarray of shape (n, k)
    clss = np.argmin(dist, axis=1)  # numpy.ndarray of shape (n,)

    return C, clss
