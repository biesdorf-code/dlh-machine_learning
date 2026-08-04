#!/usr/bin/env python3
"""module that calculates the total intra-cluster variance for a data set"""
import numpy as np


def variance(X, C):
    """Calculate the total intra-cluster variance for a data set.

    Returns: var, or None on failure
        var: total variance (float)
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    try:
        # (n, 1, d) - (k, d) broadcasts to (n, k, d)
        diff = X[:, np.newaxis] - C  # numpy.ndarray of shape (n, k, d)
        dist = np.linalg.norm(diff, axis=2)  # numpy.ndarray of shape (n, k)
        min_dist = np.min(dist, axis=1)  # numpy.ndarray of shape (n,)

        return np.sum(min_dist ** 2)  # numpy scalar (float64)
    except Exception:
        return None
