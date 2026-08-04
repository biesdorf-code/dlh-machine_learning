#!/usr/bin/env python3
"""module that calculates the max step in the EM algorithm for a GMM"""
import numpy as np


def maximization(X, g):
    """Calculate the maximization step in the EM algorithm for a GMM.

    Returns: pi, m, S, or None, None, None on failure
        pi: numpy.ndarray of shape (k,)
        m: numpy.ndarray of shape (k, d)
        S: numpy.ndarray of shape (k, d, d)
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    if g.shape[1] != n:
        return None, None, None
    if not np.all(np.isclose(np.sum(g, axis=0), 1)):
        return None, None, None

    try:
        nk = np.sum(g, axis=1)

        pi = nk / n
        m = (g @ X) / nk[:, np.newaxis]
        S = np.zeros((k, d, d))

        for i in range(k):
            diff = X - m[i]
            S[i] = (g[i] * diff.T) @ diff / nk[i]

        return pi, m, S
    except Exception:
        return None, None, None
