#!/usr/bin/env python3
""" calculates the symmetric P affinities of a data set """
import numpy as np
P_init = __import__('2-P_init').P_init
HP = __import__('3-entropy').HP


def P_affinities(X, tol=1e-5, perplexity=30.0):
    """ calculates the symmetric P affinities of a data set.

    Args:
        X: numpy.ndarray of shape (n, d), the dataset.
        tol: maximum tolerance allowed for the difference in Shannon
            entropy from perplexity.
        perplexity: the perplexity all Gaussian distributions should have.

    Returns:
        P: numpy.ndarray of shape (n, n), the symmetric P affinities.
    """
    n = X.shape[0]
    D, P, betas, H = P_init(X, perplexity)

    for i in range(n):
        low, high = None, None
        Di = np.delete(D[i], i)
        Hi, Pi = HP(Di, betas[i])

        while np.abs(Hi - H) > tol:
            if Hi > H:
                low = betas[i, 0]
                if high is None:
                    betas[i, 0] = betas[i, 0] * 2
                else:
                    betas[i, 0] = (low + high) / 2
            else:
                high = betas[i, 0]
                if low is None:
                    betas[i, 0] = betas[i, 0] / 2
                else:
                    betas[i, 0] = (low + high) / 2
            Hi, Pi = HP(Di, betas[i])

        P[i] = np.insert(Pi, i, 0)

    return (P + P.T) / (2 * n)
