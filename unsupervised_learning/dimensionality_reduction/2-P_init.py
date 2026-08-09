#!/usr/bin/env python3
""" initializes the variables required to calculate the P affinities """
import numpy as np


def P_init(X, perplexity):
    """ initializes all variables required to calculate the P affinities.

    Args:
        X: numpy.ndarray of shape (n, d), the dataset.
        perplexity: the perplexity all Gaussian distributions should have.

    Returns:
        (D, P, betas, H):
            D: (n, n) squared pairwise distances, 0s on the diagonal.
            P: (n, n) P affinities, initialized to 0s.
            betas: (n, 1) beta values, initialized to 1s.
            H: the Shannon entropy for perplexity, with a base of 2.
    """
    n = X.shape[0]
    sum_sq = np.sum(np.square(X), axis=1)
    D = sum_sq + (sum_sq[:, None] - 2 * np.matmul(X, X.T))
    np.fill_diagonal(D, 0)
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)
    return D, P, betas, H
