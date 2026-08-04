#!/usr/bin/env python3
"""module that finds the best number of clusters for a GMM using the BIC"""
import numpy as np
expectation_maximization = __import__('8-EM').expectation_maximization


def BIC(X, kmin=1, kmax=None, iterations=1000, tol=1e-5, verbose=False):
    """Find the best number of clusters for a GMM using the Bayesian
    Information Criterion.

    Returns: best_k, best_result, l, b, or None * 4 on failure
        best_k: int, best value for k based on its BIC
        best_result: tuple of (pi, m, S) for the best k
        l: numpy.ndarray of shape (kmax - kmin + 1)
        b: numpy.ndarray of shape (kmax - kmin + 1)
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None
    if not isinstance(kmin, int) or kmin <= 0:
        return None, None, None, None
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmax, int) or kmax <= 0 or kmax < kmin:
        return None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None

    n, d = X.shape

    likelihoods = []
    bics = []
    results = []

    for k in range(kmin, kmax + 1):
        pi, m, S, g, likelihood = expectation_maximization(
            X, k, iterations, tol, verbose)
        if pi is None:
            return None, None, None, None

        p = (k * d) + (k * d * (d + 1) / 2) + (k - 1)
        bic = p * np.log(n) - 2 * likelihood

        results.append((pi, m, S))
        likelihoods.append(likelihood)
        bics.append(bic)

    likelihoods = np.array(likelihoods)
    bics = np.array(bics)

    best = np.argmin(bics)

    return kmin + best, results[best], likelihoods, bics
