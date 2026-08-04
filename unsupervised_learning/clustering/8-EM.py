#!/usr/bin/env python3
"""module that performs the expectation maximization for a GMM"""
import numpy as np
initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """Perform the expectation maximization for a GMM.

    Returns: pi, m, S, g, l, or None * 5 on failure
        pi: numpy.ndarray of shape (k,)
        m: numpy.ndarray of shape (k, d)
        S: numpy.ndarray of shape (k, d, d)
        g: numpy.ndarray of shape (k, n)
        l: log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None
    if not isinstance(iterations, int) or iterations <= 0:
        return None, None, None, None, None
    if not isinstance(tol, float) or tol < 0:
        return None, None, None, None, None
    if not isinstance(verbose, bool):
        return None, None, None, None, None

    pi, m, S = initialize(X, k)
    if pi is None:
        return None, None, None, None, None

    g, likelihood = expectation(X, pi, m, S)
    if g is None:
        return None, None, None, None, None

    for i in range(iterations):
        if verbose and i % 10 == 0:
            print('Log Likelihood after {} iterations: {}'
                  .format(i, round(likelihood, 5)))

        pi, m, S = maximization(X, g)
        if pi is None:
            return None, None, None, None, None

        g, new_likelihood = expectation(X, pi, m, S)
        if g is None:
            return None, None, None, None, None

        if abs(new_likelihood - likelihood) <= tol:
            likelihood = new_likelihood
            break

        likelihood = new_likelihood

    if verbose:
        print('Log Likelihood after {} iterations: {}'
              .format(i + 1, round(likelihood, 5)))

    return pi, m, S, g, likelihood
