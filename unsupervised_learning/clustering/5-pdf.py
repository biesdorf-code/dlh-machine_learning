#!/usr/bin/env python3
"""module that calculates the probability density function of a Gaussian"""
import numpy as np


def pdf(X, m, S):
    """Calculate the PDF of a Gaussian distribution.

    Returns: P, or None on failure
        P: numpy.ndarray of shape (n,)
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    d = X.shape[1]
    if m.shape[0] != d or S.shape[0] != d or S.shape[1] != d:
        return None

    try:
        det = np.linalg.det(S)
        inv = np.linalg.inv(S)

        diff = X - m
        exponent = -0.5 * np.sum(diff @ inv * diff, axis=1)
        norm = 1 / np.sqrt(((2 * np.pi) ** d) * det)
        P = norm * np.exp(exponent)

        return np.maximum(P, 1e-300)
    except Exception:
        return None
