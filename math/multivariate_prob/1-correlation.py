#!/usr/bin/env python3
"""module for computing a correlation matrix"""
import numpy as np


def correlation(C):
    """calculate the correlation matrix from a covariance matrix

    Args:
        C (numpy.ndarray): shape (d, d) — covariance matrix.

    Returns:
        numpy.ndarray: shape (d, d) — correlation matrix.

    Raises:
        TypeError: if C is not a numpy.ndarray.
        ValueError: if C is not a 2D square matrix.
    """
    if not isinstance(C, np.ndarray):
        raise TypeError("C must be a numpy.ndarray")

    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        raise ValueError("C must be a 2D square matrix")

    d = C.shape[0]

    # Standard deviations from diagonal variances
    std = np.zeros(d)  # initializing
    for i in range(d):
        std[i] = C[i, i] ** 0.5

    # Denominator: std[i] * std[j] for every pair
    denominator = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            denominator[i, j] = std[i] * std[j]

    correlation_matrix = C / denominator  # broadcasting

    return correlation_matrix
