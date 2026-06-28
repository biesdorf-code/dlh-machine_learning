#!/usr/bin/env python3
"""module for computing the mean and covariance"""
import numpy as np


def mean_cov(X):
    """Calculate the mean and covariance matrix of x

    Args:
        X (numpy.ndarray): shape (n, d) — n data points, d dimensions.

    Returns:
        mean (numpy.ndarray): shape (1, d) — mean of each dimension.
        cov (numpy.ndarray): shape (d, d) — covariance matrix.

    """
    if not isinstance(X, np.ndarray) or X.ndim != 2:
        raise TypeError("X must be a 2D numpy.ndarray")

    n, d = X.shape

    if n < 2:
        raise ValueError("X must contain multiple data points")

    # Mean: sum each column, divide by n, keep shape (1, d)
    mean = np.sum(X, axis=0, keepdims=True) / n

    # Center the data: subtract mean from every row (broadcasting)
    X_centered = X - mean

    # Covariance: (X_c.T @ X_c) / (n - 1), shape (d, d)
    covariance_matrix = (X_centered.T @ X_centered) / (n - 1)

    return mean, covariance_matrix
