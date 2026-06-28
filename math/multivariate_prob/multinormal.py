#!/usr/bin/env python3
"""module for representing a Multivariate Normal distribution"""
import numpy as np


class MultiNormal:
    """Multivariate Normal distribution

    Attributes:
        mean (numpy.ndarray): shape (d, 1) — mean of the data
        cov (numpy.ndarray): shape (d, d) — covariance matrix of the data
    """

    def __init__(self, data):
        """Initialize a MultiNormal instance.

        Args:
            data (numpy.ndarray): shape (d, n) — d dimensions, n data points.

        """
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")

        d, n = data.shape

        if n < 2:
            raise ValueError("data must contain multiple data points")

        # Mean: sum across !columns, divide by n, shape (d, 1)
        self.mean = np.sum(data, axis=1, keepdims=True) / n

        # Center the data: subtract mean from every column
        data_centered = data - self.mean

        # Covariance: (d,n) @ (n,d) / (n-1), shape (d, d)
        self.cov = (data_centered @ data_centered.T) / (n - 1)
