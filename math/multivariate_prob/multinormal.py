#!/usr/bin/env python3
"""module for representing a Multivariate Normal distribution"""
import numpy as np


class MultiNormal:
    """Represents a Multivariate Normal distribution.

    Attributes:
        mean (numpy.ndarray): shape (d, 1) — mean of the data.
        cov (numpy.ndarray): shape (d, d) — covariance matrix of the data.
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

        # Mean: sum across columns, divide by n, shape (d, 1)
        self.mean = np.sum(data, axis=1, keepdims=True) / n

        # Center the data: subtract mean from every column
        data_centered = data - self.mean

        # Covariance: (d,n) @ (n,d) / (n-1), shape (d, d)
        self.cov = (data_centered @ data_centered.T) / (n - 1)

    def pdf(self, x):
        """alculate the PDF at n-dimentional x point

        Args:
            x (numpy.ndarray): shape (d, 1)

        Returns:
            float: the value of the PDF at x.

        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")

        d = self.cov.shape[0]

        if x.shape != (d, 1):
            raise ValueError("x must have the shape ({}, 1)".format(d))

        # Normalization coefficient: 1 / ((2π)^(d/2) * |Σ|^(1/2))
        det = np.linalg.det(self.cov)
        coefficient = 1 / ((2 * np.pi) ** (d / 2) * np.sqrt(det))

        # Centered point
        x_centered = x - self.mean

        # Exponent: mahalanobis
        inv_cov = np.linalg.inv(self.cov)
        exponent = -0.5 * (x_centered.T @ inv_cov @ x_centered).squeeze()

        return float(coefficient * np.exp(exponent))
