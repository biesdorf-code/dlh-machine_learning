#!/usr/bin/env python3
"""definiteness of a matrix"""

import numpy as np


def definiteness(matrix):
    """definiteness of a matrix."""

    # check if it is a numpy.ndarray
    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    # check if the matrix is valid:
    # must be 2D, square, and symmetric
    if matrix.ndim != 2:
        return None
    n, m = matrix.shape
    if n != m:
        return None
    if not np.allclose(matrix, matrix.T):
        return None

    # get the eigenvalues
    # definiteness is determined by the signs of all eigenvalues
    eigenvalues = np.linalg.eigvalsh(matrix)

    # all eigenvalues > 0 → Positive definite
    if np.all(eigenvalues > 0):
        return "Positive definite"

    # all eigenvalues >= 0 (some are zero) → Positive semi-definite
    if np.all(eigenvalues >= 0):
        return "Positive semi-definite"

    # all eigenvalues < 0 → Negative definite
    if np.all(eigenvalues < 0):
        return "Negative definite"

    # all eigenvalues <= 0 (some are zero) → Negative semi-definite
    if np.all(eigenvalues <= 0):
        return "Negative semi-definite"

    # mix of positive and negative → Indefinite
    return "Indefinite"
