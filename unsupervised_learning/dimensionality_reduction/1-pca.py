#!/usr/bin/env python3
""" performs principal components analysis (PCA) on a dataset """
import numpy as np


def pca(X, ndim):
    """ performs PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d), the dataset.
        ndim: the new dimensionality of the transformed X.

    Returns:
        T: numpy.ndarray of shape (n, ndim), the transformed X.
    """
    X_m = X - np.mean(X, axis=0)
    _, _, Vt = np.linalg.svd(X_m)
    return np.matmul(X_m, Vt[:ndim].T)
