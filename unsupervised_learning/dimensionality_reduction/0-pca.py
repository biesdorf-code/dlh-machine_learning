#!/usr/bin/env python3
""" performs principal components analysis (PCA) on a dataset """
import numpy as np


def pca(X, var=0.95):
    """ performs PCA on a dataset.

    Args:
        X: numpy.ndarray of shape (n, d), zero-mean dataset.
        var: fraction of the variance to maintain.

    Returns:
        W: numpy.ndarray of shape (d, nd), the weights matrix.
    """
    _, S, Vt = np.linalg.svd(X)
    ratios = np.cumsum(S) / np.sum(S)
    nd = np.argwhere(ratios >= var)[0, 0]
    return Vt[:nd + 1].T
