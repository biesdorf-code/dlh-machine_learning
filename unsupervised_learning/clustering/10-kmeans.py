#!/usr/bin/env python3
"""module that performs K-means on a dataset using sklearn"""
import sklearn.cluster


def kmeans(X, k):
    """Perform K-means on a dataset.

    Returns: C, clss
        C: numpy.ndarray of shape (k, d)
        clss: numpy.ndarray of shape (n,)
    """
    model = sklearn.cluster.KMeans(n_clusters=k).fit(X)

    C = model.cluster_centers_
    clss = model.labels_

    return C, clss
