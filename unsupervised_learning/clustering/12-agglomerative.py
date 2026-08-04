#!/usr/bin/env python3
"""module that performs agglomerative clustering on a dataset"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """Perform agglomerative clustering with Ward linkage and display
    the dendrogram.

    Returns: clss
        clss: numpy.ndarray of shape (n,) with the cluster indices
    """
    links = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(
        links, t=dist, criterion='distance')

    plt.figure()
    scipy.cluster.hierarchy.dendrogram(links, color_threshold=dist)
    plt.show()

    return clss
