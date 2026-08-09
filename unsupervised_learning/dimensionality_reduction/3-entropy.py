#!/usr/bin/env python3
""" calculates the Shannon entropy and P affinities of a data point """
import numpy as np


def HP(Di, beta):
    """ calculates the Shannon entropy and P affinities of a data point.

    Args:
        Di: numpy.ndarray of shape (n - 1,), the pairwise distances between
            a data point and all the other points except itself.
        beta: numpy.ndarray of shape (1,), the beta value for the Gaussian.

    Returns:
        (Hi, Pi):
            Hi: the Shannon entropy of the points.
            Pi: numpy.ndarray of shape (n - 1,), the P affinities.
    """
    num = np.exp(-Di * beta)
    Pi = num / np.sum(num)
    Hi = -np.sum(Pi * np.log2(Pi))
    return Hi, Pi
