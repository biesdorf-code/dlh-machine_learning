#!/usr/bin/env python3
"""module that calculates a GMM from a dataset using sklearn"""
import sklearn.mixture


def gmm(X, k):
    """Calculate a GMM from a dataset.

    Returns: pi, m, S, clss, bic
        pi: numpy.ndarray of shape (k,)
        m: numpy.ndarray of shape (k, d)
        S: numpy.ndarray of shape (k, d, d)
        clss: numpy.ndarray of shape (n,)
        bic: BIC value for the model
    """
    model = sklearn.mixture.GaussianMixture(n_components=k).fit(X)

    pi = model.weights_
    m = model.means_
    S = model.covariances_
    clss = model.predict(X)
    bic = model.bic(X)

    return pi, m, S, clss, bic
