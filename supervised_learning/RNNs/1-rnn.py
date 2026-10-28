#!/usr/bin/env python3
"""Defines forward propagation for a simple RNN."""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """Performs forward propagation for a simple RNN.

    rnn_cell: instance of RNNCell used for the forward propagation
    X: numpy.ndarray of shape (t, m, i) with the data
    h_0: numpy.ndarray of shape (m, h) with the initial hidden state
    Returns: H, Y
    """
    t, m, _ = X.shape
    H = np.zeros((t + 1, m, h_0.shape[1]))
    H[0] = h_0
    Y = []
    for step in range(t):
        H[step + 1], y = rnn_cell.forward(H[step], X[step])
        Y.append(y)
    return H, np.array(Y)
