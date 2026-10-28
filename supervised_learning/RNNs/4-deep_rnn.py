#!/usr/bin/env python3
"""Defines forward propagation for a deep RNN."""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """Performs forward propagation for a deep RNN.

    rnn_cells: list of RNNCell instances of length l, one per layer
    X: numpy.ndarray of shape (t, m, i) with the data
    h_0: numpy.ndarray of shape (l, m, h) with the initial hidden states
    Returns: H, Y
    """
    t = X.shape[0]
    H = np.zeros((t + 1,) + h_0.shape)
    H[0] = h_0
    Y = []
    for step in range(t):
        x = X[step]
        for layer, cell in enumerate(rnn_cells):
            H[step + 1, layer], y = cell.forward(H[step, layer], x)
            x = H[step + 1, layer]
        Y.append(y)
    return H, np.array(Y)
