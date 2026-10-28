#!/usr/bin/env python3
"""Defines a cell of a simple RNN."""
import numpy as np


class RNNCell:
    """Represents a cell of a simple RNN."""

    def __init__(self, i, h, o):
        """Initializes the cell's weights and biases.

        i: dimensionality of the data
        h: dimensionality of the hidden state
        o: dimensionality of the outputs
        """
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step.

        h_prev: numpy.ndarray of shape (m, h) with the previous hidden state
        x_t: numpy.ndarray of shape (m, i) with the data input for the cell
        Returns: h_next, y
        """
        h_next = np.tanh(np.concatenate((h_prev, x_t), axis=1) @ self.Wh
                         + self.bh)
        z = h_next @ self.Wy + self.by
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        y = e / np.sum(e, axis=1, keepdims=True)
        return h_next, y
