#!/usr/bin/env python3
"""Defines a bidirectional RNN cell."""
import numpy as np


class BidirectionalCell:
    """Represents a bidirectional cell of an RNN."""

    def __init__(self, i, h, o):
        """Initializes the cell's weights and biases.

        i: dimensionality of the data
        h: dimensionality of the hidden states
        o: dimensionality of the outputs
        """
        self.Whf = np.random.randn(i + h, h)
        self.Whb = np.random.randn(i + h, h)
        self.Wy = np.random.randn(2 * h, o)
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Calculates the forward-direction hidden state for one time step.

        h_prev: numpy.ndarray of shape (m, h) with the previous hidden state
        x_t: numpy.ndarray of shape (m, i) with the data input for the cell
        Returns: h_next, the next hidden state
        """
        hx = np.concatenate((h_prev, x_t), axis=1)
        return np.tanh(hx @ self.Whf + self.bhf)
