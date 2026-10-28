#!/usr/bin/env python3
"""Defines a gated recurrent unit cell."""
import numpy as np


class GRUCell:
    """Represents a gated recurrent unit."""

    def __init__(self, i, h, o):
        """Initializes the cell's weights and biases.

        i: dimensionality of the data
        h: dimensionality of the hidden state
        o: dimensionality of the outputs
        """
        self.Wz = np.random.randn(i + h, h)
        self.Wr = np.random.randn(i + h, h)
        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """Performs forward propagation for one time step.

        h_prev: numpy.ndarray of shape (m, h) with the previous hidden state
        x_t: numpy.ndarray of shape (m, i) with the data input for the cell
        Returns: h_next, y
        """
        hx = np.concatenate((h_prev, x_t), axis=1)
        z = 1 / (1 + np.exp(-(hx @ self.Wz + self.bz)))
        r = 1 / (1 + np.exp(-(hx @ self.Wr + self.br)))
        rhx = np.concatenate((r * h_prev, x_t), axis=1)
        h_tilde = np.tanh(rhx @ self.Wh + self.bh)
        h_next = (1 - z) * h_prev + z * h_tilde
        s = h_next @ self.Wy + self.by
        e = np.exp(s - np.max(s, axis=1, keepdims=True))
        y = e / np.sum(e, axis=1, keepdims=True)
        return h_next, y
