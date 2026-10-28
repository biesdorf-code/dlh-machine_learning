#!/usr/bin/env python3
"""Defines an LSTM cell."""
import numpy as np


class LSTMCell:
    """Represents an LSTM unit."""

    def __init__(self, i, h, o):
        """Initializes the cell's weights and biases.

        i: dimensionality of the data
        h: dimensionality of the hidden state
        o: dimensionality of the outputs
        """
        self.Wf = np.random.randn(i + h, h)
        self.Wu = np.random.randn(i + h, h)
        self.Wc = np.random.randn(i + h, h)
        self.Wo = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """Performs forward propagation for one time step.

        h_prev: numpy.ndarray of shape (m, h) with the previous hidden state
        c_prev: numpy.ndarray of shape (m, h) with the previous cell state
        x_t: numpy.ndarray of shape (m, i) with the data input for the cell
        Returns: h_next, c_next, y
        """
        hx = np.concatenate((h_prev, x_t), axis=1)
        f = 1 / (1 + np.exp(-(hx @ self.Wf + self.bf)))
        u = 1 / (1 + np.exp(-(hx @ self.Wu + self.bu)))
        c_tilde = np.tanh(hx @ self.Wc + self.bc)
        o = 1 / (1 + np.exp(-(hx @ self.Wo + self.bo)))
        c_next = f * c_prev + u * c_tilde
        h_next = o * np.tanh(c_next)
        s = h_next @ self.Wy + self.by
        e = np.exp(s - np.max(s, axis=1, keepdims=True))
        y = e / np.sum(e, axis=1, keepdims=True)
        return h_next, c_next, y
