#!/usr/bin/env python3
"""Positional encoding for a transformer"""
import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculates the sinusoidal positional encoding for a transformer

    max_seq_len: maximum sequence length
    dm: model depth
    Returns: numpy.ndarray of shape (max_seq_len, dm)
    """
    pos = np.arange(max_seq_len)[:, np.newaxis]
    i = np.arange(dm)[np.newaxis, :]
    angles = pos / np.power(10000, (2 * (i // 2)) / np.float32(dm))
    PE = np.zeros((max_seq_len, dm))
    PE[:, 0::2] = np.sin(angles[:, 0::2])
    PE[:, 1::2] = np.cos(angles[:, 1::2])
    return PE
