#!/usr/bin/env python3
""" normalizes an unactivated output using batch normalization """
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """ normalizes an unactivated output using batch normalization """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)
    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)

    return gamma * Z_norm + beta
