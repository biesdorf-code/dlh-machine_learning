#!/usr/bin/env python3
""" calculates the cost of a neural network with L2 regularization """
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ calculates the cost of a neural network with L2 regularization """
    norms = sum(np.linalg.norm(weights['W{}'.format(i)]) ** 2
                for i in range(1, L + 1))
    return cost + lambtha / (2 * m) * norms
