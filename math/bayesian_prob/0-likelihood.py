#!/usr/bin/env python3
"""mopdule for likelihood function, as opposed to probability"""
import numpy as np


def likelihood(x, n, P):
    """Calculates the likelihood of obtaining the data given side effects."""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P < 0) or np.any(P > 1):  # any() assesses an array here
        raise ValueError("All values in P must be in the range [0, 1]")

    factorial = np.math.factorial  # np.math will not work in Numpy 2.0

    # Binomial PMF: L(x | n, P) = C(n, x) * P^x * (1 - P)^(n - x)
    # The binomial coefficient C(n, x) = n! / (x! * (n - x)!) counts the
    # number of distinct ways x successes can occur among n trials.
    n_factorial = factorial(n)
    x_factorial = factorial(x)
    n_minus_x_factorial = factorial(n - x)
    coeff = n_factorial / (x_factorial * n_minus_x_factorial)

    # P^x is the probability of observing exactly x successes (side effects),
    # evaluated across each hypothetical probability in P.
    success_term = P ** x

    # (1 - P)^(n - x) is the probability of the remaining (n - x) failures
    # (patients without severe side effects)
    failure_term = (1 - P) ** (n - x)

    # Likelihood for each hypothesis in P: combine the count of arrangements
    # with the probability of one such arrangement.
    likelihoods = coeff * success_term * failure_term

    return likelihoods  # 1D nparray
