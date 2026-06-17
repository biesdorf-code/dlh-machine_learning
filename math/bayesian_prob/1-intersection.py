#!/usr/bin/env python3
"""module for intersection: joint probability of data and each hypothesis"""
import numpy as np


def intersection(x, n, P, Pr):
    """Calculates the intersection of obtaining the data with each prior."""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    # Pr must match P exactly in type and shape to pair priors with hypotheses
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(P < 0) or np.any(P > 1):  # any() assesses an array here
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr < 0) or np.any(Pr > 1):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    # priors form a distribution, so they must sum to 1 (isclose for floats)
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

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

    # Likelihood for each hypothesis in P: L(x | n, P)
    likelihoods = coeff * success_term * failure_term

    # Intersection (joint probability): P(data AND P) = L(x | n, P) * Pr(P)
    # Element-wise weighting of each likelihood by its prior belief.
    intersections = likelihoods * Pr

    return intersections  # 1D nparray
