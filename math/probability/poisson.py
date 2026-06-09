#!/usr/bin/env python3
"""module that defines the class Poisson probability distribution class"""


class Poisson:
    """a Poisson distribution"""

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """initialize the instance"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)  # cast
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """calculates the value of the PMF for a given number of successes"""

        # PMF - Probability Mass Function, Mass is for discrete, whole numbers

        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0

        factorial = 1
        for i in range(1, k + 1):
            factorial *= i

        # core: P(k)

        lambtha_to_the_k = self.lambtha ** k
        e_to_the_neg_lambtha = Poisson.e ** -self.lambtha

        numerator = lambtha_to_the_k * e_to_the_neg_lambtha
        pmf_value = numerator / factorial

        return pmf_value
