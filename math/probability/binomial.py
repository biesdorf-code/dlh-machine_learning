#!/usr/bin/env python3
"""module that defines the class Binomial prob. distribution"""


class Binomial:
    """a Binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """initialize the instance"""
        if data is None:
            if n <= 0:  # n is the count of trials
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:  # if there is data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)

            squared_diffs = 0
            for value in data:
                squared_diffs += (value - mean) ** 2
            variance = squared_diffs / len(data)

            # moments

            p = 1 - (variance / mean)

            # n = mean / p, rounded because it represent a count of trials
            self.n = round(mean / p)

            # recalculate p
            self.p = float(mean / self.n)

    def pmf(self, k):
        """calculates the value of the PMF for a given number of successes"""

        # PMF - prob. of getting exactly k successes out of n trials
        # P(k) = C(n, k) * p^k * (1-p)^(n-k)
        if not isinstance(k, int):
            k = int(k)
        if k < 0 or k > self.n:  # Cant have more success than trials
            return 0

        # C(n, k) = n! / (k! * (n-k)!), built from hand-rolled factorials
        n_factorial = 1  # 0! = 1
        for i in range(1, self.n + 1):
            n_factorial *= i

        k_factorial = 1  # 0! = 1
        for i in range(1, k + 1):
            k_factorial *= i

        n_minus_k_factorial = 1  # 0! = 1
        for i in range(1, self.n - k + 1):
            n_minus_k_factorial *= i

        # factorials built, now the Bernoulli math
        # binomial coefficient represent the ways I can win, order of luck.

        binomial_coefficient = n_factorial / \
            (k_factorial * n_minus_k_factorial)

        success_part = self.p ** k
        failure_part = (1 - self.p) ** (self.n - k)

        pmf_value = binomial_coefficient * success_part * failure_part
        return pmf_value
