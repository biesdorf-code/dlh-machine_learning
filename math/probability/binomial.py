#!/usr/bin/env python3
"""module that defines the class Binomial prob. distribution"""


class Binomial:
    """a Binomial distribution"""

    def __init__(self, data=None, n=1, p=0.5):
        """initialize the instance"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)

            squared_diffs = 0
            for value in data:
                squared_diffs += (value - mean) ** 2
            variance = squared_diffs / len(data)

            # method of moments:
            # mean = n*p   and   variance = n*p*(1-p)
            # so variance/mean = (1-p)  ->  p = 1 - variance/mean
            p = 1 - (variance / mean)

            # n = mean / p, rounded (not cast!) to nearest integer
            self.n = round(mean / p)

            # recalculate p now that n is a clean integer
            self.p = float(mean / self.n)
