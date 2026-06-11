#!/usr/bin/env python3
"""module that defines the class Exponential probability distribution class"""


class Exponential:
    """an Exponential distribution"""

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
            self.lambtha = float(1 / (sum(data) / len(data)))

    def pdf(self, x):
        """calculates the value of the PDF for a given time period"""

        # PDF - Probability Density Function, Density is for continuous values

        if x < 0:
            return 0

        e_to_the_neg_lambtha_x = Exponential.e ** (-self.lambtha * x)
        pdf_value = self.lambtha * e_to_the_neg_lambtha_x

        # the future wait doesn't care how long you've already waited.

        return pdf_value

    def cdf(self, x):
        """calculates the value of the CDF for a given time period"""
        # what's the probability the event happens within time x
        # CDF - Cumulative Distribution Function, P(wait <= x)

        if x < 0:
            return 0

        e_to_the_neg_lambtha_x = Exponential.e ** (-self.lambtha * x)
        cdf_value = 1 - e_to_the_neg_lambtha_x
        # denovo - what's the probability the event happens within time x
        return cdf_value
