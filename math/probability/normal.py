#!/usr/bin/env python3
"""module that defines the class Normal prob. distro class"""


class Normal:
    """a Normal/Gaussian distribution"""

    e = 2.7182818285  # tip from project
    pi = 3.1415926536  # tip from project

    def __init__(self, data=None, mean=0., stddev=1.):
        """initialize the instance"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)  # cast
            self.stddev = float(stddev)  # cast
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = float(sum(data) / len(data))

            squared_diffs = 0
            for value in data:
                squared_diffs += (value - self.mean) ** 2
            variance = squared_diffs / len(data)
            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """calculates the z-score of a given x-value"""

        # z-score: how many standard deviations x is from the mean
        # Zones: -2, -1, 0, 1, 2

        z = (x - self.mean) / self.stddev
        return z

    def x_value(self, z):
        """calculates the x-value of a given z-score"""

        # inverse of z_score: convert a z back into an x-value

        x = self.mean + z * self.stddev
        return x

    def pdf(self, x):
        """calculates the value of the PDF for a given x-value"""

        # PDF - the height of the bell
        # f(x) = (1 / (stddev * sqrt(2*pi))) * e^(-(x-mean)^2 / (2*stddev^2))

        left_multiplier = 1 / (self.stddev * (2 * Normal.pi) ** 0.5)
        exponent = -((x - self.mean) ** 2) / (2 * self.stddev ** 2)
        pdf_value = left_multiplier * (Normal.e ** exponent)

        return pdf_value
