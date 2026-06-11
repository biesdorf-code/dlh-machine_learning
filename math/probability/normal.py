#!/usr/bin/env python3
"""module that defines the class Normal probability distribution class"""


class Normal:
    """a Normal distribution"""

    def __init__(self, data=None, mean=0., stddev=1.):
        """initialize the instance"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
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
