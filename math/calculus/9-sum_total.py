#!/usr/bin/env python3
"""i squared"""


def summation_i_squared(n):
    """SUM i^2 until whatever n"""
    if not isinstance(n, int) or n < 1:
        return None
    return (n * (n + 1) * (2 * n + 1)) // 6
