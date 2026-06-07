#!/usr/bin/env python3
"""integral of a polynomial"""


def poly_integral(poly, C=0):
    """integral of poly as a coefficient list"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not all(isinstance(c, (int, float)) for c in poly):
        return None
    if not isinstance(C, (int, float)):
        return None

    integral = [C]
    for power in range(len(poly)):
        new_coef = poly[power] / (power + 1)
        if new_coef.is_integer():
            new_coef = int(new_coef)
        integral.append(new_coef)

    while len(integral) > 1 and integral[-1] == 0:
        integral.pop()

    return integral