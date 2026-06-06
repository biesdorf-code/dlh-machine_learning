#!/usr/bin/env python3
"""derivative of a polynomial"""


def poly_derivative(poly):
    """derivative of poly as a coefficient list"""
    if not isinstance(poly, list) or len(poly) == 0:
        return None

    for coef in poly:
        if not isinstance(coef, (int, float)):
            return None

    if len(poly) == 1:
        return [0]

    derivative = []
    for power in range(1, len(poly)):
        coef = poly[power]
        new_coef = coef * power
        derivative.append(new_coef)

    return derivative
