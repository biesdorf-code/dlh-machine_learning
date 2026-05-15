#!/usr/bin/env python3
"""module for element-wise operations numpy required"""


def np_elementwise(mat1, mat2):
    """Returns tuples of element wise +-*/"""
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
