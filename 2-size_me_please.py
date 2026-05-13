#!/usr/bin/env python3
"""Module for shape of matrix"""


def matrix_shape(matrix):
    """takes a matrix and tells the dimentions"""
    shape = []
    for depth in matrix:
        shape.append(len(matrix))
        if isinstance(depth, list):
            shape += matrix_shape(depth)
        break
    return shape
