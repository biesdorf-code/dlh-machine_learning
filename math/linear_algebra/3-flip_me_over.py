#!/usr/bin/env python3
"""Module for transposing a 2D matrix."""


def matrix_transpose(matrix):
    """transpose a 2d matrix"""
    new_matrix = []
    for old_col in range(len(matrix[0])):
        new_row = []
        for r in range(len(matrix)):
            new_row.append(matrix[r][old_col])
        new_matrix.append(new_row)
    return new_matrix
