#!/usr/bin/env python3
"""module for matrix multiplication"""


def mat_mul(mat1, mat2):
    """multiplies by hand."""
    if len(mat1[0]) != len(mat2):
        return None
    new_mat = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat2[0])):
            total = 0
            for k in range(len(mat2)):
                val1 = mat1[i][k]
                val2 = mat2[k][j]
                product = val1 * val2
                total = total + product
            row.append(total)
        new_mat.append(row)
    return new_mat
