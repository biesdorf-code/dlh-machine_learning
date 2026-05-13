#!/usr/bin/env python3
"""module for adding two 2D matrices"""


def add_matrices2D(mat1, mat2):
    """adds 2D matrices, same shapes is a must"""
    if len(mat1) != len(mat2):
        return None
    if len(mat1[0]) != len(mat2[0]):
        return None
    new_mat = []
    for i in range(len(mat1)):
        row = []
        for j in range(len(mat1[0])):
            row.append(mat1[i][j] + mat2[i][j])
        new_mat.append(row)
    return new_mat
