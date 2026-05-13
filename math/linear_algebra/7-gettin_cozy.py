#!/usr/bin/env python3
"""module for concatenating two 2D matrices in 0 or 1 axis"""


def cat_matrices2D(mat1, mat2, axis=0):
    """concatenates two 2D matrices along 0 or 1"""
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        new_mat = []
        for row in mat1:
            new_mat.append(row[:])
        for row in mat2:
            new_mat.append(row[:])
        return new_mat
    if axis == 1:
        if len(mat1) != len(mat2):
            return None
        new_mat = []
        for i in range(len(mat1)):
            new_mat.append(mat1[i][:] + mat2[i][:])
        return new_mat
    return None
