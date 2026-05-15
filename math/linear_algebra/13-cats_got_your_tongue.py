#!/usr/bin/env python3
"""module to cat two numpy.ndarrays"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """Concats two tuple matrices along 0=rows, 1=columns"""
    return np.concatenate((mat1, mat2), axis=axis)
