#!/usr/bin/env python3
"""odule for adding two arrays"""


def add_arrays(arr1, arr2):
    """adding the elements of arrays"""
    if len(arr1) != len(arr2):
        return None
    new_arr = []
    for i in range(len(arr1)):
        new_arr.append(arr1[i] + arr2[i])
    return new_arr
