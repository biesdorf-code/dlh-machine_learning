#!/usr/bin/env python3
""" normalizes (standardizes) a matrix """


def normalize(X, m, s):
    """ normalizes (standardizes) a matrix """
    return (X - m) / s
