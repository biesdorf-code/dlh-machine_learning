#!/usr/bin/env python3
""" calculates the F1 score for each class in a confusion matrix """
import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """ calculates the F1 score for each class in a confusion matrix """
    s = sensitivity(confusion)
    p = precision(confusion)
    return 2 * p * s / (p + s)
