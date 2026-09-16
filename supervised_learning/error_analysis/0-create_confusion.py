#!/usr/bin/env python3
""" creates a confusion matrix from one-hot labels and predictions """
import numpy as np


def create_confusion_matrix(labels, logits):
    """ creates a confusion matrix from one-hot labels and predictions """
    return np.matmul(labels.T, logits)
