#!/usr/bin/env python3
"""a function to create a pd.DataFrame from a
np.ndarray.
"""
import pandas as pd


def from_numpy(array):
    """creates a new a pd.DataFrame from a np.ndarray.

    Returns: DataFrame with columns labeled in alphabetical
    order and capitalized.
    """
    columns = [chr(65 + i) for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=columns)
