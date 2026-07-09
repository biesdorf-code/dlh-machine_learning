#!/usr/bin/env python3
"""selects the last 10 rows of High and Close as a numpy array
"""
import pandas as pd


def array(df):
    """selects the last 10 rows of the High and Close columns
    and converts them into a numpy.ndarray

    Returns: the numpy.ndarray
    """
    df = df[["High", "Close"]].tail(10)
    return df.to_numpy()
