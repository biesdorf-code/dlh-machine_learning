#!/usr/bin/env python3
"""selects specific columns and every 60th row """
import pandas as pd


def slice(df):
    """extracts High, Low, Close, Volume_(BTC) and selects
    every 60th row

    Returns: the sliced pd.DataFrame
    """
    df = df[["High", "Low", "Close", "Volume_(BTC)"]]
    df = df[::60]  # slice
    return df
