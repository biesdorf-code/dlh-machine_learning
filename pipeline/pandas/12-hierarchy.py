#!/usr/bin/env python3
"""concatenates coinbase and bitstamp with Timestamp as the
first index level
"""
import pandas as pd
index = __import__('10-index').index


def hierarchy(df1, df2):
    """indexes both dataframes on Timestamp
    Returns: the concatenated pd.DataFrame
    """
    df1 = index(df1)
    df2 = index(df2)

    df1 = df1[(df1.index >= 1417411980) & (df1.index <= 1417417980)]
    df2 = df2[(df2.index >= 1417411980) & (df2.index <= 1417417980)]

    df = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
    df = df.reorder_levels([1, 0])
    df = df.sort_index(level=0)
    return df
