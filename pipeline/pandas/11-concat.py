#!/usr/bin/env python3
"""concatenates coinbase and bitstamp dataframes indexed by Timestamp """

index = __import__('10-index').index


def concat(df1, df2):
    """indexes both dataframes on Timestamp"""

    df1 = index(df1)
    df2 = index(df2)
    df2 = df2[df2.index <= 1417411920]
    df = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
    return df
