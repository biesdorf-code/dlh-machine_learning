#!/usr/bin/env python3
"""sorts a pd.DataFrame by the High price in descending order
"""


def high(df):
    """sorts the pd.DataFrame by High price, descending

    Returns: the sorted pd.DataFrame
    """
    df = df.sort_values(by="High", ascending=False)
    return df
