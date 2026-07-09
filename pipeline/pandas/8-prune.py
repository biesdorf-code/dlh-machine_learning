#!/usr/bin/env python3
"""removes rows where Close has NaN values
"""


def prune(df):
    """removes any entries where Close is NaN

    Returns: the modified pd.DataFrame
    """
    df = df.dropna(subset=["Close"])
    return df
