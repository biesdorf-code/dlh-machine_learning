#!/usr/bin/env python3
"""sorts data in reverse chronological order and transposes it """


def flip_switch(df):
    """sorts the pd.DataFrame in reverse chronological order
    and transposes the sorted dataframe

    Returns: the transformed pd.DataFrame
    """
    df = df.sort_values(by="Timestamp", ascending=False)
    df = df.transpose()
    return df
