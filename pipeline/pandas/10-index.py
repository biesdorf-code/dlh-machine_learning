#!/usr/bin/env python3
"""sets the Timestamp column as the index of a pd.DataFrame
"""


def index(df):
    """sets Timestamp as the index

    Returns: the modified pd.DataFrame
    """
    df = df.set_index("Timestamp")
    return df
