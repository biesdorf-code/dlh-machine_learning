#!/usr/bin/env python3
"""computes descriptive statistics for all columns except Timestamp"""


def analyze(df):
    """computes descriptive statistics
    returns: a new pd.DataFrame containing these statistics
    """
    df = df.drop(columns=["Timestamp"])
    stats = df.describe()
    return stats
