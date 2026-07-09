#!/usr/bin/env python3
"""renames the Timestamp column to Datetime and converts it
"""
import pandas as pd


def rename(df):
    """renames Timestamp to Datetime, converts to datetime values,
    and keeps only Datetime and Close

    Returns: the modified pd.DataFrame
    """
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit="s")
    df = df[["Datetime", "Close"]]
    return df
