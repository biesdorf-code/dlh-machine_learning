#!/usr/bin/env python3
''' organizes the data, and plots it. 
'''
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# remove, rename, change shape
df = df.drop(columns=["Weighted_Price"])
df = df.rename(columns={"Timestamp": "Date"})
df["Date"] = pd.to_datetime(df["Date"], unit="s")
df = df.set_index("Date")

# data quality upgrade
df["Close"] = df["Close"].ffill()
df["High"] = df["High"].fillna(df["Close"])
df["Low"] = df["Low"].fillna(df["Close"])
df["Open"] = df["Open"].fillna(df["Close"])
df["Volume_(BTC)"] = df["Volume_(BTC)"].fillna(0)
df["Volume_(Currency)"] = df["Volume_(Currency)"].fillna(0)

df = df[df.index >= '2017-01-01']  # keep data after 1st Jan 2017

# downsampling from minute to days,
# The agg()regates the columns accordign to stat. funnctions
daily = df.resample("D").agg({
    "High": "max",
    "Low": "min",
    "Open": "mean",
    "Close": "mean",
    "Volume_(BTC)": "sum",
    "Volume_(Currency)": "sum"
})

print(daily)  # yes, just to the standartd output

daily.plot()  # prep the plot
plt.show()  # show the plot
