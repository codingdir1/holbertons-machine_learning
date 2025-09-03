#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

df = df.drop(columns = ["Weighted_Price"])
df = df.rename(columns = {"Timestamp" : "Date"})
df.loc[:, "Date"] = pd.to_datetime(df.loc[:, "Date"], unit = "s")
df.set_index("Date", inplace = True)
df.loc[:, "Close"].ffill(inplace = True)
df.loc[:, "High"].fillna(df.loc[:, "Close"], inplace = True)
df.loc[:, "Low"].fillna(df.loc[:, "Close"], inplace = True)
df.loc[:, "Open"].fillna(df.loc[:, "Close"], inplace = True)
df.loc[:, "Volume_(BTC)"].fillna(value = 0, inplace = True)
df.loc[:, "Volume_(Currency)"].fillna(value = 0, inplace = True)

print(df)
df.plot()
plt.show()
