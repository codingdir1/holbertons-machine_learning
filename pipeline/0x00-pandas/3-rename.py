#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE

df = df.rename(columns = {"Timestamp" : "Datetime"})
df.loc[:, "Datetime"] = pd.to_datetime(df.loc[:, "Datetime"], unit = "s")
df = df.drop(["High", "Low", "Open", "Volume_(BTC)", "Volume_(Currency)", "Weighted_Price"], axis=1)


print(df.tail())
