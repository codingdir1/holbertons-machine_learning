#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# YOUR CODE HERE
#df.drop(columns = ["Weighted_Price"], inplace = True)
df.loc[:, "Close"].ffill(inplace = True)
df.loc[:, "High"].fillna(df.loc[:, "Close"], inplace = True)
df.loc[:, "Low"].fillna(df.loc[:, "Close"], inplace = True)
df.loc[:, "Open"].fillna(df.loc[:, "Close"], inplace = True)
df.loc[:, "Volume_(BTC)"].fillna(0, inplace = True)
df.loc[:, "Volume_(Currency)"].fillna(0, inplace = True)


print(df.head())
print(df.tail())
