import pandas as pd

print("Before cleaning:")
print("Number of rows:", len(data_all))
print("Missing values:\n", data_all.isnull().sum())
print(data_all.describe())

data_all = data_all.dropna()

data_all = data_all[(data_all["pH"] >= 0) & (data_all["pH"] <= 14)]

data_all = data_all[data_all["Cond"] >= 0]
data_all = data_all[data_all["Tur"] >= 0]

print("\nAfter cleaning:")
print("Number of rows:", len(data_all))
print("Missing values:\n", data_all.isnull().sum())
print(data_all.describe())

data_all.to_csv("all_locations_timeseries_cleaned.csv", index=False)
