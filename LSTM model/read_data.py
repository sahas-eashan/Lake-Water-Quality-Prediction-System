import pandas as pd
import glob
import re

files = sorted(glob.glob(r"C:\Users\Cyborg\Downloads\Lake\Lake\lake_*.csv"))
dfs = []
for file in files:
    match = re.search(r"lake_(\d{4})_(\d{1,2})\.csv", file)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
    else:
        continue

    df = pd.read_csv(file)
    df = df[["ID", "pH", "Cond", "Tur"]]
    df["year"] = year
    df["month"] = month
    dfs.append(df)

data_all = pd.concat(dfs, ignore_index=True)

data_all = data_all.rename(columns={"ID": "location_id"})

print(data_all.head())
print("Number of rows:", len(data_all))
print("Shape of data_all:", data_all.shape)

data_all.to_csv("all_locations_timeseries.csv", index=False)
