import glob
import os
import pandas as pd
folder = "data/historical/weather"
files = glob.glob(os.path.join(folder, "*.csv"))
dfs = []
for file in sorted(files):
    print(f"Reading {os.path.basename(file)}")
    dfs.append(pd.read_csv(file))
weather = pd.concat(dfs, ignore_index=True)
weather.drop_duplicates(inplace=True)
weather["datetime_local"] = pd.to_datetime(weather["datetime_local"])
weather.sort_values("datetime_local", inplace=True)
weather.reset_index(drop=True, inplace=True)
weather.to_csv(
    "data/historical/karachi_weather_all.csv",
    index=False
)
print(weather.head())
print(f"\nTotal Rows: {len(weather)}")