import glob
import os
import pandas as pd
DATA_FOLDER = "data/historical/pollutants"
csv_files = glob.glob(os.path.join(DATA_FOLDER, "*.csv"))
print(f"Found {len(csv_files)} CSV files.")
dfs = []
for file in sorted(csv_files):
    print(f"Reading {os.path.basename(file)}")
    df = pd.read_csv(file)
    dfs.append(df)
merged_df = pd.concat(dfs, ignore_index=True)
merged_df.drop_duplicates(inplace=True)
merged_df["datetime_utc"] = pd.to_datetime(merged_df["datetime_utc"])
merged_df.sort_values(by="datetime_utc", inplace=True)
merged_df.reset_index(drop=True, inplace=True)
output_path = "data/historical/karachi_pm25_all.csv"
merged_df.to_csv(output_path, index=False)
print("\n===================================")
print(f"Total Rows : {len(merged_df)}")
print(f"Saved To   : {output_path}")
print("===================================")