import pandas as pd
import time
# Read datasets
pm25 = pd.read_csv("data/historical/karachi_pm25_all.csv")
weather = pd.read_csv("data/historical/karachi_weather_all.csv")
pm25["datetime_local"] = pd.to_datetime(
    pm25["datetime_local"],
    format="mixed"
)

weather["datetime_local"] = pd.to_datetime(
    weather["datetime_local"],
    format="mixed"
)
pm25["datetime_local"] = pm25["datetime_local"].dt.tz_localize(None)
# Convert to datetime
pm25["datetime_local"] = pd.to_datetime(pm25["datetime_local"])
weather["datetime_local"] = pd.to_datetime(weather["datetime_local"])

merged = pd.merge_asof(
    weather.sort_values("datetime_local"),
    pm25.sort_values("datetime_local"),
    on="datetime_local",
    direction="nearest",
    tolerance=pd.Timedelta("30min")
)

# Sort
merged = merged.sort_values("datetime_local")

# Reset index
merged.reset_index(drop=True, inplace=True)

print(merged.head())

print("\nMissing PM2.5 values:")
print(merged["pm25"].isnull().sum())

merged.to_csv(
    "data/processed/training_dataset.csv",
    index=False
)

print("\nDataset Saved!")
print("Rows:", len(merged))