import pandas as pd
df = pd.read_csv("data/processed/training_dataset_aqi_clean.csv")

df["datetime_local"] = pd.to_datetime(df["datetime_local"])
df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])

df = df.sort_values("datetime_local").reset_index(drop=True)

df["hour"] = df["datetime_local"].dt.hour
df["day"] = df["datetime_local"].dt.day
df["month"] = df["datetime_local"].dt.month
df["year"] = df["datetime_local"].dt.year
df["day_of_week"] = df["datetime_local"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
lags = [1, 3, 6, 12, 24]
for lag in lags:
    df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)
windows = [6, 12, 24]
for window in windows:
    df[f"aqi_mean_{window}"] = (
        df["aqi"]
        .rolling(window=window)
        .mean()
    )
    df[f"aqi_std_{window}"] = (
        df["aqi"]
        .rolling(window=window)
        .std()
    )
# Predict AQI 72 hours (3 days) ahead
df["target_aqi"] = df["aqi"].shift(-72)
print("Before Cleaning:", df.shape)
df = df.dropna().reset_index(drop=True)
print("After Cleaning :", df.shape)
df.to_csv(
    "data/processed/training_dataset_features.csv",
    index=False
)
print("\nFeature Engineering Completed Successfully!")
print("\nDataset Shape:", df.shape)
print("\nColumns:")
print(df.columns.tolist())
print("\nFirst 5 Rows:")
print(df.head())