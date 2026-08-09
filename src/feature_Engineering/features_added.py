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

import pandas as pd

# ==========================================
# 1. Load cleaned AQI data
# ==========================================
df = pd.read_csv(
    "data/processed/training_dataset_aqi_clean.csv"
)

# Convert datetime columns
df["datetime_local"] = pd.to_datetime(df["datetime_local"])
df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])

# Sort chronologically
df = df.sort_values("datetime_local").reset_index(drop=True)

# ==========================================
# 2. Time Features
# ==========================================
df["hour"] = df["datetime_local"].dt.hour
df["day"] = df["datetime_local"].dt.day
df["month"] = df["datetime_local"].dt.month
df["year"] = df["datetime_local"].dt.year
df["day_of_week"] = df["datetime_local"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)

# ==========================================
# 3. Lag Features
# ==========================================
lags = [1, 3, 6, 12, 24]

for lag in lags:
    df[f"aqi_lag_{lag}"] = df["aqi"].shift(lag)

# ==========================================
# 4. Rolling Features
# ==========================================
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

# ==========================================
# 5. Create 72-Hour Future Targets
# ==========================================

# Create timestamp → AQI lookup
aqi_lookup = (
    df.set_index("datetime_local")["aqi"]
)

for hour_ahead in range(1, 73):

    future_time = (
        df["datetime_local"]
        + pd.Timedelta(hours=hour_ahead)
    )

    df[f"target_aqi_{hour_ahead}"] = (
        future_time.map(aqi_lookup)
    )

# ==========================================
# 6. Remove rows with missing values
# ==========================================
print("Before Cleaning:", df.shape)

df = df.dropna().reset_index(drop=True)

print("After Cleaning:", df.shape)

# ==========================================
# 7. Save Feature Dataset
# ==========================================
output_path = (
    "data/processed/training_dataset_features.csv"
)

df.to_csv(
    output_path,
    index=False
)

# ==========================================
# 8. Display Results
# ==========================================
print("\nFeature Engineering Completed Successfully!")

print("\nDataset Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 5 Rows:")
print(df.head())

print("\nNumber of Target Columns:")
print(
    len([
        col for col in df.columns
        if col.startswith("target_aqi_")
    ])
)