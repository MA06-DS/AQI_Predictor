
# ============================================================
# AQI FEATURE ENGINEERING
# ============================================================
#
# INPUT:
#   data/2026/karachi_2026_combined.csv
#
# OUTPUT:
#   data/2026/training_dataset_features.csv
#
# FEATURES:
#
# datetime_local
# temperature
# humidity
# pressure
# wind_speed
# wind_direction
# precipitation
# cloud_cover
# datetime_utc
# pm25
# aqi
#
# hour
# day
# month
# year
# day_of_week
# is_weekend
#
# aqi_lag_1
# aqi_lag_3
# aqi_lag_6
# aqi_lag_12
# aqi_lag_24
#
# aqi_mean_6
# aqi_std_6
# aqi_mean_12
# aqi_std_12
# aqi_mean_24
# aqi_std_24
#
# target_aqi_1 ... target_aqi_72
#
# ============================================================


import pandas as pd
import numpy as np
from pathlib import Path


# ============================================================
# 1. FILE PATHS
# ============================================================

INPUT_FILE = Path(
    "data/2026/karachi_2026_combined.csv"
)

OUTPUT_FILE = Path(
    "data/2026/training_dataset_features.csv"
)


# ============================================================
# 2. AQI FUNCTION
# ============================================================

def pm25_to_aqi(pm25):

    if pd.isna(pm25):
        return np.nan

    try:
        pm25 = float(pm25)

    except (ValueError, TypeError):
        return np.nan

    if pm25 < 0:
        return np.nan

    # Truncate to one decimal place
    pm25 = np.floor(pm25 * 10) / 10

    breakpoints = [

        (0.0, 12.0, 0, 50),

        (12.1, 35.4, 51, 100),

        (35.5, 55.4, 101, 150),

        (55.5, 150.4, 151, 200),

        (150.5, 250.4, 201, 300),

        (250.5, 350.4, 301, 400),

        (350.5, 500.4, 401, 500),

    ]

    for (

        c_low,
        c_high,
        i_low,
        i_high

    ) in breakpoints:

        if c_low <= pm25 <= c_high:

            aqi = (

                (
                    i_high - i_low
                )
                /
                (
                    c_high - c_low
                )
                *
                (
                    pm25 - c_low
                )
                +
                i_low

            )

            return round(aqi)

    if pm25 > 500.4:
        return 500

    return np.nan


# ============================================================
# 3. START
# ============================================================

print()
print("=" * 70)
print("AQI FEATURE ENGINEERING")
print("=" * 70)


# ============================================================
# 4. CHECK INPUT
# ============================================================

if not INPUT_FILE.exists():

    raise FileNotFoundError(

        f"\nInput file not found:\n"
        f"{INPUT_FILE}"

    )


# ============================================================
# 5. LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)

print()
print(
    "Input dataset shape:",
    df.shape
)


# ============================================================
# 6. REQUIRED COLUMNS
# ============================================================

required_columns = [

    "datetime_local",
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_direction",
    "precipitation",
    "cloud_cover",
    "datetime_utc",
    "pm25",

]


missing = [

    col

    for col in required_columns

    if col not in df.columns

]


if missing:

    raise ValueError(

        "\nMissing columns:\n"
        + "\n".join(missing)

    )


print()
print("All required columns found.")


# ============================================================
# 7. DATETIME
# ============================================================

print()
print("=" * 70)
print("PROCESSING DATETIME")
print("=" * 70)


df["datetime_local"] = pd.to_datetime(
    df["datetime_local"],
    errors="coerce"
)


df["datetime_utc"] = pd.to_datetime(
    df["datetime_utc"],
    errors="coerce"
)


# Remove invalid timestamps

df = df[
    df["datetime_local"].notna()
].copy()


# Sort

df = (

    df

    .sort_values(
        "datetime_local"
    )

    .reset_index(
        drop=True
    )

)


# ============================================================
# 8. DUPLICATES
# ============================================================

duplicates = (

    df["datetime_local"]
    .duplicated()
    .sum()

)


print(
    "Duplicate timestamps:",
    duplicates
)


if duplicates > 0:

    df = (

        df

        .drop_duplicates(

            subset=[
                "datetime_local"
            ],

            keep="first"

        )

        .reset_index(
            drop=True
        )

    )


# ============================================================
# 9. CONVERT NUMERIC COLUMNS
# ============================================================

numeric_columns = [

    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_direction",
    "precipitation",
    "cloud_cover",
    "pm25",

]


for col in numeric_columns:

    df[col] = pd.to_numeric(

        df[col],

        errors="coerce"

    )


# ============================================================
# 10. AQI
# ============================================================

print()
print("=" * 70)
print("CALCULATING AQI")
print("=" * 70)


df["aqi"] = (

    df["pm25"]

    .apply(
        pm25_to_aqi
    )

)


print(
    "PM2.5 missing:",
    df["pm25"].isna().sum()
)


print(
    "AQI missing:",
    df["aqi"].isna().sum()
)


# ============================================================
# 11. TIME FEATURES
# ============================================================

print()
print("=" * 70)
print("TIME FEATURES")
print("=" * 70)


df["hour"] = (
    df["datetime_local"].dt.hour
)

df["day"] = (
    df["datetime_local"].dt.day
)

df["month"] = (
    df["datetime_local"].dt.month
)

df["year"] = (
    df["datetime_local"].dt.year
)

df["day_of_week"] = (
    df["datetime_local"].dt.dayofweek
)

df["is_weekend"] = (

    df["day_of_week"] >= 5

).astype(int)


print("Time features created.")


# ============================================================
# 12. LAG FEATURES
# ============================================================

print()
print("=" * 70)
print("AQI LAG FEATURES")
print("=" * 70)


lags = [
    1,
    3,
    6,
    12,
    24
]


for lag in lags:

    name = f"aqi_lag_{lag}"

    df[name] = (
        df["aqi"].shift(lag)
    )

    print(
        "Created:",
        name
    )


# ============================================================
# 13. ROLLING FEATURES
# ============================================================

print()
print("=" * 70)
print("AQI ROLLING FEATURES")
print("=" * 70)


for window in [6, 12, 24]:

    mean_name = (
        f"aqi_mean_{window}"
    )

    std_name = (
        f"aqi_std_{window}"
    )


    # IMPORTANT:
    # shift(1) prevents current AQI
    # from entering the feature.

    previous_aqi = (
        df["aqi"].shift(1)
    )


    df[mean_name] = (

        previous_aqi

        .rolling(
            window=window
        )

        .mean()

    )


    df[std_name] = (

        previous_aqi

        .rolling(
            window=window
        )

        .std()

    )


    print(
        "Created:",
        mean_name
    )

    print(
        "Created:",
        std_name
    )


# ============================================================
# 14. FUTURE TARGETS
# ============================================================

print()
print("=" * 70)
print("CREATING FUTURE AQI TARGETS")
print("=" * 70)


target_columns = []


for hours in range(1, 73):

    name = (
        f"target_aqi_{hours}"
    )

    df[name] = (

        df["aqi"]

        .shift(
            -hours
        )

    )

    target_columns.append(
        name
    )


print(
    "Created 72 target columns."
)


# ============================================================
# 15. TARGET AVAILABILITY
# ============================================================

print()
print("=" * 70)
print("TARGET AVAILABILITY")
print("=" * 70)


for hours in [

    1,
    6,
    12,
    24,
    48,
    72

]:

    name = (
        f"target_aqi_{hours}"
    )

    count = (

        df[name]
        .notna()
        .sum()

    )

    print(
        f"{name}: {count} rows"
    )


# ============================================================
# 16. HISTORICAL FEATURE CLEANING
# ============================================================

history_columns = [

    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_12",
    "aqi_lag_24",

    "aqi_mean_6",
    "aqi_std_6",

    "aqi_mean_12",
    "aqi_std_12",

    "aqi_mean_24",
    "aqi_std_24",

]


print()
print("=" * 70)
print("CLEANING HISTORY FEATURES")
print("=" * 70)


before = len(df)


df = df.dropna(
    subset=history_columns
)


print(
    "Rows before:",
    before
)


print(
    "Rows after:",
    len(df)
)


print(
    "Removed:",
    before - len(df)
)


# ============================================================
# 17. CURRENT AQI CLEANING
# ============================================================

before = len(df)


df = df[
    df["aqi"].notna()
].copy()


print()
print(
    "Rows removed because current AQI was missing:",
    before - len(df)
)


# ============================================================
# 18. FUTURE TARGET CLEANING
# ============================================================
#
# IMPORTANT:
#
# We require all 72 targets only if we want to train
# a single multi-output 72-hour model.
#
# ============================================================

before = len(df)


df = df.dropna(
    subset=target_columns
)


print()
print(
    "Rows before 72-hour target cleaning:",
    before
)


print(
    "Rows after 72-hour target cleaning:",
    len(df)
)


print(
    "Rows removed:",
    before - len(df)
)


# ============================================================
# 19. EMPTY DATASET CHECK
# ============================================================

if df.empty:

    print()
    print("=" * 70)
    print("NO TRAINING ROWS AVAILABLE")
    print("=" * 70)

    print()
    print(
        "The dataset does not contain enough "
        "continuous AQI observations to create "
        "24-hour historical features AND "
        "72-hour future targets."
    )

    print()
    print(
        "This is NOT a Python indexing error."
    )

    print()
    print(
        "You currently have:",
        len(pd.read_csv(INPUT_FILE)),
        "raw rows."
    )

    print()
    print(
        "For 72-hour forecasting, the data needs "
        "sufficient future AQI observations."
    )

    print()
    print(
        "The script will NOT create an empty CSV."
    )

    raise RuntimeError(

        "\nNo usable training rows were produced. "
        "Collect/fetch more continuous historical "
        "AQI data before creating 72-hour targets."

    )


# ============================================================
# 20. FINAL COLUMN ORDER
# ============================================================

base_columns = [

    "datetime_local",

    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_direction",
    "precipitation",
    "cloud_cover",

    "datetime_utc",

    "pm25",

    "aqi",

]


time_columns = [

    "hour",
    "day",
    "month",
    "year",
    "day_of_week",
    "is_weekend",

]


lag_columns = [

    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_12",
    "aqi_lag_24",

]


rolling_columns = [

    "aqi_mean_6",
    "aqi_std_6",

    "aqi_mean_12",
    "aqi_std_12",

    "aqi_mean_24",
    "aqi_std_24",

]


final_columns = (

    base_columns

    +

    time_columns

    +

    lag_columns

    +

    rolling_columns

    +

    target_columns

)


df = df[
    final_columns
]


# ============================================================
# 21. RESET INDEX
# ============================================================

df = (
    df.reset_index(
        drop=True
    )
)


# ============================================================
# 22. FINAL CHECK
# ============================================================

print()
print("=" * 70)
print("FINAL DATASET")
print("=" * 70)


print()
print(
    "Rows:",
    len(df)
)


print(
    "Columns:",
    len(df.columns)
)


print()
print(
    "Missing values:"
)


missing_values = (
    df.isna().sum()
)


missing_values = (
    missing_values[
        missing_values > 0
    ]
)


if missing_values.empty:

    print(
        "No missing values."
    )

else:

    print(
        missing_values
    )


print()
print(
    "Duplicate timestamps:",
    df[
        "datetime_local"
    ].duplicated().sum()
)


# ============================================================
# 23. SAVE
# ============================================================

OUTPUT_FILE.parent.mkdir(

    parents=True,

    exist_ok=True

)


df.to_csv(

    OUTPUT_FILE,

    index=False

)


print()
print("=" * 70)
print("DATASET SAVED")
print("=" * 70)


print()
print(
    "File:",
    OUTPUT_FILE
)


print(
    "Rows:",
    len(df)
)


print(
    "Columns:",
    len(df.columns)
)


# ============================================================
# 24. SAFE DATE DISPLAY
# ============================================================

if len(df) > 0:

    print()
    print(
        "First timestamp:",
        df[
            "datetime_local"
        ].iloc[0]
    )

    print(
        "Last timestamp:",
        df[
            "datetime_local"
        ].iloc[-1]
    )


# ============================================================
# 25. FINAL COLUMN LIST
# ============================================================

print()
print("=" * 70)
print("FINAL COLUMNS")
print("=" * 70)


for i, column in enumerate(

    df.columns,

    start=1

):

    print(
        f"{i:3d}. {column}"
    )


# ============================================================
# 26. COMPLETE
# ============================================================

print()
print("=" * 70)
print("FEATURE ENGINEERING COMPLETED")
print("=" * 70)

print()
print("SUCCESS")

print(
    "=" * 70
)
