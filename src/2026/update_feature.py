
# ============================================================
# FEATURE ENGINEERING FOR AQI PREDICTION
# ============================================================
#
# INPUT:
#   data/processed/karachi_2026_combined.csv
#
# This dataset should already contain:
#
#   datetime_local
#   temperature
#   humidity
#   pressure
#   wind_speed
#   wind_direction
#   precipitation
#   cloud_cover
#   datetime_utc
#   pm25
#
#
# THIS SCRIPT CREATES:
#
#   aqi
#
#   hour
#   day
#   month
#   year
#   day_of_week
#   is_weekend
#
#   aqi_lag_1
#   aqi_lag_3
#   aqi_lag_6
#   aqi_lag_12
#   aqi_lag_24
#
#   aqi_mean_6
#   aqi_std_6
#   aqi_mean_12
#   aqi_std_12
#   aqi_mean_24
#   aqi_std_24
#
#   target_aqi_1
#   target_aqi_2
#   ...
#   target_aqi_72
#
#
# OUTPUT:
#   data/processed/training_dataset_features.csv
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
# 2. AQI CALCULATION
# ============================================================

def pm25_to_aqi(pm25):

    if pd.isna(pm25):
        return np.nan

    try:
        pm25 = float(pm25)

    except (ValueError, TypeError):
        return np.nan

    # Invalid concentration
    if pm25 < 0:
        return np.nan

    # EPA PM2.5 concentration is truncated
    # to one decimal place
    pm25 = np.floor(pm25 * 10) / 10

    breakpoints = [

        # PM2.5 concentration
        # AQI

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
                    (i_high - i_low)
                    /
                    (c_high - c_low)
                )
                *
                (pm25 - c_low)
                +
                i_low

            )

            return round(aqi)

    # Above highest breakpoint
    if pm25 > 500.4:
        return 500

    return np.nan


# ============================================================
# 3. CHECK INPUT FILE
# ============================================================

print()
print("=" * 70)
print("AQI FEATURE ENGINEERING")
print("=" * 70)

print()
print("Input file:")
print(INPUT_FILE)


if not INPUT_FILE.exists():

    raise FileNotFoundError(
        f"\nInput file not found:\n{INPUT_FILE}"
    )


# ============================================================
# 4. LOAD DATA
# ============================================================

df = pd.read_csv(
    INPUT_FILE
)

print()
print("Original dataset shape:")
print(df.shape)


# ============================================================
# 5. REQUIRED COLUMNS
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


missing_columns = [

    column

    for column in required_columns

    if column not in df.columns

]


if missing_columns:

    raise ValueError(

        "\nMissing required columns:\n"

        + "\n".join(
            missing_columns
        )

    )


print()
print("All required columns found.")


# ============================================================
# 6. DATETIME CONVERSION
# ============================================================

print()
print("=" * 70)
print("DATETIME PROCESSING")
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


# Sort chronologically

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
# 7. REMOVE DUPLICATE TIMESTAMPS
# ============================================================

duplicate_count = (

    df["datetime_local"]
    .duplicated()
    .sum()

)


print(
    "Duplicate timestamps:",
    duplicate_count
)


if duplicate_count > 0:

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
# 8. NUMERIC CONVERSION
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


for column in numeric_columns:

    df[column] = pd.to_numeric(

        df[column],

        errors="coerce"

    )


# ============================================================
# 9. CALCULATE AQI
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
    "Missing AQI values:",
    df["aqi"].isna().sum()
)


# ============================================================
# 10. TIME FEATURES
# ============================================================

print()
print("=" * 70)
print("CREATING TIME FEATURES")
print("=" * 70)


df["hour"] = (

    df["datetime_local"]
    .dt.hour

)


df["day"] = (

    df["datetime_local"]
    .dt.day

)


df["month"] = (

    df["datetime_local"]
    .dt.month

)


df["year"] = (

    df["datetime_local"]
    .dt.year

)


df["day_of_week"] = (

    df["datetime_local"]
    .dt.dayofweek

)


df["is_weekend"] = (

    df["day_of_week"]
    >= 5

).astype(int)


print(
    "Time features created."
)


# ============================================================
# 11. AQI LAG FEATURES
# ============================================================

print()
print("=" * 70)
print("CREATING AQI LAG FEATURES")
print("=" * 70)


lags = [

    1,
    3,
    6,
    12,
    24,

]


for lag in lags:

    column_name = (
        f"aqi_lag_{lag}"
    )

    df[column_name] = (

        df["aqi"]

        .shift(
            lag
        )

    )

    print(
        f"Created: {column_name}"
    )


# ============================================================
# 12. ROLLING AQI FEATURES
# ============================================================
#
# We use shift(1) so the current AQI is NOT included
# in its own rolling statistics.
#
# This prevents data leakage.
#
# ============================================================

print()
print("=" * 70)
print("CREATING ROLLING AQI FEATURES")
print("=" * 70)


windows = [

    6,
    12,
    24,

]


for window in windows:

    mean_column = (
        f"aqi_mean_{window}"
    )

    std_column = (
        f"aqi_std_{window}"
    )


    df[mean_column] = (

        df["aqi"]

        .shift(1)

        .rolling(
            window=window
        )

        .mean()

    )


    df[std_column] = (

        df["aqi"]

        .shift(1)

        .rolling(
            window=window
        )

        .std()

    )


    print(
        f"Created: {mean_column}"
    )

    print(
        f"Created: {std_column}"
    )


# ============================================================
# 13. CREATE EXACT FUTURE AQI TARGETS
# ============================================================
#
# target_aqi_1
#   AQI exactly 1 hour later
#
# target_aqi_2
#   AQI exactly 2 hours later
#
# ...
#
# target_aqi_72
#   AQI exactly 72 hours later
#
#
# We use datetime lookup instead of shift(-72).
#
# This is important because if one hourly observation
# is missing, shift(-72) would NOT necessarily represent
# exactly 72 hours in the future.
#
# ============================================================

print()
print("=" * 70)
print("CREATING 72-HOUR FUTURE AQI TARGETS")
print("=" * 70)


# Create datetime -> AQI lookup

aqi_lookup = (

    df

    .set_index(
        "datetime_local"
    )["aqi"]

)


for hour_ahead in range(
    1,
    73
):

    target_column = (
        f"target_aqi_{hour_ahead}"
    )


    future_datetime = (

        df["datetime_local"]

        +

        pd.Timedelta(
            hours=hour_ahead
        )

    )


    df[target_column] = (

        future_datetime

        .map(
            aqi_lookup
        )

    )


    print(
        f"Created: {target_column}"
    )


# ============================================================
# 14. REMOVE ROWS WITHOUT CURRENT AQI
# ============================================================

before = len(df)


df = df[
    df["aqi"].notna()
].copy()


print()
print(
    "Rows removed because AQI was missing:",
    before - len(df)
)


# ============================================================
# 15. REMOVE ROWS WITH INCOMPLETE HISTORICAL FEATURES
# ============================================================
#
# The largest historical window is 24 hours.
#
# Therefore the first 24 observations cannot have
# complete lag/rolling features.
#
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


before = len(df)


df = df.dropna(

    subset=history_columns

).copy()


print(
    "Rows removed due to insufficient history:",
    before - len(df)
)


# ============================================================
# 16. REMOVE ROWS WITHOUT ALL 72 TARGETS
# ============================================================

target_columns = [

    f"target_aqi_{i}"

    for i in range(
        1,
        73
    )

]


before = len(df)


df = df.dropna(

    subset=target_columns

).copy()


print(
    "Rows removed due to missing future targets:",
    before - len(df)
)


# ============================================================
# 17. FINAL COLUMN ORDER
# ============================================================

print()
print("=" * 70)
print("CREATING FINAL COLUMN ORDER")
print("=" * 70)


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
# 18. RESET INDEX
# ============================================================

df = (

    df

    .reset_index(
        drop=True
    )

)


# ============================================================
# 19. FINAL DATA QUALITY CHECK
# ============================================================

print()
print("=" * 70)
print("FINAL DATA QUALITY CHECK")
print("=" * 70)


print()
print(
    "Final dataset shape:",
    df.shape
)


print()
print(
    "Missing values:"
)


missing = (
    df.isna().sum()
)


missing = missing[
    missing > 0
]


if missing.empty:

    print(
        "No missing values."
    )

else:

    print(
        missing
    )


print()
print(
    "Duplicate timestamps:",
    df["datetime_local"]
    .duplicated()
    .sum()
)


# ============================================================
# 20. CHECK TARGET COLUMNS
# ============================================================

number_of_targets = len(

    [

        column

        for column in df.columns

        if column.startswith(
            "target_aqi_"
        )

    ]

)


print()
print(
    "Number of target columns:",
    number_of_targets
)


if number_of_targets != 72:

    raise RuntimeError(

        f"Expected 72 target columns "
        f"but found {number_of_targets}."

    )


# ============================================================
# 21. SAVE FINAL DATASET
# ============================================================

print()
print("=" * 70)
print("SAVING FINAL DATASET")
print("=" * 70)


df.to_csv(

    OUTPUT_FILE,

    index=False

)


print()
print(
    "Saved successfully:"
)

print(
    OUTPUT_FILE
)


# ============================================================
# 22. PRINT FINAL INFORMATION
# ============================================================

print()
print("=" * 70)
print("FINAL DATASET INFORMATION")
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
    "First timestamp:",
    df["datetime_local"].iloc[0]
)


print(
    "Last timestamp:",
    df["datetime_local"].iloc[-1]
)


print()
print(
    "Columns:"
)


for number, column in enumerate(
    df.columns,
    start=1
):

    print(
        f"{number:3d}. {column}"
    )


# ============================================================
# 23. DISPLAY SAMPLE
# ============================================================

print()
print("=" * 70)
print("FIRST 5 ROWS")
print("=" * 70)


print(
    df.head()
)


# ============================================================
# 24. COMPLETE
# ============================================================

print()
print("=" * 70)
print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
print("=" * 70)

print()
print(
    "Output file:"
)

print(
    OUTPUT_FILE
)

print()
print(
    "SUCCESS"
)

print(
    "=" * 70
)
