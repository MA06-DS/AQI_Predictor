# ============================================================
# KARACHI AQI - HOURLY CI/CD DATA PIPELINE (UPSERT DESIGN)
# ============================================================
#
# PURPOSE
# -------
# Run this script once every hour from GitHub Actions / CI/CD.
#
# WHY THIS VERSION IS DIFFERENT
# ------------------------------
# The previous version only ever INSERTED brand-new rows.
# That meant every row's `target_aqi_1 ... target_aqi_72`
# columns were frozen at whatever was known at insert time.
# Since the AQI 72 hours in the future doesn't exist yet when
# a row is first written, those targets stayed NULL forever.
#
# This version instead:
#
#   1. Reads the last ~96-120 rows from Hopsworks as context.
#   2. Fetches the newly missing hour(s) from Open-Meteo/OpenAQ.
#   3. Combines history + new data and recalculates lag and
#      rolling features over that combined window.
#   4. Recalculates target_aqi_1..72 for the WHOLE combined
#      window using a plain row-based shift (the same way
#      lag/rolling features are computed). This naturally fills
#      in targets for older rows now that the AQI needed to
#      compute them has finally arrived.
#   5. target_aqi_1 should be NULL only for the single most
#      recent row (assuming no gaps in AQI coverage).
#   6. target_aqi_72 is naturally NULL for the most recent 72
#      rows, since that far into the future doesn't exist yet.
#   7. UPSERTS (not just inserts) the affected rows back into
#      Hopsworks, keyed on `datetime_local`, so older rows get
#      their targets patched in-place.
#   8. Net effect: every hourly run backfills targets for rows
#      up to 72 hours in the past, in addition to adding the
#      newest row.
#
# REQUIREMENT
# -----------
# For step 7 to actually behave as an upsert (rather than
# creating duplicate rows), the Hopsworks Feature Group must
# be Hudi-backed with `datetime_local` (or an equivalent
# timestamp column) declared as its primary key. This script
# checks for that and warns loudly if it isn't set up that way.
#
# ============================================================


# ============================================================
# 1. IMPORTS
# ============================================================

import os
import sys
import requests
import numpy as np
import pandas as pd
from datetime import datetime
import hopsworks
from dotenv import load_dotenv


# ============================================================
# 2. ENVIRONMENT
# ============================================================

load_dotenv()


# ============================================================
# 3. CONFIGURATION
# ============================================================

PROJECT_NAME = "anaskaaqi"

HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
OPENAQ_API_KEY = os.getenv("OPENAQ_API_KEY")

if not HOPSWORKS_API_KEY:
    raise RuntimeError("HOPSWORKS_API_KEY is not set.")

if not OPENAQ_API_KEY:
    raise RuntimeError("OPENAQ_API_KEY is not set.")


# ============================================================
# 4. KARACHI CONFIGURATION
# ============================================================

CITY = "Karachi"
LATITUDE = 24.8607
LONGITUDE = 67.0011
TIMEZONE = "Asia/Karachi"


# ============================================================
# 5. FIXED OPENAQ SOURCE
# ============================================================

OPENAQ_LOCATION_ID = 6135426
OPENAQ_LOCATION_NAME = "Aga Khan University Main Campus"
OPENAQ_SENSOR_ID = 14744851
OPENAQ_PARAMETER = "pm25"


# ============================================================
# 6. API ENDPOINTS
# ============================================================

OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPENAQ_URL = "https://api.openaq.org/v3"


# ============================================================
# 7. HOPSWORKS CONFIGURATION
# ============================================================

FEATURE_GROUP_NAME = "aqi_training_features"
FEATURE_GROUP_VERSION = 2
FEATURE_VIEW_NAME = "aqi_72_hour_forecast"
FEATURE_VIEW_VERSION = 2

# Primary key expected on the feature group so that fg.insert()
# behaves as an upsert instead of an append.
EXPECTED_PRIMARY_KEY = "datetime_local"


# ============================================================
# 8. HISTORY / UPSERT WINDOW CONFIGURATION
# ============================================================
#
# HISTORY_READ_ROWS
#   How many of the most recent Hopsworks rows we pull down as
#   context every run. Needs to comfortably cover:
#     - lag_24 / rolling_24  (needs 24 rows of pure lookback)
#     - a reasonable margin for gaps in the source data
#
# LOOKBACK_BUFFER_ROWS
#   The first N rows of that read history are used ONLY to give
#   the lag/rolling calculations something to look back on. They
#   are never themselves re-upserted, because we can't guarantee
#   *their* lag/rolling had a full 24-row lookback within our
#   read window (their true lookback lives further back in
#   Hopsworks than what we bothered to read).
#
# Everything from position LOOKBACK_BUFFER_ROWS onward (plus all
# newly fetched rows) is a genuine "upsert candidate": either a
# brand-new row, or an existing row whose target_aqi_* columns
# may have just become fillable.
#
# ============================================================

HISTORY_READ_ROWS = 120
LOOKBACK_BUFFER_ROWS = 24


# ============================================================
# 9. FEATURE COLUMNS
# ============================================================

FEATURE_COLUMNS = [
    "temperature", "humidity", "pressure", "wind_speed",
    "wind_direction", "precipitation", "cloud_cover",
    "pm25", "aqi",
    "hour", "day", "month", "year", "day_of_week", "is_weekend",
    "aqi_lag_1", "aqi_lag_3", "aqi_lag_6", "aqi_lag_12", "aqi_lag_24",
    "aqi_mean_6", "aqi_std_6",
    "aqi_mean_12", "aqi_std_12",
    "aqi_mean_24", "aqi_std_24",
]


# ============================================================
# 10. TARGET COLUMNS
# ============================================================

TARGET_COLUMNS = [f"target_aqi_{i}" for i in range(1, 73)]


# ============================================================
# 11. FINAL COLUMNS
# ============================================================

FINAL_COLUMNS = (
    ["datetime_local", "datetime_utc"] + FEATURE_COLUMNS + TARGET_COLUMNS
)


# ============================================================
# 12. HTTP SESSION
# ============================================================

SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "Karachi-AQI-Predictor/1.0"})


# ============================================================
# 13. TIMESTAMP HELPERS
# ============================================================

def normalize_local_timestamp(value):
    """Convert a timestamp into a timezone-naive Karachi local timestamp."""
    if value is None:
        return pd.NaT
    try:
        ts = pd.to_datetime(value, format="mixed", errors="coerce")
        if pd.isna(ts):
            return pd.NaT
        if getattr(ts, "tzinfo", None) is not None:
            ts = ts.tz_convert(TIMEZONE).tz_localize(None)
        return ts
    except Exception:
        try:
            ts = pd.to_datetime(value, utc=True, errors="coerce")
            if pd.isna(ts):
                return pd.NaT
            return ts.tz_convert(TIMEZONE).tz_localize(None)
        except Exception:
            return pd.NaT


def normalize_utc_timestamp(value):
    """Convert a timestamp into a timezone-naive UTC timestamp."""
    if value is None:
        return pd.NaT
    try:
        ts = pd.to_datetime(value, format="mixed", errors="coerce")
        if pd.isna(ts):
            return pd.NaT
        if getattr(ts, "tzinfo", None) is None:
            ts = ts.tz_localize(TIMEZONE)
        return ts.tz_convert("UTC").tz_localize(None)
    except Exception:
        return pd.NaT


def get_current_local_hour():
    now = pd.Timestamp.now(tz=TIMEZONE)
    return now.floor("h").tz_localize(None)


# ============================================================
# 14. FETCH OPEN-METEO
# ============================================================

def fetch_open_meteo(start_local, end_local):
    print("\n" + "=" * 70)
    print("FETCHING OPEN-METEO WEATHER")
    print("=" * 70)

    start_local = pd.Timestamp(start_local)
    end_local = pd.Timestamp(end_local)

    params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "start_date": start_local.strftime("%Y-%m-%d"),
        "end_date": end_local.strftime("%Y-%m-%d"),
        "hourly": ",".join([
            "temperature_2m", "relative_humidity_2m", "pressure_msl",
            "wind_speed_10m", "wind_direction_10m", "precipitation",
            "cloud_cover",
        ]),
        "timezone": TIMEZONE,
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "precipitation_unit": "mm",
    }

    response = SESSION.get(OPEN_METEO_URL, params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()

    if "hourly" not in payload:
        raise RuntimeError("Open-Meteo did not return hourly data.")

    hourly = payload["hourly"]

    weather = pd.DataFrame({
        "datetime_local": hourly["time"],
        "temperature": hourly["temperature_2m"],
        "humidity": hourly["relative_humidity_2m"],
        "pressure": hourly["pressure_msl"],
        "wind_speed": hourly["wind_speed_10m"],
        "wind_direction": hourly["wind_direction_10m"],
        "precipitation": hourly["precipitation"],
        "cloud_cover": hourly["cloud_cover"],
    })

    weather["datetime_local"] = pd.to_datetime(
        weather["datetime_local"], errors="coerce"
    )
    weather = weather[weather["datetime_local"].notna()]
    weather = weather[
        (weather["datetime_local"] >= start_local)
        & (weather["datetime_local"] <= end_local)
    ]

    weather["datetime_utc"] = (
        weather["datetime_local"]
        .dt.tz_localize(TIMEZONE)
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )

    print("Open-Meteo rows:", len(weather))
    return weather.reset_index(drop=True)


# ============================================================
# 15. FETCH OPENAQ
# ============================================================

def fetch_openaq_sensor(start_local, end_local):
    print("\n" + "=" * 70)
    print("FETCHING OPENAQ PM2.5")
    print("=" * 70)
    print("Location:", OPENAQ_LOCATION_ID, OPENAQ_LOCATION_NAME)
    print("Sensor:", OPENAQ_SENSOR_ID)
    print("Parameter:", OPENAQ_PARAMETER)

    start_utc = pd.Timestamp(start_local).tz_localize(TIMEZONE).tz_convert("UTC")
    end_utc = (
        pd.Timestamp(end_local).tz_localize(TIMEZONE).tz_convert("UTC")
        + pd.Timedelta(hours=1)
    )

    url = f"{OPENAQ_URL}/sensors/{OPENAQ_SENSOR_ID}/measurements/hourly"
    headers = {"X-API-Key": OPENAQ_API_KEY}
    params = {
        "datetime_from": start_utc.isoformat(),
        "datetime_to": end_utc.isoformat(),
        "limit": 1000,
    }

    response = SESSION.get(url, params=params, headers=headers, timeout=120)

    if response.status_code == 404:
        print("OpenAQ returned 404.")
        return pd.DataFrame(columns=["datetime_local", "pm25"])

    response.raise_for_status()
    payload = response.json()
    results = payload.get("results", [])
    print("Raw OpenAQ rows:", len(results))

    if not results:
        print("No OpenAQ measurements.")
        return pd.DataFrame(columns=["datetime_local", "pm25"])

    rows = []
    for row in results:
        value = row.get("value")
        if value is None:
            continue

        period = row.get("period", {})
        datetime_from = (
            period.get("datetimeFrom")
            or period.get("datetime_from")
            or row.get("datetimeFrom")
            or row.get("datetime")
        )

        if isinstance(datetime_from, dict):
            datetime_value = datetime_from.get("utc") or datetime_from.get("local")
        else:
            datetime_value = datetime_from

        if datetime_value is None:
            continue

        local_timestamp = normalize_local_timestamp(datetime_value)
        if pd.isna(local_timestamp):
            continue

        try:
            pm25_value = float(value)
        except (TypeError, ValueError):
            continue

        rows.append({"datetime_local": local_timestamp, "pm25": pm25_value})

    if not rows:
        print("No usable OpenAQ measurements.")
        return pd.DataFrame(columns=["datetime_local", "pm25"])

    aqi = pd.DataFrame(rows)

    # Multiple measurements for the same hour: take the average.
    aqi = aqi.groupby("datetime_local", as_index=False)["pm25"].mean()

    aqi = aqi[
        (aqi["datetime_local"] >= pd.Timestamp(start_local))
        & (aqi["datetime_local"] <= pd.Timestamp(end_local))
    ]

    print("OpenAQ usable rows:", len(aqi))
    return aqi.reset_index(drop=True)


# ============================================================
# 16. PM2.5 -> US EPA AQI
# ============================================================

def aqi_from_pm25_value(pm25):
    if pd.isna(pm25):
        return np.nan

    try:
        concentration = float(pm25)
    except (TypeError, ValueError):
        return np.nan

    if concentration < 0:
        return np.nan

    # EPA PM2.5 truncation to one decimal place.
    c = np.floor(concentration * 10) / 10

    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500),
    ]

    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= c <= c_high:
            aqi = ((i_high - i_low) / (c_high - c_low)) * (c - c_low) + i_low
            return round(aqi)

    if c > 500.4:
        return 500

    return np.nan


def calculate_aqi(data):
    data = data.copy()
    data["aqi"] = data["pm25"].apply(aqi_from_pm25_value)
    return data


# ============================================================
# 17. CONNECT HOPSWORKS
# ============================================================

def connect_hopsworks():
    print("\n" + "=" * 70)
    print("CONNECTING TO HOPSWORKS")
    print("=" * 70)

    project = hopsworks.login(project=PROJECT_NAME, api_key_value=HOPSWORKS_API_KEY)
    print("Connected to Hopsworks.")

    fs = project.get_feature_store()
    fg = fs.get_feature_group(name=FEATURE_GROUP_NAME, version=FEATURE_GROUP_VERSION)
    print("Feature Group:", FEATURE_GROUP_NAME, "v", FEATURE_GROUP_VERSION)

    # Upserting only works if the feature group has a primary key.
    # Otherwise fg.insert() will silently append duplicate rows.
    primary_key = getattr(fg, "primary_key", None)
    if not primary_key or EXPECTED_PRIMARY_KEY not in primary_key:
        print()
        print("WARNING: Feature Group primary_key is", primary_key)
        print(
            "         Upsert-based target backfilling requires "
            f"'{EXPECTED_PRIMARY_KEY}' to be the (or part of the) primary key."
        )
        print(
            "         Without it, re-writing existing rows will create "
            "duplicates instead of patching them in place."
        )

    return project, fs, fg


# ============================================================
# 18. READ RECENT HOPSWORKS HISTORY
# ============================================================

def read_recent_history(fg, rows=HISTORY_READ_ROWS):
    print("\n" + "=" * 70)
    print("READING RECENT HOPSWORKS HISTORY")
    print("=" * 70)

    try:
        all_data = fg.read()
    except Exception as exc:
        print("Could not read Feature Group.")
        print("Error:", exc)
        raise

    if all_data is None or all_data.empty:
        return pd.DataFrame()

    data = all_data.copy()

    if "datetime_local" in data.columns:
        data["datetime_local"] = data["datetime_local"].apply(normalize_local_timestamp)

    if "datetime_utc" in data.columns:
        data["datetime_utc"] = data["datetime_utc"].apply(normalize_utc_timestamp)

    for column in ["pm25", "aqi"]:
        if column in data.columns:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    data = (
        data.dropna(subset=["datetime_local"])
        .sort_values("datetime_local")
        .drop_duplicates(subset=["datetime_local"], keep="last")
        .reset_index(drop=True)
    )

    print("Total Hopsworks rows:", len(data))

    recent = data.tail(rows).copy().reset_index(drop=True)
    print("Recent history rows read:", len(recent))

    if not recent.empty:
        print("History first:", recent["datetime_local"].min())
        print("History last:", recent["datetime_local"].max())

    return recent


# ============================================================
# 19. GET LAST TIMESTAMP
# ============================================================

def get_last_timestamp(fg):
    print("\n" + "=" * 70)
    print("FINDING LAST HOPSWORKS TIMESTAMP")
    print("=" * 70)

    try:
        data = fg.read()
    except Exception as exc:
        print("Could not read Hopsworks.")
        raise exc

    if data is None or data.empty or "datetime_local" not in data.columns:
        return None

    timestamps = data["datetime_local"].apply(normalize_local_timestamp).dropna()

    if timestamps.empty:
        return None

    last_timestamp = pd.Timestamp(timestamps.max())
    print("Last Hopsworks timestamp:", last_timestamp)
    return last_timestamp


# ============================================================
# 20. DETERMINE FETCH WINDOW
# ============================================================

def determine_fetch_window(last_timestamp):
    current_hour = get_current_local_hour()

    print()
    print("Current Karachi hour:", current_hour)

    if last_timestamp is None:
        # Safety fallback only (e.g. first ever run).
        start = current_hour - pd.Timedelta(hours=HISTORY_READ_ROWS)
    else:
        start = last_timestamp + pd.Timedelta(hours=1)

    end = current_hour

    if start > end:
        print("\nNo new hour to process.")
        return None, None

    print("Fetch start:", start)
    print("Fetch end:", end)
    return start, end


# ============================================================
# 21. CALENDAR FEATURES
# ============================================================

def create_calendar_features(data):
    data = data.copy()
    dt = pd.to_datetime(data["datetime_local"], errors="coerce")

    data["hour"] = dt.dt.hour.astype("int64")
    data["day"] = dt.dt.day.astype("int64")
    data["month"] = dt.dt.month.astype("int64")
    data["year"] = dt.dt.year.astype("int64")
    data["day_of_week"] = dt.dt.dayofweek.astype("int64")
    data["is_weekend"] = (data["day_of_week"] >= 5).astype("int64")

    return data


# ============================================================
# 22. LAG FEATURES
# ============================================================

def create_lag_features(data):
    data = data.sort_values("datetime_local").reset_index(drop=True).copy()

    for lag in [1, 3, 6, 12, 24]:
        data[f"aqi_lag_{lag}"] = data["aqi"].shift(lag)

    return data


# ============================================================
# 23. ROLLING FEATURES
# ============================================================

def create_rolling_features(data):
    data = data.copy()

    # shift(1) excludes the current AQI value -> no target leakage.
    past_aqi = data["aqi"].shift(1)

    for window in [6, 12, 24]:
        data[f"aqi_mean_{window}"] = past_aqi.rolling(window=window, min_periods=window).mean()
        data[f"aqi_std_{window}"] = past_aqi.rolling(window=window, min_periods=window).std()

    return data


# ============================================================
# 24. TARGET COLUMNS (EXACT TIMESTAMP LOOKUP)
# ============================================================

def create_targets(data):
    data = data.copy()

    # Exact timestamp lookup so target_aqi_24 really means "24 hours
    # later", not "the 24th next row" (which would break across gaps).
    aqi_lookup = data.set_index("datetime_local")["aqi"]

    for horizon in range(1, 73):

        target_column = f"target_aqi_{horizon}"

        data[target_column] = (
            data["aqi"].shift(-horizon)
        )

        print(f"Created: {target_column}")

    return data


# ============================================================
# 25. PREPARE NEW DATA
# ============================================================

def prepare_new_data(weather, aqi):
    print("\n" + "=" * 70)
    print("MERGING WEATHER + OPENAQ")
    print("=" * 70)

    if weather.empty:
        return pd.DataFrame()

    if aqi.empty:
        print("No AQI data available.")
        # We DO NOT create rows with NULL AQI. This is intentional.
        return pd.DataFrame()

    aqi = calculate_aqi(aqi)

    before = len(aqi)
    aqi = aqi[aqi["aqi"].notna()].copy()
    print("OpenAQ rows removed because AQI was NULL:", before - len(aqi))

    if aqi.empty:
        print("No valid AQI rows available.")
        return pd.DataFrame()

    weather["datetime_local"] = pd.to_datetime(weather["datetime_local"], errors="coerce")
    aqi["datetime_local"] = pd.to_datetime(aqi["datetime_local"], errors="coerce")

    merged = pd.merge(
        weather,
        aqi[["datetime_local", "pm25", "aqi"]],
        on="datetime_local",
        how="inner",
    )

    # INNER JOIN -> only hours with valid AQI are kept, so we never
    # create rows with AQI = NULL / lag = NULL / rolling = NULL.
    print("Merged valid AQI rows:", len(merged))

    if merged.empty:
        return pd.DataFrame()

    merged["datetime_utc"] = (
        merged["datetime_local"]
        .dt.tz_localize(TIMEZONE)
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )

    return merged.reset_index(drop=True)


# ============================================================
# 26. COMBINE HISTORY + NEW DATA
# ============================================================

def combine_history_and_new(history, new_data):
    print("\n" + "=" * 70)
    print("COMBINING RECENT HISTORY + NEW DATA")
    print("=" * 70)

    if history.empty:
        combined = new_data.copy()
    elif new_data.empty:
        combined = history.copy()
    else:
        combined = pd.concat([history, new_data], ignore_index=True, sort=False)

    if combined.empty:
        return pd.DataFrame()

    combined["datetime_local"] = combined["datetime_local"].apply(normalize_local_timestamp)
    combined = combined[combined["datetime_local"].notna()].copy()

    # Remove rows without AQI. THIS MUST HAPPEN BEFORE LAG/ROLLING.
    if "aqi" in combined.columns:
        combined["aqi"] = pd.to_numeric(combined["aqi"], errors="coerce")
        before = len(combined)
        combined = combined[combined["aqi"].notna()].copy()
        print("Rows removed because AQI is NULL:", before - len(combined))

    combined = (
        combined.sort_values("datetime_local")
        .drop_duplicates(subset=["datetime_local"], keep="last")
        .reset_index(drop=True)
    )

    print("Combined valid-AQI rows:", len(combined))
    return combined


# ============================================================
# 27. FEATURE ENGINEERING
# ============================================================

def engineer_features(combined):
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)

    if combined.empty:
        return pd.DataFrame()

    data = combined.sort_values("datetime_local").reset_index(drop=True).copy()

    data = create_calendar_features(data)
    data = create_lag_features(data)
    data = create_rolling_features(data)
    data = create_targets(data)

    return data


# ============================================================
# 28. SELECT UPSERT CANDIDATES
# ============================================================
#
# This replaces the old "select_new_rows" logic. Instead of only
# keeping rows inside [start, end] (the truly-new hours), we keep
# every row EXCEPT the first LOOKBACK_BUFFER_ROWS rows of the
# history we read. Those buffer rows exist solely to give lag/
# rolling features something to look back on; their own lag/
# rolling may not have full lookback within our limited read
# window, so we leave them untouched in Hopsworks.
#
# Everything else -- old rows whose targets may have just become
# fillable, plus brand-new rows -- is a genuine upsert candidate.
#
# ============================================================

def select_upsert_rows(engineered, history):
    if engineered.empty:
        return pd.DataFrame()

    if history.empty or len(history) <= LOOKBACK_BUFFER_ROWS:
        cutoff_timestamp = engineered["datetime_local"].min()
    else:
        cutoff_timestamp = history["datetime_local"].iloc[LOOKBACK_BUFFER_ROWS]

    candidates = engineered[engineered["datetime_local"] >= cutoff_timestamp].copy()

    return candidates.sort_values("datetime_local").reset_index(drop=True)


# ============================================================
# 29. CHECK REQUIRED FEATURE NULLS
# ============================================================

def check_required_nulls(data):
    print("\n" + "=" * 70)
    print("NULL CHECK BEFORE UPSERT")
    print("=" * 70)

    if data.empty:
        print("No rows to check.")
        return pd.DataFrame()

    # Targets are excluded on purpose: future AQI legitimately does
    # not exist yet for the most recent rows.
    required_columns = [
        "datetime_local", "datetime_utc",
        "temperature", "humidity", "pressure", "wind_speed",
        "wind_direction", "precipitation", "cloud_cover",
        "pm25", "aqi",
        "hour", "day", "month", "year", "day_of_week", "is_weekend",
        "aqi_lag_1", "aqi_lag_3", "aqi_lag_6", "aqi_lag_12", "aqi_lag_24",
        "aqi_mean_6", "aqi_std_6",
        "aqi_mean_12", "aqi_std_12",
        "aqi_mean_24", "aqi_std_24",
    ]

    existing_required = [c for c in required_columns if c in data.columns]

    null_counts = data[existing_required].isna().sum()
    null_counts = null_counts[null_counts > 0]

    if null_counts.empty:
        print("No NULL values in required columns.")
        return data

    print("\nColumns containing NULL values:")
    print(null_counts)

    before = len(data)
    data = data.dropna(subset=existing_required).copy()
    print("\nRows removed because required features contain NULL:", before - len(data))

    return data.reset_index(drop=True)


# ============================================================
# 30. PREPARE HOPSWORKS TYPES
# ============================================================

def prepare_for_hopsworks(data):
    print("\n" + "=" * 70)
    print("PREPARING HOPSWORKS DATA TYPES")
    print("=" * 70)

    if data.empty:
        return data

    data = data.copy()
    data.replace([np.inf, -np.inf], np.nan, inplace=True)

    for column in FINAL_COLUMNS:
        if column not in data.columns:
            data[column] = np.nan

    data = data[FINAL_COLUMNS].copy()

    data["datetime_local"] = pd.to_datetime(data["datetime_local"], errors="coerce")

    # Hopsworks expects datetime_utc as a string, not a timestamp.
    data["datetime_utc"] = (
        pd.to_datetime(data["datetime_utc"], errors="coerce")
        .dt.strftime("%Y-%m-%d %H:%M:%S")
    )

    bigint_columns = ["hour", "day", "month", "year", "day_of_week", "is_weekend"]
    for column in bigint_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce").astype("int64")

    numeric_columns = [c for c in FINAL_COLUMNS if c not in ["datetime_local", "datetime_utc"] + bigint_columns]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data[data["datetime_local"].notna()].copy()

    data = (
        data.sort_values("datetime_local")
        .drop_duplicates(subset=["datetime_local"], keep="last")
        .reset_index(drop=True)
    )

    print("Prepared rows:", len(data))
    return data


# ============================================================
# 31. FINAL NULL CHECK
# ============================================================

def final_null_check(data):
    print("\n" + "=" * 70)
    print("FINAL NULL CHECK")
    print("=" * 70)

    if data.empty:
        return False

    required_columns = [
        "datetime_local", "datetime_utc",
        "temperature", "humidity", "pressure", "wind_speed",
        "wind_direction", "precipitation", "cloud_cover",
        "pm25", "aqi",
        "hour", "day", "month", "year", "day_of_week", "is_weekend",
        "aqi_lag_1", "aqi_lag_3", "aqi_lag_6", "aqi_lag_12", "aqi_lag_24",
        "aqi_mean_6", "aqi_std_6",
        "aqi_mean_12", "aqi_std_12",
        "aqi_mean_24", "aqi_std_24",
    ]

    null_counts = data[required_columns].isna().sum()
    null_counts = null_counts[null_counts > 0]

    if not null_counts.empty:
        print("NULL values detected:")
        print(null_counts)
        return False

    print("All required feature columns are complete.")

    target_nulls = data[TARGET_COLUMNS].isna().sum()
    target_nulls = target_nulls[target_nulls > 0]

    if not target_nulls.empty:
        print("\nTarget NULLs:")
        print(target_nulls)
        print("\nTarget NULLs are allowed for the most recent rows, since")
        print("future AQI is not yet available for them.")

    return True


# ============================================================
# 32. UPSERT INTO HOPSWORKS
# ============================================================

def upsert_into_hopsworks(fg, data):
    print("\n" + "=" * 70)
    print("UPSERTING INTO HOPSWORKS")
    print("=" * 70)

    if data.empty:
        print("No rows to upsert.")
        return

    print("Rows:", len(data))
    print("First:", data["datetime_local"].min())
    print("Last:", data["datetime_local"].max())

    if not final_null_check(data):
        raise RuntimeError(
            "Upsert stopped because required feature columns contain NULL values."
        )

    # Relies on the Feature Group's primary key (datetime_local) to
    # overwrite existing rows in place rather than duplicating them.
    fg.insert(data, write_options={"wait_for_job": True})

    print("\nHopsworks upsert completed successfully.")


# ============================================================
# 33. DATA QUALITY REPORT
# ============================================================

def print_quality(data):
    print("\n" + "=" * 70)
    print("DATA QUALITY")
    print("=" * 70)

    if data.empty:
        print("No rows.")
        return

    print("Rows:", len(data))
    print("PM2.5 available:", data["pm25"].notna().sum())
    print("AQI available:", data["aqi"].notna().sum())
    print()
    print("Missing PM2.5:", data["pm25"].isna().sum())
    print("Missing AQI:", data["aqi"].isna().sum())
    print()

    for lag in [1, 3, 6, 12, 24]:
        print(f"AQI lag {lag} available:", data[f"aqi_lag_{lag}"].notna().sum())

    print()
    for window in [6, 12, 24]:
        print(f"AQI mean {window} available:", data[f"aqi_mean_{window}"].notna().sum())

    print()
    print("target_aqi_1 NULLs:", data["target_aqi_1"].isna().sum(), "(expect at most 1)")
    print("target_aqi_72 NULLs:", data["target_aqi_72"].isna().sum(), "(expect up to 72)")


# ============================================================
# 34. MAIN PIPELINE
# ============================================================

def main():
    print("\n" + "=" * 70)
    print("KARACHI AQI HOURLY CI/CD PIPELINE (UPSERT DESIGN)")
    print("=" * 70)
    print("Run time:", datetime.now())

    # --------------------------------------------------------
    # CONNECT
    # --------------------------------------------------------
    project, fs, fg = connect_hopsworks()

    # --------------------------------------------------------
    # LAST TIMESTAMP / FETCH WINDOW
    # --------------------------------------------------------
    last_timestamp = get_last_timestamp(fg)
    start, end = determine_fetch_window(last_timestamp)

    if start is None:
        print("\n" + "=" * 70)
        print("NOTHING NEW TO PROCESS")
        print("=" * 70)
        return

    # --------------------------------------------------------
    # READ RECENT HISTORY (CONTEXT FOR LAG/ROLLING/TARGETS)
    # --------------------------------------------------------
    history = read_recent_history(fg, HISTORY_READ_ROWS)

    existing_timestamps = set()
    if not history.empty:
        existing_timestamps = set(history["datetime_local"].dropna().tolist())

    # --------------------------------------------------------
    # FETCH NEW DATA
    # --------------------------------------------------------
    weather = fetch_open_meteo(start, end)
    aqi = fetch_openaq_sensor(start, end)

    new_data = prepare_new_data(weather, aqi)

    if new_data.empty:
        print("\n" + "=" * 70)
        print("NO VALID AQI DATA AVAILABLE")
        print("=" * 70)
        print("\nNothing to upsert this run.")
        print("The missing hour will be attempted again on the next CI/CD run.")
        return

    # --------------------------------------------------------
    # COMBINE HISTORY + NEW
    # --------------------------------------------------------
    combined = combine_history_and_new(history, new_data)

    if combined.empty:
        print("Combined dataset is empty.")
        return

    # --------------------------------------------------------
    # FEATURE ENGINEERING OVER THE FULL COMBINED WINDOW
    # --------------------------------------------------------
    engineered = engineer_features(combined)

    # --------------------------------------------------------
    # SELECT UPSERT CANDIDATES
    # (new rows + existing rows whose targets may have changed)
    # --------------------------------------------------------
    upsert_rows = select_upsert_rows(engineered, history)

    print("\n" + "=" * 70)
    print("UPSERT CANDIDATES BEFORE NULL CHECK:", len(upsert_rows))

    if not upsert_rows.empty:
        new_count = (~upsert_rows["datetime_local"].isin(existing_timestamps)).sum()
        backfill_count = len(upsert_rows) - new_count
        print("  - brand-new rows:", new_count)
        print("  - existing rows eligible for target backfill:", backfill_count)
        print("First:", upsert_rows["datetime_local"].min())
        print("Last:", upsert_rows["datetime_local"].max())

    # --------------------------------------------------------
    # REMOVE ROWS WITH NULL REQUIRED (NON-TARGET) FEATURES
    # --------------------------------------------------------
    upsert_rows = check_required_nulls(upsert_rows)

    if upsert_rows.empty:
        print("\n" + "=" * 70)
        print("NO VALID ROWS REMAIN AFTER NULL CHECK")
        print("=" * 70)
        print("\nNothing upserted.")
        return

    # --------------------------------------------------------
    # QUALITY REPORT
    # --------------------------------------------------------
    print_quality(upsert_rows)

    # --------------------------------------------------------
    # PREPARE TYPES + UPSERT
    # --------------------------------------------------------
    upsert_rows = prepare_for_hopsworks(upsert_rows)
    upsert_into_hopsworks(fg, upsert_rows)

    # --------------------------------------------------------
    # COMPLETE
    # --------------------------------------------------------
    print("\n" + "=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print("Upserted rows:", len(upsert_rows))
    print("Upserted from:", upsert_rows["datetime_local"].min())
    print("Upserted through:", upsert_rows["datetime_local"].max())
    print()
    print("OpenAQ Sensor:", OPENAQ_SENSOR_ID)
    print("OpenAQ Location:", OPENAQ_LOCATION_ID)
    print("Feature Group:", FEATURE_GROUP_NAME, "v", FEATURE_GROUP_VERSION)
    print()
    print("History rows read for context:", HISTORY_READ_ROWS)
    print("Lookback buffer rows (not upserted):", LOOKBACK_BUFFER_ROWS)
    print("=" * 70)


# ============================================================
# 35. ERROR HANDLING
# ============================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nPipeline interrupted.")
        sys.exit(130)
    except Exception as exc:
        print("\n" + "=" * 70)
        print("PIPELINE FAILED")
        print("=" * 70)
        print(type(exc).__name__, ":", exc)
        raise
